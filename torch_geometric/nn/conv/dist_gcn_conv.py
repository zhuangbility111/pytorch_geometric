from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add
# from torch_sparse import SparseTensor, fill_diag, matmul, mul
from torch_sparse import SparseTensor, fill_diag, matmul, mul, matmul_with_cached_transposed
from torch_sparse import sum as sparsesum

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes


@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    # type: (Tensor, OptTensor, Optional[int], bool, bool, Optional[int]) -> PairTensor  # noqa
    pass


@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    # type: (SparseTensor, OptTensor, Optional[int], bool, bool, Optional[int]) -> SparseTensor  # noqa
    pass


def gcn_norm(local_edge_index, remote_edge_index, 
             local_nodes_required_by_other, num_remote_nodes_on_part,
             num_local_nodes, rank, num_part, 
             local_edge_weight=None, remote_edge_weight=None, improved=False,
             add_self_loops=True, dtype=None):

    fill_value = 2. if improved else 1.

    if isinstance(local_edge_index, SparseTensor):
        print("not support yet.")
        '''
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sparsesum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t
        '''
    else:
        # num_local_nodes = maybe_num_nodes(edge_index, num_nodes)
        if local_edge_weight is None:
            local_edge_weight = torch.ones((local_edge_index.size(1), ), dtype=torch.float32,
                                        device=local_edge_index.device)

        if remote_edge_weight is None:
            remote_edge_weight = torch.ones((remote_edge_index.size(1), ), dtype=torch.float32,
                                        device=remote_edge_index.device)

        print("before adding self loops, local_edge_index.size(1) = {}".format(local_edge_index.size(1)))
        # add self loops to the local_edge_index based on the local subgraph
        if add_self_loops:
            local_edge_index, tmp_edge_weight = add_remaining_self_loops(
                    local_edge_index, local_edge_weight, fill_value, num_local_nodes)
            assert tmp_edge_weight is not None
            local_edge_weight = tmp_edge_weight
        print("after adding self loops, local_edge_index.size(1) = {}".format(local_edge_index.size(1)))

        # concatenate local edges and remote edges to a total edges
        # since we have mapped the global id of remote node to the local id on remote_edge_index
        # so we can regard the all_edge_index as the edge index of local subgraph
        all_edge_index = torch.cat((local_edge_index, remote_edge_index), dim=-1)
        all_edge_weight = torch.cat((local_edge_weight, remote_edge_weight), dim=-1)
        print("initial all_edge_weight: , initial all_edge_weight.size(0) = {}".format(all_edge_weight.size(0)))
        print(all_edge_weight)

        # compute in-degree of each local nodes based on local edges and remote edges
        src_nodes, dst_nodes = all_edge_index[0], all_edge_index[1]
        deg = scatter_add(all_edge_weight, dst_nodes, dim=0, dim_size=num_local_nodes)
        print("inital deg: , inital deg.size(0) = {}".format(deg.size(0)))
        print(deg)

        # prepare the in-degree of local nodes required by other subgraphs
        print("local_nodes_required_by_other:")
        print(local_nodes_required_by_other)
        send_degs = [deg.index_select(0, indices) for indices in local_nodes_required_by_other]
        recv_degs = [torch.zeros(num_remote_nodes_on_part[i], dtype=torch.float32) for i in range(num_part)]
        print("send_degs: ")
        print(send_degs)

        # send the in-degree of local node to other subgraphs and recv the in-degree of remote node from other subgraphs
        dist.all_to_all(recv_degs, send_degs)
        print("recv_degs: ")
        print(recv_degs)

        # obtain the in-degree of remote nodes from other subgraphs
        for d in recv_degs:
            deg = torch.cat((deg, d), dim=-1)

        print("after catting deg: , cat_deg.size(0) = {}".format(deg.size(0)))
        print(deg)

        # in the end, compute the weight of each edge (local edges and remote edges)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return all_edge_index, deg_inv_sqrt[src_nodes] * all_edge_weight * deg_inv_sqrt[dst_nodes]

class DistGCNConv(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` tensor.

    Its node-wise formulation is given by:

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}^{\top} \sum_{j \in
        \mathcal{N}(v) \cup \{ i \}} \frac{e_{j,i}}{\sqrt{\hat{d}_j
        \hat{d}_i}} \mathbf{x}_j

    with :math:`\hat{d}_i = 1 + \sum_{j \in \mathcal{N}(i)} e_{j,i}`, where
    :math:`e_{j,i}` denotes the edge weight from source node :obj:`j` to target
    node :obj:`i` (default: :obj:`1.0`)

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops and compute
            symmetric normalization coefficients on the fly.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})`
    """

    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels: int, out_channels: int,
                 local_nodes_required_by_other: Tensor, remote_nodes: Tensor,
                 num_remote_nodes_on_part: Tensor, rank: int, num_part: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        # Parameters regarding to distributed training
        self.local_nodes_required_by_other = local_nodes_required_by_other
        self.remote_nodes = remote_nodes
        self.num_remote_nodes_on_part = num_remote_nodes_on_part
        self.rank = rank
        self.num_part = num_part

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.lin = Linear(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def propagate(self, all_edge_index: Adj, **kwargs):

        # prepare the local nodes' feature which are required by other subgraphs
        all_node_feats = kwargs['x']
        all_edge_weight = kwargs['edge_weight']
        num_local_nodes = all_node_feats.size(0)

        send_node_feats = []
        for indices in self.local_nodes_required_by_other:
            # if the send_node_feats.requires_grad == True
            # the routine dist.all_to_all will incur an error
            # besides, the dist.all_to_all is not a grad_fn
            # so at present we don't consider the gradient passed from remote subgraph
            # it should be fixed in the future
            # here we use the torch.no_grad() to ensure the send_node_feats.requires_grad = False
            # which means that we don't consider the gradient of send_node_feats
            with torch.no_grad():
                temp_node_feats = all_node_feats.index_select(0, indices)
            send_node_feats.append(temp_node_feats)
            # send_node_feats.append(torch.zeros(indices.size(0) * all_node_feats.size(-1), dtype = torch.float32))

        # send the local nodes' feature to other subgraphs and obtain the remote nodes' feature from other subgraphs
        recv_node_feats = [torch.zeros((self.num_remote_nodes_on_part[i], all_node_feats.size(-1)), dtype=torch.float32) for i in range(self.num_part)]
        
        # this routine could be changed to all_to_all_single
        dist.all_to_all(recv_node_feats, send_node_feats)

        # add the remote_node_feats to all_node_feats
        for feats in recv_node_feats:
            all_node_feats = torch.cat((all_node_feats, feats), dim=0)
        
        if (isinstance(all_edge_index, SparseTensor) and self.fuse
                and not self._explain):
            print("not support yet.")
        elif isinstance(all_edge_index, Tensor) or not self.fuse:
            # expand local nodes' features to local edges' features
            src_nodes, dst_nodes = all_edge_index[0], all_edge_index[1]
            all_node_feats_j = all_node_feats.index_select(0, src_nodes)
            # calling message to process the edges' features
            all_edge_feats = self.message(all_node_feats_j, all_edge_weight)

            # aggregate the local and remote neighbors
            out = self.aggregate(all_edge_feats, dst_nodes)
        return out

    def forward(self, x: Tensor, local_edge_index: Adj, remote_edge_index: Adj, 
                local_edge_weight: OptTensor = None, remote_edge_weight: OptTensor = None) -> Tensor:
        """"""
        if self.normalize:
            if isinstance(local_edge_index, Tensor):
                cache_edge_index = self._cached_edge_index
                # cache_deg_inv_sqrt = self._cached_deg_inv_sqrt
                if cache_edge_index is None:
                    # compute the weight of each local edge and the in-degree of each local node
                    all_edge_index, all_edge_weight = gcn_norm(  # yapf: disable
                                        local_edge_index, remote_edge_index, 
                                        self.local_nodes_required_by_other, 
                                        self.num_remote_nodes_on_part,
                                        x.size(0), self.rank, self.num_part, 
                                        local_edge_weight, remote_edge_weight,
                                        self.improved, self.add_self_loops)
                    print(all_edge_index)
                    print(all_edge_weight)
                    if self.cached:
                        self._cached_edge_index = (all_edge_index, all_edge_weight)
                        # self._cached_deg_inv_sqrt = deg_inv_sqrt
                else:
                    all_edge_index, all_edge_weight = cache_edge_index[0], cache_edge_index[1]

            elif isinstance(local_edge_index, SparseTensor):
                print("not support yet.")
                '''
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache
                '''

        # neural operation on nodes
        x = self.lin(x)

        out = self.propagate(all_edge_index, x=x, edge_weight=all_edge_weight, size=None)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        # return matmul(adj_t, x, reduce=self.aggr)
        return matmul_with_cached_transposed(adj_t, x, reduce=self.aggr)
