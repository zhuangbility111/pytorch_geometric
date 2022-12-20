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

import time


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
        local_adj_t = local_edge_index
        remote_adj_t = remote_edge_index
        # print("before fill_value, the shape of local_adj_t:")
        # print(local_adj_t.sizes())
        if not local_adj_t.has_value():
            local_adj_t = local_adj_t.fill_value(1., dtype=dtype)

        if not remote_adj_t.has_value():
            remote_adj_t = remote_adj_t.fill_value(1., dtype=dtype)

        # print("before add_self_loops, the shape of local_adj_t:")
        # print(local_adj_t.sizes())
        if add_self_loops:
            local_adj_t = fill_diag(local_adj_t, fill_value)
        
        # print("after add_self_loops, the shape of local_adj_t:")
        # print(local_adj_t.sizes())
        # compute in-degree of each local nodes based on local edges and remote edges
        deg = sparsesum(local_adj_t, dim=1)
        if remote_adj_t.size(0) != 0:
            deg += sparsesum(remote_adj_t, dim=1)

        # prepare the in-degree of local nodes required by other subgraphs
        send_degs = [deg.index_select(0, indices) for indices in local_nodes_required_by_other]
        recv_degs = [torch.zeros(num_remote_nodes_on_part[i], dtype=torch.float32) for i in range(num_part)]

        # send the in-degree of local node to other subgraphs and recv the in-degree of remote node from other subgraphs
        dist.all_to_all(recv_degs, send_degs)

        # obtain the in-degree of remote nodes from other subgraphs
        for d in recv_degs:
            deg = torch.cat((deg, d), dim=-1)

        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)

        # compute the weight of each local edges
        local_edge_inv_sqrt = deg_inv_sqrt[:num_local_nodes]
        local_adj_t = mul(local_adj_t, local_edge_inv_sqrt.view(-1, 1))
        local_adj_t = mul(local_adj_t, deg_inv_sqrt.view(1, -1))

        # compute the weight of each remote edges 
        if remote_adj_t.size(0) != 0:
            remote_adj_t = mul(remote_adj_t, local_edge_inv_sqrt.view(-1, 1))
            remote_adj_t = mul(remote_adj_t, deg_inv_sqrt.view(1, -1))
        # adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        # adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return local_adj_t, remote_adj_t
    else:
        # num_local_nodes = maybe_num_nodes(edge_index, num_nodes)
        if local_edge_weight is None:
            local_edge_weight = torch.ones((local_edge_index.size(1), ), dtype=torch.float32,
                                        device=local_edge_index.device)

        if remote_edge_weight is None:
            remote_edge_weight = torch.ones((remote_edge_index.size(1), ), dtype=torch.float32,
                                        device=remote_edge_index.device)

        # add self loops to the local_edge_index based on the local subgraph
        if add_self_loops:
            local_edge_index, tmp_edge_weight = add_remaining_self_loops(
                    local_edge_index, local_edge_weight, fill_value, num_local_nodes)
            assert tmp_edge_weight is not None
            local_edge_weight = tmp_edge_weight

        # concatenate local edges and remote edges to a total edges
        # since we have mapped the global id of remote node to the local id on remote_edge_index
        # so we can regard the all_edge_index as the edge index of local subgraph
        all_edge_index = torch.cat((local_edge_index, remote_edge_index), dim=-1)
        all_edge_weight = torch.cat((local_edge_weight, remote_edge_weight), dim=-1)

        # compute in-degree of each local nodes based on local edges and remote edges
        src_nodes, dst_nodes = all_edge_index[0], all_edge_index[1]
        deg = scatter_add(all_edge_weight, dst_nodes, dim=0, dim_size=num_local_nodes)

        # prepare the in-degree of local nodes required by other subgraphs
        send_degs = [deg.index_select(0, indices) for indices in local_nodes_required_by_other]
        recv_degs = [torch.zeros(num_remote_nodes_on_part[i], dtype=torch.float32) for i in range(num_part)]

        # send the in-degree of local node to other subgraphs and recv the in-degree of remote node from other subgraphs
        dist.all_to_all(recv_degs, send_degs)

        # obtain the in-degree of remote nodes from other subgraphs
        for d in recv_degs:
            deg = torch.cat((deg, d), dim=-1)

        # in the end, compute the weight of each edge (local edges and remote edges)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        return all_edge_index, deg_inv_sqrt[src_nodes] * all_edge_weight * deg_inv_sqrt[dst_nodes]

class Communicate_for_remote_feats(torch.autograd.Function):
    '''
    @staticmethod
    def forward(ctx, local_node_feats, local_nodes_required_by_other, num_remote_nodes_on_part, local_range_nodes_on_part):
        num_local_nodes = local_node_feats.size(0)
        ctx.local_nodes_required_by_other = local_nodes_required_by_other
        ctx.num_remote_nodes_on_part = num_remote_nodes_on_part
        ctx.local_range_nodes_on_part = local_range_nodes_on_part
        ctx.num_local_nodes = num_local_nodes
        num_part = num_remote_nodes_on_part.size(0)
        ctx.num_part = num_part

        prepare_send_node_begin = time.perf_counter()
        # prepare node feats to send
        send_node_feats = []
        for indices in local_nodes_required_by_other:
            temp_node_feats = local_node_feats.index_select(0, indices)
            send_node_feats.append(temp_node_feats)
        
        prepare_recv_node_begin = time.perf_counter()
        # send the local nodes' feature to other subgraphs and obtain the remote nodes' feature from other subgraphs
        recv_node_feats = [torch.zeros((num_remote_nodes_on_part[i], local_node_feats.size(-1)), dtype=torch.float32) for i in range(num_part)]


        comm_begin = time.perf_counter()
        dist.all_to_all(recv_node_feats, send_node_feats)
        all_node_feats = local_node_feats

        concatenate_begin = time.perf_counter()
        # add the remote_node_feats to all_node_feats
        for feats in recv_node_feats:
            all_node_feats = torch.cat((all_node_feats, feats), dim=0)

        concatenate_end = time.perf_counter()
        print('$$$$')
        print("Time of prepare send data(ms): {}".format((prepare_recv_node_begin - prepare_send_node_begin) * 1000.0))
        print("Time of prepare recv data(ms): {}".format((comm_begin - prepare_recv_node_begin) * 1000.0))
        print("Time of comm data (all to all)(ms): {}".format((concatenate_begin - comm_begin) * 1000.0))
        print("Time of concatenate data(ms): {}".format((concatenate_end - concatenate_begin) * 1000.0))
        print('$$$$')

        return all_node_feats
    '''

    @staticmethod
    def forward(ctx, local_node_feats, local_nodes_required_by_other, num_remote_nodes_on_part, local_range_nodes_on_part):
        num_local_nodes = local_node_feats.size(0)
        ctx.local_nodes_required_by_other = local_nodes_required_by_other
        ctx.num_remote_nodes_on_part = num_remote_nodes_on_part
        ctx.local_range_nodes_on_part = local_range_nodes_on_part
        ctx.num_local_nodes = num_local_nodes
        num_part = num_remote_nodes_on_part.size(0)
        ctx.num_part = num_part

        prepare_send_node_begin = time.perf_counter()
        local_nodes_indices_required_by_other = local_nodes_required_by_other[0]
        # prepare node feats to send
        for i in range(1, len(local_nodes_required_by_other)):
            local_nodes_indices_required_by_other = torch.cat((local_nodes_indices_required_by_other, local_nodes_required_by_other[i]), dim=0)
        send_node_feats = local_node_feats.index_select(0, local_nodes_indices_required_by_other)
        # send_node_feats = torch.ones((local_nodes_indices_required_by_other.size(0), local_node_feats.size(-1)))
        send_node_feats_splits = [indices.size(0) for indices in local_nodes_required_by_other]
        
        prepare_recv_node_begin = time.perf_counter()
        # send the local nodes' feature to other subgraphs and obtain the remote nodes' feature from other subgraphs
        # recv_node_feats = [torch.zeros((num_remote_nodes_on_part[i], local_node_feats.size(-1)), dtype=torch.float32) for i in range(num_part)]
        num_remote_nodes = sum(num_remote_nodes_on_part)
        recv_node_feats = torch.empty((num_remote_nodes, local_node_feats.size(-1)), dtype=torch.float32)
        recv_node_feats_splits = num_remote_nodes_on_part.tolist()

        comm_begin = time.perf_counter()
        # dist.all_to_all(recv_node_feats, send_node_feats)
        dist.all_to_all_single(recv_node_feats, send_node_feats, recv_node_feats_splits, send_node_feats_splits)
        all_node_feats = local_node_feats

        concatenate_begin = time.perf_counter()
        # add the remote_node_feats to all_node_feats
        all_node_feats = torch.cat((all_node_feats, recv_node_feats), dim=0)

        concatenate_end = time.perf_counter()
        print('$$$$')
        print("Time of prepare send data(ms): {}".format((prepare_recv_node_begin - prepare_send_node_begin) * 1000.0))
        print("Time of prepare recv data(ms): {}".format((comm_begin - prepare_recv_node_begin) * 1000.0))
        print("Time of comm data (all to all)(ms): {}".format((concatenate_begin - comm_begin) * 1000.0))
        print("Time of concatenate data(ms): {}".format((concatenate_end - concatenate_begin) * 1000.0))
        print('$$$$')

        return all_node_feats

    def backward(ctx, grad_output):
        local_nodes_required_by_other = ctx.local_nodes_required_by_other
        num_remote_nodes_on_part = ctx.num_remote_nodes_on_part
        local_range_nodes_on_part = ctx.local_range_nodes_on_part
        num_local_nodes = ctx.num_local_nodes
        num_part = ctx.num_part
        grad_input = None
        if ctx.needs_input_grad[0]:
            # print("before processing grad_output:")
            # print(grad_output.shape)
            # print(grad_output)
            grad_input = grad_output
            print("grad_input:")
            print(grad_input.shape)
            print(grad_input)
            # prepare the node gradient to send
            send_node_grads = [grad_input[(num_local_nodes + local_range_nodes_on_part[i]):
                                    (num_local_nodes + local_range_nodes_on_part[i+1])] for i in range(num_part)]

            # allocate memory to save the local node grads from other subgraph
            recv_node_grads = [torch.zeros((indices.size(0), grad_output.size(-1)), dtype=torch.float32) for indices in local_nodes_required_by_other]

            # communicate to obtain the local node grads from other subgraph
            dist.all_to_all(recv_node_grads, send_node_grads)

            grad_input = grad_input[:num_local_nodes]

            # then accumulate the local node grads
            for i in range(len(local_nodes_required_by_other)): 
                indices = local_nodes_required_by_other[i]
                if indices.size(0) != 0:
                    grad_input.index_add_(dim=0, index=indices, source=recv_node_grads[i])

            # print("after processing grad_input:")
            # print(grad_input.shape)
            # print(grad_input)

        return grad_input, None, None, None

def comm_for_remote_feats(local_node_feats, local_nodes_required_by_other, 
                            num_remote_nodes_on_part, local_range_nodes_on_part):
    # need to decide the input parameters
    return Communicate_for_remote_feats.apply(local_node_feats, local_nodes_required_by_other, 
                                                num_remote_nodes_on_part, local_range_nodes_on_part)

'''
class Aggregate_for_sparse_tensor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, local_adj_t, remote_adj_t, all_node_feats):
        # local aggregation
        local_out = matmul(local_adj_t, all_node_feats, reduce=aggr)
        # remote aggregation
        remote_out = matmul(remote_adj_t, all_node_feats, reduce=aggr)

        # combine the result of local aggregation and remote aggregation
        if remote_out.size(0) != 0:
            return local_out + remote_out

        return local_out

    @staticmethod
    def backward(ctx, grad_output):
        # some backward funtion
        

def Aggregate_for_sparse_tensor():
    return Aggregate_for_sparse_tensor.apply()
'''

class DistGCNConvGrad(MessagePassing):
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
    _cached_adj_t: Optional[Tuple[SparseTensor, SparseTensor]]

    def __init__(self, in_channels: int, out_channels: int,
                 local_nodes_required_by_other: Tensor, remote_nodes: Tensor,
                 num_remote_nodes_on_part: Tensor, local_range_nodes_on_part: Tensor,
                 rank: int, num_part: int,
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
        self.local_range_nodes_on_part = local_range_nodes_on_part
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
        local_node_feats = kwargs['x']
        num_local_nodes = local_node_feats.size(0)

        '''
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
        '''
        comm_for_remote_feats_begin = time.perf_counter()
        all_node_feats = comm_for_remote_feats(local_node_feats, self.local_nodes_required_by_other,
                                               self.num_remote_nodes_on_part, self.local_range_nodes_on_part)

        if isinstance(all_edge_index, SparseTensor) and not self._explain:
            local_adj_t = all_edge_index
            remote_adj_t = kwargs['remote_edge_index']

            local_aggregate_begin = time.perf_counter()
            # local aggregation
            local_out = self.message_and_aggregate(local_adj_t, all_node_feats)

            remote_aggregate_begin = time.perf_counter()
            # remote aggregation
            remote_out = self.message_and_aggregate(remote_adj_t, all_node_feats)

            if remote_adj_t.nnz() != 0:
                combine_result_begin = time.perf_counter()
                tmp_out = local_out + remote_out
                combine_result_end = time.perf_counter()
                print("##########")
                print("Time of Communicate_for_remote_feats(ms): {}".format((local_aggregate_begin - comm_for_remote_feats_begin) * 1000.0))
                print("Time of local aggregation(ms): {}".format((remote_aggregate_begin - local_aggregate_begin) * 1000.0))
                print("Time of remote aggregation(ms): {}".format((combine_result_begin - remote_aggregate_begin) * 1000.0))
                # print("Time of combine result(ms): {}".format((combine_result_end - combine_result_begin) * 1000.0))
                print('##########')
                return tmp_out
            else:
                return local_out

        elif isinstance(all_edge_index, Tensor):
            all_edge_weight = kwargs['edge_weight']
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
        norm_begin = time.perf_counter()
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
                    # print(all_edge_index)
                    # print(all_edge_weight)
                    if self.cached:
                        self._cached_edge_index = (all_edge_index, all_edge_weight)
                        # self._cached_deg_inv_sqrt = deg_inv_sqrt
                else:
                    all_edge_index, all_edge_weight = cache_edge_index[0], cache_edge_index[1]

            elif isinstance(local_edge_index, SparseTensor):
                cache_adj_t = self._cached_adj_t
                if cache_adj_t is None:
                    local_edge_index, remote_edge_index = gcn_norm(  # yapf: disable
                                        local_edge_index, remote_edge_index, 
                                        self.local_nodes_required_by_other, 
                                        self.num_remote_nodes_on_part,
                                        x.size(0), self.rank, self.num_part, 
                                        local_edge_weight, remote_edge_weight,
                                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_adj_t = (local_edge_index, remote_edge_index)
                else:
                    local_edge_index, remote_edge_index = cache_adj_t[0], cache_adj_t[1]
                '''
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache
                '''

        linear_begin = time.perf_counter()

        # neural operation on nodes
        x = self.lin(x)

        propagate_begin = time.perf_counter()

        if isinstance(local_edge_index, Tensor):
            out = self.propagate(all_edge_index, x=x, edge_weight=all_edge_weight, size=None)
        elif isinstance(local_edge_index, SparseTensor):
            out = self.propagate(local_edge_index, x=x, remote_edge_index=remote_edge_index, size=None)

        # print(out.grad_fn)
        add_bias_begin = time.perf_counter()

        if self.bias is not None:
            out += self.bias

        add_bias_end = time.perf_counter()
        print("**************")
        # print("Time of norm(ms): {}".format((linear_begin - norm_begin) * 1000.0))
        print("Time of linear(ms): {}".format((propagate_begin -linear_begin) * 1000.0))
        print("Time of propagate(ms): {}".format((add_bias_begin - propagate_begin) * 1000.0))
        # print("Time of add_bias(ms): {}".format((add_bias_end - add_bias_begin) * 1000.0))
        print("Time of 1 dist conv forward(ms): {}".format((add_bias_end - norm_begin) * 1000.0))
        print("**************")

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)
        # return matmul_with_cached_transposed(adj_t, x, reduce=self.aggr)
