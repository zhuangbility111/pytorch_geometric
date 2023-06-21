from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add
# from torch_sparse import SparseTensor, fill_diag, matmul, mul
from torch_sparse import SparseTensor, fill_diag, mul, spmm_sum_without_backward, matmul
from torch_sparse import sum as sparsesum

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.nn.spmm_kernel import SPMM_forward, SPMM_backward

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
             local_nodes_required_by_other, num_local_nodes_required_by_other, 
             num_remote_nodes_from_part,
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
        local_deg = sparsesum(local_adj_t, dim=1)
        if remote_adj_t.size(0) != 0:
            local_deg += sparsesum(remote_adj_t, dim=1)

        # split local_nodes_required_by_other
        splits_local_nodes_required_by_other = torch.split(local_nodes_required_by_other, num_local_nodes_required_by_other.tolist())

        # prepare the in-degree of local nodes required by other subgraphs
        send_degs = [local_deg.index_select(0, indices) for indices in splits_local_nodes_required_by_other]
        recv_degs = [torch.zeros(num_remote_nodes_from_part[i], dtype=torch.float32) for i in range(num_part)]

        # send the in-degree of local node to other subgraphs and recv the in-degree of remote node from other subgraphs
        dist.all_to_all(recv_degs, send_degs)

        # obtain the in-degree of remote nodes from other subgraphs
        '''
        remote_deg = recv_degs[0]
        for i in range(1, len(recv_degs)):
            remote_deg = torch.cat((remote_deg, recv_degs[i]), dim=-1)
        '''
        remote_deg = torch.cat(recv_degs, dim=-1)

        # process local node's deg
        local_deg_inv_sqrt = local_deg.pow_(-0.5)
        local_deg_inv_sqrt.masked_fill_(local_deg_inv_sqrt == float('inf'), 0.)

        # process remote node's deg
        remote_deg_inv_sqrt = remote_deg.pow_(-0.5)
        remote_deg_inv_sqrt.masked_fill_(remote_deg_inv_sqrt == float('inf'), 0.)
        
        # compute the weight of each local edges
        local_adj_t = mul(local_adj_t, local_deg_inv_sqrt.view(-1, 1))
        local_adj_t = mul(local_adj_t, local_deg_inv_sqrt.view(1, -1))

        # compute the weight of each remote edges 
        if remote_adj_t.size(0) != 0:
            remote_adj_t = mul(remote_adj_t, local_deg_inv_sqrt.view(-1, 1))
            remote_adj_t = mul(remote_adj_t, remote_deg_inv_sqrt.view(1, -1))

        '''
        print("remote_adj_t:")
        print(remote_adj_t.sparse_sizes())
        print(remote_adj_t)
        '''
        # adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        # adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return local_adj_t, remote_adj_t
    
def comm_for_remote_nodes_forward(local_nodes_feat, local_nodes_required_by_other, 
                                  recv_nodes_feat_splits, send_nodes_feat_splits,
                                  recv_nodes_feat_buf, send_nodes_feat_buf):

    prepare_send_node_begin = time.perf_counter()
    # send_nodes_feat_buf = local_nodes_feat.index_select(0, local_nodes_indices_required_by_other)
    torch.index_select(local_nodes_feat, 0, local_nodes_required_by_other, out=send_nodes_feat_buf)

    prepare_recv_node_begin = time.perf_counter()
    # send the local nodes' feature to other subgraphs and obtain the remote nodes' feature from other subgraphs
    # recv_nodes_feat_buf = torch.empty((sum(recv_node_feats_splits), local_nodes_feat.size(-1)), dtype=torch.float32)
    # recv_node_feats = torch.zeros((sum(recv_node_feats_splits), local_nodes_feat.size(-1)), dtype=torch.float32)

    barrier_begin = time.perf_counter()
    # dist.barrier()
    comm_begin = time.perf_counter()
    # handle = dist.all_to_all_single(recv_node_feats, send_node_feats, recv_node_feats_splits, send_node_feats_splits, async_op=True)
    dist.all_to_all_single(recv_nodes_feat_buf, send_nodes_feat_buf, recv_nodes_feat_splits, send_nodes_feat_splits)
    comm_end = time.perf_counter()

    print('$$$$')
    print("Time of prepare send data(ms): {}".format((prepare_recv_node_begin - prepare_send_node_begin) * 1000.0))
    print("Time of prepare recv data(ms): {}".format((barrier_begin - prepare_recv_node_begin) * 1000.0))
    print("Time of barrier (all to all)(ms): {}".format((comm_begin - barrier_begin) * 1000.0))
    print("Time of comm data (all to all)(ms): {}".format((comm_end - comm_begin) * 1000.0))
    print('$$$$')

    # return recv_node_feats, handle
    return None

def comm_for_remote_nodes_backward(recv_nodes_grad_buf, send_nodes_grad_buf,
                                   recv_nodes_grad_splits, send_nodes_grad_splits):
    # prepare the node gradient to send
    # send_nodes_grad = remote_nodes_grad
    
    # allocate memory to save the local node grads from other subgraph
    # recv_node_grads = torch.empty((sum(recv_node_grads_splits), remote_nodes_grad.size(-1)), dtype=torch.float32)

    # handle = dist.all_to_all_single(recv_node_grads, send_node_grads, recv_node_grads_splits, send_node_grads_splits, async_op=True)
    # dist.all_to_all_single(recv_node_grads, send_node_grads, recv_node_grads_splits, send_node_grads_splits)
    dist.all_to_all_single(recv_nodes_grad_buf, send_nodes_grad_buf, recv_nodes_grad_splits, send_nodes_grad_splits)

    # return recv_node_grads, handle
    # return recv_node_grads
    return None

class Aggregate_for_local_and_remote(torch.autograd.Function):
    @staticmethod
    def forward(ctx, local_adj_t, remote_adj_t, local_nodes_feat, 
                local_nodes_required_by_other, num_local_nodes_required_by_other,
                num_remote_nodes_from_part,
                send_nodes_feat_buf, recv_nodes_feat_buf):
        ctx.local_nodes_required_by_other = local_nodes_required_by_other
        ctx.local_adj_t = local_adj_t
        ctx.remote_adj_t = remote_adj_t
    
        prepare_comm_begin = time.perf_counter()
        remote_node_splits = num_remote_nodes_from_part.tolist()
        local_node_splits = num_local_nodes_required_by_other.tolist()
        ctx.remote_node_splits = remote_node_splits
        ctx.local_node_splits = local_node_splits
        ctx.send_nodes_feat_buf = send_nodes_feat_buf
        ctx.recv_nodes_feat_buf = recv_nodes_feat_buf

        num_recv_nodes = sum(remote_node_splits)
        num_send_nodes = sum(local_node_splits)
        send_nodes_feat_buf.resize_(num_send_nodes, local_nodes_feat.size(-1))
        recv_nodes_feat_buf.resize_(num_recv_nodes, local_nodes_feat.size(-1))

        comm_begin = time.perf_counter()
        # communicate for remote nodes feat
        '''
        remote_nodes_feat, comm_handle = comm_for_remote_nodes_forward(local_nodes_feat, 
                                            local_nodes_indices_required_by_other,
                                            remote_node_splits, local_node_splits)
        '''
        '''
        remote_nodes_feat = comm_for_remote_nodes_forward(local_nodes_feat, 
                                            local_nodes_indices_required_by_other,
                                            remote_node_splits, local_node_splits)
        '''
        send_nodes_feat_buf.zero_()
        comm_for_remote_nodes_forward(local_nodes_feat, 
                                      local_nodes_required_by_other,
                                      remote_node_splits, local_node_splits,
                                      recv_nodes_feat_buf, send_nodes_feat_buf)
        
        local_aggregate_begin = time.perf_counter()
        out = torch.zeros([local_adj_t.sparse_size(0), local_nodes_feat.size(-1)], dtype=torch.float)
        # aggregate message from local nodes
        # local_out = SPMM_forward(local_adj_t, local_nodes_feat)
        SPMM_forward(local_adj_t, local_nodes_feat, out)

        async_wait_begin = time.perf_counter()
        # comm_handle.wait()

        remote_aggregate_begin = time.perf_counter()
        remote_nodes_feat = recv_nodes_feat_buf
        # aggregate message from remote nodes
        # remote_out = SPMM_forward(remote_adj_t, remote_nodes_feat)
        SPMM_forward(remote_adj_t, remote_nodes_feat, out)

        sum_message_begin = time.perf_counter()
        # then sum up the message
        '''
        if remote_adj_t.nnz() != 0:
            local_out = local_out + remote_out
        '''

        sum_message_end = time.perf_counter()

        print('#########')
        print("Time of prepare comm_forward(ms): {}".format((comm_begin - prepare_comm_begin) * 1000.0))
        print("Time of comm_forward(ms): {}".format((local_aggregate_begin - comm_begin) * 1000.0))
        print("Time of local aggregate(ms): {}".format((async_wait_begin - local_aggregate_begin) * 1000.0))
        print("Time of async wait(ms): {}".format((remote_aggregate_begin - async_wait_begin) * 1000.0))
        print("Time of remote aggregate(ms): {}".format((sum_message_begin - remote_aggregate_begin) * 1000.0))
        print("Time of sum up message(ms): {}".format((sum_message_end - sum_message_begin) * 1000.0))
        print("Time of 1 dist conv forward(inner)(ms): {}".format((sum_message_end - prepare_comm_begin) * 1000.0))
        print('#########')

        # return local_out
        return out

    @staticmethod
    def backward(ctx, local_out_grad):
        local_nodes_required_by_other = ctx.local_nodes_required_by_other
        remote_node_splits = ctx.remote_node_splits
        local_node_splits = ctx.local_node_splits
        local_adj_t = ctx.local_adj_t
        remote_adj_t = ctx.remote_adj_t

        if ctx.needs_input_grad[2]:
            # scatter gradient to remote nodes
            # remote_nodes_grad = SPMM_backward(remote_adj_t, local_out_grad)
            # remote_nodes_grad = torch.zeros([remote_adj_t.sparse_size(-1), local_out_grad.size(-1)], dtype=torch.float)
            remote_nodes_grad_buf = ctx.recv_nodes_feat_buf
            local_nodes_grad_buf = ctx.send_nodes_feat_buf

            num_send_nodes = sum(remote_node_splits)
            num_recv_nodes = sum(local_node_splits)

            remote_nodes_grad_buf.resize_(num_send_nodes, local_out_grad.size(-1))
            local_nodes_grad_buf.resize_(num_recv_nodes, local_out_grad.size(-1))

            remote_nodes_grad_buf.zero_()
            SPMM_backward(remote_adj_t, local_out_grad, remote_nodes_grad_buf)

            # communicate to obtain the local node grads from other subgraph
            '''
            local_nodes_grad_from, comm_handle = comm_for_remote_nodes_backward(remote_nodes_grad,
                                                                                local_node_splits, remote_node_splits)
            '''
            comm_for_remote_nodes_backward(local_nodes_grad_buf, remote_nodes_grad_buf,
                                           local_node_splits, remote_node_splits)
            
            # scatter gradient to local nodes
            # local_nodes_grad = SPMM_backward(local_adj_t, local_out_grad)
            local_nodes_grad = torch.zeros([local_adj_t.sparse_size(-1), local_out_grad.size(-1)], dtype=torch.float)
            SPMM_backward(local_adj_t, local_out_grad, local_nodes_grad)

            # comm_handle.wait()
            # then accumulate the local node grads
            local_nodes_grad_from = local_nodes_grad_buf
            local_nodes_grad.index_add_(dim=0, index=local_nodes_required_by_other,
                                        source=local_nodes_grad_from)

        return None, None, local_nodes_grad, None, None, None, None, None

def aggregate_for_local_and_remote(local_adj_t, remote_adj_t, local_nodes_feat, 
                local_nodes_required_by_other, num_local_nodes_required_by_other, num_remote_nodes_from_part,
                send_nodes_feat_buf, recv_nodes_feat_buf):
    return Aggregate_for_local_and_remote.apply(local_adj_t, remote_adj_t, local_nodes_feat, 
                local_nodes_required_by_other, num_local_nodes_required_by_other, num_remote_nodes_from_part,
                send_nodes_feat_buf, recv_nodes_feat_buf)

class DistGCNConvGrad(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int,
                 local_nodes_required_by_other: Tensor, num_local_nodes_required_by_other: Tensor,
                 remote_nodes: Tensor,
                 num_remote_nodes_from_part: Tensor, local_range_nodes_on_part: Tensor,
                 rank: int, num_part: int,
                 send_nodes_feat_buf: Tensor,
                 recv_nodes_feat_buf: Tensor,
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
        self.num_local_nodes_required_by_other = num_local_nodes_required_by_other
        self.remote_nodes = remote_nodes
        self.num_remote_nodes_from_part = num_remote_nodes_from_part
        self.local_range_nodes_on_part = local_range_nodes_on_part
        self.rank = rank
        self.num_part = num_part

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.send_nodes_feat_buf = send_nodes_feat_buf
        self.recv_nodes_feat_buf = recv_nodes_feat_buf

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
        local_nodes_feat = kwargs['x']

        if isinstance(all_edge_index, SparseTensor) and not self._explain:
            local_adj_t = all_edge_index
            remote_adj_t = kwargs['remote_edge_index']

            propagate_begin = time.perf_counter()
            local_out = aggregate_for_local_and_remote(local_adj_t, remote_adj_t, local_nodes_feat, 
                                            self.local_nodes_required_by_other, self.num_local_nodes_required_by_other, self.num_remote_nodes_from_part,
                                            self.send_nodes_feat_buf, self.recv_nodes_feat_buf)
            propagate_end = time.perf_counter()
            print("Time of propagate(inner)(ms) = {}".format((propagate_end - propagate_begin) * 1000.0))

            '''
            print("local_out:")
            print(local_out)
            print(local_out.requires_grad)
            print("remote_out:")
            print(remote_out)
            print(remote_out.requires_grad)
            '''
            return local_out

    def forward(self, x: Tensor, local_edge_index: Adj, remote_edge_index: Adj, 
                local_edge_weight: OptTensor = None, remote_edge_weight: OptTensor = None) -> Tensor:
        """"""
        norm_begin = time.perf_counter()
        if self.normalize:
            if isinstance(local_edge_index, SparseTensor):
                cache_adj_t = self._cached_adj_t
                if cache_adj_t is None:
                    local_edge_index, remote_edge_index = gcn_norm(  # yapf: disable
                                        local_edge_index, remote_edge_index, 
                                        self.local_nodes_required_by_other, 
                                        self.num_local_nodes_required_by_other,
                                        self.num_remote_nodes_from_part,
                                        x.size(0), self.rank, self.num_part, 
                                        local_edge_weight, remote_edge_weight,
                                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_adj_t = (local_edge_index, remote_edge_index)
                else:
                    local_edge_index, remote_edge_index = cache_adj_t[0], cache_adj_t[1]

        linear_begin = time.perf_counter()
        # neural operation on nodes
        x = self.lin(x)

        propagate_begin = time.perf_counter()
        if isinstance(local_edge_index, SparseTensor):
            out = self.propagate(local_edge_index, x=x, remote_edge_index=remote_edge_index, size=None)

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
