from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn import Parameter
# from torch_sparse import SparseTensor, fill_diag, matmul, mul
from torch_sparse import SparseTensor, fill_diag, mul, spmm_sum_without_backward, matmul
from torch_sparse import sum as sparsesum

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes

import time

class DistributedGraph():
    def __init__(self, local_adj_t: SparseTensor, 
                 local_nodes_idx_pre_aggr_from: Tensor, 
                 adj_t_post_aggr_from: SparseTensor, 
                 local_nodes_idx_post_aggr_to: Tensor, 
                 adj_t_pre_aggr_to: SparseTensor,
                 buf_pre_aggr_from: Tensor, 
                 buf_post_aggr_from: Tensor,
                 buf_post_aggr_to: Tensor,
                 buf_pre_aggr_to: Tensor,
                 pre_aggr_from_splits: list,
                 post_aggr_from_splits: list,
                 pre_aggr_to_splits: list, 
                 post_aggr_to_splits: list,
                 in_degrees: Tensor
                 ) -> None:
        
        self.local_adj_t = local_adj_t
        self.local_nodes_idx_pre_aggr_from = local_nodes_idx_pre_aggr_from
        self.adj_t_post_aggr_from = adj_t_post_aggr_from
        self.local_nodes_idx_post_aggr_to = local_nodes_idx_post_aggr_to
        self.adj_t_pre_aggr_to = adj_t_pre_aggr_to
        self.buf_pre_aggr_from = buf_pre_aggr_from
        self.buf_post_aggr_from = buf_post_aggr_from 
        self.buf_post_aggr_to = buf_post_aggr_to
        self.buf_pre_aggr_to = buf_pre_aggr_to
        self.pre_aggr_from_splits = pre_aggr_from_splits
        self.post_aggr_from_splits = post_aggr_from_splits
        self.pre_aggr_to_splits = pre_aggr_to_splits 
        self.post_aggr_to_splits = post_aggr_to_splits
        self.in_degrees = in_degrees

    def resize_buffer(self, size_pre_aggr_from: tuple, \
                      size_post_aggr_from: tuple, \
                      size_post_aggr_to: tuple, \
                      size_pre_aggr_to: tuple
                     ) -> None:
        self.buf_pre_aggr_from.resize_(size_pre_aggr_from)
        self.buf_post_aggr_from.resize_(size_post_aggr_from)
        self.buf_post_aggr_to.resize_(size_post_aggr_to)
        self.buf_pre_aggr_to.resize_(size_pre_aggr_to)

def SPMM_forward(src: SparseTensor, other: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    rowptr, col, value = src.csr()
    if value is not None:
        value = value.to(other.dtype)
    return spmm_sum_without_backward(rowptr, col, value, other, out)

def SPMM_backward(src: SparseTensor, other: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    # rowptr, col, value = src.csr()
    # row = src.storage.row()
    # csr2csc = src.storage.csr2csc()
    colptr = src.storage.colptr()
    # opt_value = value.view(-1, 1).index_select(0, csr2csc).view(-1)
    row_T = src.storage.row_T()
    value_T = src.storage.value_T()
    # return spmm_sum_without_backward(colptr, row.index_select(0, csr2csc), opt_value, other)
    return spmm_sum_without_backward(colptr, row_T, value_T, other, out)

def comm_for_remote_nodes_forward(local_nodes_feat, local_nodes_indices_required_by_other, 
                                  recv_nodes_feat_splits, send_nodes_feat_splits,
                                  recv_nodes_feat_buf, send_nodes_feat_buf):

    prepare_send_node_begin = time.perf_counter()
    # send_nodes_feat_buf = local_nodes_feat.index_select(0, local_nodes_indices_required_by_other)
    torch.index_select(local_nodes_feat, 0, local_nodes_indices_required_by_other, out=send_nodes_feat_buf)

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
    # handle = dist.all_to_all_single(recv_node_grads, send_node_grads, recv_node_grads_splits, send_node_grads_splits, async_op=True)
    # dist.all_to_all_single(recv_node_grads, send_node_grads, recv_node_grads_splits, send_node_grads_splits)
    dist.all_to_all_single(recv_nodes_grad_buf, send_nodes_grad_buf, recv_nodes_grad_splits, send_nodes_grad_splits)

    # return recv_node_grads, handle
    return None

class DistributedAggregation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, graph: DistributedGraph, local_nodes_feat: Tensor):
        ctx.graph = graph
        # ctx.local_nodes_required_by_other = local_nodes_required_by_other
        # ctx.local_adj_t = local_adj_t
        # ctx.remote_adj_t = remote_adj_t

        resize_buffer_begin = time.perf_counter()
        graph.resize_buffer((sum(graph.pre_aggr_from_splits), local_nodes_feat.shape[-1]), \
                            (sum(graph.post_aggr_from_splits), local_nodes_feat.shape[-1]), \
                            (sum(graph.post_aggr_to_splits), local_nodes_feat.shape[-1]), \
                            (sum(graph.pre_aggr_to_splits), local_nodes_feat.shape[-1]))

        create_out_memory_begin = time.perf_counter()
        out = torch.zeros([graph.local_adj_t.sparse_size(0), local_nodes_feat.size(-1)], dtype=torch.float)
        # 1. collect pre_aggregation result for other subgraphs with SPMM
        pre_aggr_to_begin = time.perf_counter()
        graph.buf_pre_aggr_to.zero_()
        SPMM_forward(graph.adj_t_pre_aggr_to, local_nodes_feat, graph.buf_pre_aggr_to)

        # 2. communicate to collect pre-aggregation result from other subgraphs
        comm_pre_aggr_to_begin = time.perf_counter()
        dist.all_to_all_single(graph.buf_pre_aggr_from, graph.buf_pre_aggr_to, \
                               graph.pre_aggr_from_splits, graph.pre_aggr_to_splits)

        # 2.1 collect local aggregation result with SPMM
        local_aggr_begin = time.perf_counter()
        SPMM_forward(graph.local_adj_t, local_nodes_feat, out)

        # 2.2 collect local nodes which will be sent to other subgraphs for post-aggregation
        collect_post_aggr_to_begin = time.perf_counter()
        torch.index_select(local_nodes_feat, 0, graph.local_nodes_idx_post_aggr_to, \
                           out=graph.buf_post_aggr_to)

        # 3. communicate to collect remote nodes which will be used for post-aggregation
        comm_post_aggr_to_begin = time.perf_counter()
        dist.all_to_all_single(graph.buf_post_aggr_from, graph.buf_post_aggr_to, \
                               graph.post_aggr_from_splits, graph.post_aggr_to_splits)

        # 3.1. accumulate the pre-aggregation result sent from other subgraphs
        pre_aggr_from_begin = time.perf_counter()
        out.index_add_(dim=0, index=graph.local_nodes_idx_pre_aggr_from, \
                       source=graph.buf_pre_aggr_from)

        # 4. perform post-aggregation with remote nodes sent from other subgraphs
        post_aggr_from_begin = time.perf_counter()
        SPMM_forward(graph.adj_t_post_aggr_from, graph.buf_post_aggr_from, out)
        post_aggr_from_end = time.perf_counter()
            
        print('$$$$')
        print("Time of resize buffer(ms): {}".format((create_out_memory_begin - resize_buffer_begin) * 1000.0))
        print("Time of create out memory(ms): {}".format((pre_aggr_to_begin - create_out_memory_begin) * 1000.0))
        print("Time of pre_aggr_to (ms): {}".format((comm_pre_aggr_to_begin - pre_aggr_to_begin) * 1000.0))
        print("Time of comm pre_aggr_to result (ms): {}".format((local_aggr_begin - comm_pre_aggr_to_begin) * 1000.0))
        print("Time of local aggr (ms): {}".format((collect_post_aggr_to_begin - local_aggr_begin) * 1000.0))
        print("Time of collect post_aggr_to (ms): {}".format((comm_post_aggr_to_begin - collect_post_aggr_to_begin) * 1000.0))
        print("Time of comm post_aggr_to (ms): {}".format((pre_aggr_from_begin - comm_post_aggr_to_begin) * 1000.0))
        print("Time of pre_aggr_from (ms): {}".format((post_aggr_from_begin - pre_aggr_from_begin) * 1000.0))
        print("Time of post_aggr_from (ms): {}".format((post_aggr_from_end - post_aggr_from_begin) * 1000.0))
        print('$$$$')

        return out
        
    @staticmethod
    def backward(ctx, local_out_grad):
        graph = ctx.graph

        graph.resize_buffer((sum(graph.pre_aggr_from_splits), local_out_grad.shape[-1]), \
                            (sum(graph.post_aggr_from_splits), local_out_grad.shape[-1]), \
                            (sum(graph.post_aggr_to_splits), local_out_grad.shape[-1]), \
                            (sum(graph.pre_aggr_to_splits), local_out_grad.shape[-1]))

        local_nodes_grad = torch.zeros([graph.local_adj_t.sparse_size(-1), local_out_grad.size(-1)], dtype=torch.float)

        # 1. collect input's grad (remote nodes' grad) of post-aggregation from
        graph.buf_post_aggr_from.zero_()
        SPMM_backward(graph.adj_t_post_aggr_from, local_out_grad, graph.buf_post_aggr_from)

        # 2. communicate to send the remote nodes' grad of post-aggregation from
        #    and recv the local nodes' grad of post-aggregation to 
        dist.all_to_all_single(graph.buf_post_aggr_to, graph.buf_post_aggr_from, \
                               graph.post_aggr_to_splits, graph.post_aggr_from_splits)

        # 2.1 collect input's grad (remote node's grad) of pre-aggregation from
        torch.index_select(local_out_grad, 0, graph.local_nodes_idx_pre_aggr_from, \
                           out=graph.buf_pre_aggr_from)

        # 2.2 collect input's grad (local nodes' grad) of local-aggregation
        SPMM_backward(graph.local_adj_t, local_out_grad, local_nodes_grad)

        # 3. communicate to send the remote node's grad of pre-aggregation from
        #    and recv local node's grad of pre-aggregation to
        dist.all_to_all_single(graph.buf_pre_aggr_to, graph.buf_pre_aggr_from, \
                               graph.pre_aggr_to_splits, graph.pre_aggr_from_splits)

        # 3.1 collect input's grad (local nodes' grad) of post-aggregation to
        local_nodes_grad.index_add_(dim=0, index=graph.local_nodes_idx_post_aggr_to, \
                                    source=graph.buf_post_aggr_to)

        # 4. collect input's grad (local nodes' grad) of pre-aggregation to
        SPMM_backward(graph.adj_t_pre_aggr_to, graph.buf_pre_aggr_to, local_nodes_grad)

        return None, local_nodes_grad

def aggregate_for_local_and_remote(graph: DistributedGraph, local_nodes_feat: Tensor):
    return DistributedAggregation.apply(graph, local_nodes_feat)

class DistSAGEConvGradWithPrePost(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int,
                 add_self_loops: bool = False, normalize: bool = True,
                 bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.local_deg = None

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

    def propagate(self, graph: DistributedGraph, **kwargs):

        # prepare the local nodes' feature which are required by other subgraphs
        local_nodes_feat = kwargs['x']

        propagate_begin = time.perf_counter()
        local_out = aggregate_for_local_and_remote(graph, local_nodes_feat)
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

    def forward(self, graph: DistributedGraph, x: Tensor) -> Tensor:
        """"""
        norm_begin = time.perf_counter()
        linear_begin = time.perf_counter()
        # neural operation on nodes
        x = self.lin(x)

        propagate_begin = time.perf_counter()
        # if isinstance(local_edge_index, SparseTensor):
        out = self.propagate(graph, x=x)

        add_bias_begin = time.perf_counter()
        out += x
        out /= (graph.in_degrees + 1)
        if self.bias is not None:
            out += self.bias

        add_bias_end = time.perf_counter()

        print("**************")
        print("Time of norm(ms): {}".format((linear_begin - norm_begin) * 1000.0))
        print("Time of linear(ms): {}".format((propagate_begin -linear_begin) * 1000.0))
        print("Time of propagate(ms): {}".format((add_bias_begin - propagate_begin) * 1000.0))
        print("Time of add_bias(ms): {}".format((add_bias_end - add_bias_begin) * 1000.0))
        print("Time of 1 dist conv forward(ms): {}".format((add_bias_end - norm_begin) * 1000.0))
        print("**************")

        return out

