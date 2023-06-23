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
from torch_geometric.nn.spmm_kernel import SPMM_forward, SPMM_backward

import time

class DistributedGraphPre():
    def __init__(self, local_adj_t: SparseTensor, 
                 adj_t_pre_post_aggr_from: SparseTensor, 
                 adj_t_pre_post_aggr_to: SparseTensor,
                 buf_pre_post_aggr_from: Tensor,
                 buf_pre_post_aggr_to: Tensor,
                 buf_pre_post_aggr_from_fp16: Tensor,
                 buf_pre_post_aggr_to_fp16: Tensor,
                 pre_post_aggr_from_splits: list,
                 pre_post_aggr_to_splits: list,
                 in_degrees: Tensor
                 ) -> None:
        
        self.local_adj_t = local_adj_t
        self.adj_t_pre_post_aggr_from = adj_t_pre_post_aggr_from
        self.adj_t_pre_post_aggr_to = adj_t_pre_post_aggr_to

        # fp32 message buffer
        self.buf_pre_post_aggr_from = buf_pre_post_aggr_from
        self.buf_pre_post_aggr_to = buf_pre_post_aggr_to 
        # fp16 message buffer
        self.buf_pre_post_aggr_from_fp16 = buf_pre_post_aggr_from_fp16
        self.buf_pre_post_aggr_to_fp16 = buf_pre_post_aggr_to_fp16

        self.pre_post_aggr_from_splits = pre_post_aggr_from_splits
        self.pre_post_aggr_to_splits = pre_post_aggr_to_splits
        self.in_degrees = in_degrees

    def resize_buffer(self, size_pre_post_aggr_from: tuple, size_pre_post_aggr_to: tuple) -> None:
        # resize the fp32 message buffer
        self.buf_pre_post_aggr_from.resize_(size_pre_post_aggr_from)
        self.buf_pre_post_aggr_to.resize_(size_pre_post_aggr_to)

        # resize the fp16 message buffer
        if self.buf_pre_post_aggr_from_fp16 is not None and self.buf_pre_post_aggr_to_fp16 is not None:
            self.buf_pre_post_aggr_from_fp16.resize_(size_pre_post_aggr_from)
            self.buf_pre_post_aggr_to_fp16.resize_(size_pre_post_aggr_to)

class DistributedAggregation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, graph: DistributedGraphPre, local_nodes_feat: Tensor):
        ctx.graph = graph

        resize_buffer_begin = time.perf_counter()
        graph.resize_buffer((sum(graph.pre_post_aggr_from_splits), local_nodes_feat.shape[-1]), \
                            (sum(graph.pre_post_aggr_to_splits), local_nodes_feat.shape[-1]))

        create_out_memory_begin = time.perf_counter()
        out = torch.zeros([graph.local_adj_t.sparse_size(0), local_nodes_feat.size(-1)], dtype=torch.float)
        graph.buf_pre_post_aggr_to.zero_()

        pre_aggr_to_begin = time.perf_counter()
        SPMM_forward(graph.adj_t_pre_post_aggr_to, local_nodes_feat, graph.buf_pre_post_aggr_to)

        barrier_begin = time.perf_counter()
        dist.barrier()

        comm_pre_aggr_to_begin = time.perf_counter()
        # communication in fp16
        if graph.buf_pre_post_aggr_from_fp16 is not None and graph.buf_pre_post_aggr_to_fp16 is not None:
            # convert fp32 to fp16
            graph.buf_pre_post_aggr_to_fp16.copy_(graph.buf_pre_post_aggr_to)
            handle = dist.all_to_all_single(graph.buf_pre_post_aggr_from_fp16, graph.buf_pre_post_aggr_to_fp16, \
                                            graph.pre_post_aggr_from_splits, graph.pre_post_aggr_to_splits, async_op=True)
        # communication in fp32
        else:
            handle = dist.all_to_all_single(graph.buf_pre_post_aggr_from, graph.buf_pre_post_aggr_to, \
                                            graph.pre_post_aggr_from_splits, graph.pre_post_aggr_to_splits, async_op=True)

        local_aggr_begin = time.perf_counter()
        SPMM_forward(graph.local_adj_t, local_nodes_feat, out)

        async_wait_begin = time.perf_counter()
        if handle is not None:
            handle.wait()

        if graph.buf_pre_post_aggr_from_fp16 is not None and graph.buf_pre_post_aggr_to_fp16 is not None:
            # recover fp16 to fp32
            graph.buf_pre_post_aggr_from.copy_(graph.buf_pre_post_aggr_from_fp16)

        post_aggr_from_begin = time.perf_counter()
        if graph.buf_pre_post_aggr_from.size(0) > 0:
            SPMM_forward(graph.adj_t_pre_post_aggr_from, graph.buf_pre_post_aggr_from, out)
        post_aggr_from_end = time.perf_counter()
 
        print('$$$$')
        # print("Time of resize buffer(ms): {}".format((create_out_memory_begin - resize_buffer_begin) * 1000.0))
        print("Time of create out memory(ms): {}".format((pre_aggr_to_begin - create_out_memory_begin) * 1000.0))
        print("Time of pre_aggr_to (ms): {}".format((barrier_begin - pre_aggr_to_begin) * 1000.0))
        print("Time of barrier (ms): {}".format((comm_pre_aggr_to_begin - barrier_begin) * 1000.0))
        print("Time of comm pre_aggr_to result (ms): {}".format((local_aggr_begin - comm_pre_aggr_to_begin) * 1000.0))
        print("Time of local aggr (ms): {}".format((async_wait_begin - local_aggr_begin) * 1000.0))
        print("Time of async wait (ms): {}".format((post_aggr_from_begin - async_wait_begin) * 1000.0))
        print("Time of post_aggr_from (ms): {}".format((post_aggr_from_end - post_aggr_from_begin) * 1000.0))
        print('$$$$')

        return out
        
    @staticmethod
    def backward(ctx, local_out_grad):
        graph = ctx.graph

        local_nodes_grad = torch.zeros([graph.local_adj_t.sparse_size(-1), local_out_grad.size(-1)], dtype=torch.float)
        graph.resize_buffer((sum(graph.pre_post_aggr_from_splits), local_out_grad.shape[-1]), \
                            (sum(graph.pre_post_aggr_to_splits), local_out_grad.shape[-1]))

        # 1. collect input's grad (remote nodes' grad) of post-aggregation from
        graph.buf_pre_post_aggr_from.zero_()
        SPMM_backward(graph.adj_t_pre_post_aggr_from, local_out_grad, graph.buf_pre_post_aggr_from)

        if graph.buf_pre_post_aggr_from_fp16 is not None and graph.buf_pre_post_aggr_to_fp16 is not None:
            # convert fp32 to fp16
            graph.buf_pre_post_aggr_from_fp16.copy_(graph.buf_pre_post_aggr_from)
            handle = dist.all_to_all_single(graph.buf_pre_post_aggr_to_fp16, graph.buf_pre_post_aggr_from_fp16, \
                                            graph.pre_post_aggr_to_splits, graph.pre_post_aggr_from_splits, async_op=True)
        else:
            handle = dist.all_to_all_single(graph.buf_pre_post_aggr_to, graph.buf_pre_post_aggr_from, \
                                            graph.pre_post_aggr_to_splits, graph.pre_post_aggr_from_splits, async_op=True)

        # 2.2 collect input's grad (local nodes' grad) of local-aggregation
        SPMM_backward(graph.local_adj_t, local_out_grad, local_nodes_grad)

        if handle is not None:
            handle.wait()

        # recover fp16 to fp32
        if graph.buf_pre_post_aggr_from_fp16 is not None and graph.buf_pre_post_aggr_to_fp16 is not None:
            graph.buf_pre_post_aggr_to.copy_(graph.buf_pre_post_aggr_to_fp16)

        # 4. collect input's grad (local nodes' grad) of pre-aggregation to
        SPMM_backward(graph.adj_t_pre_post_aggr_to, graph.buf_pre_post_aggr_to, local_nodes_grad)

        return None, local_nodes_grad

def aggregate_for_local_and_remote(graph: DistributedGraphPre, local_nodes_feat: Tensor):
    return DistributedAggregation.apply(graph, local_nodes_feat)

class DistSAGEConvGradWithPre(MessagePassing):
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

    def propagate(self, graph: DistributedGraphPre, **kwargs):

        # prepare the local nodes' feature which are required by other subgraphs
        local_nodes_feat = kwargs['x']

        # propagate_begin = time.perf_counter()
        local_out = aggregate_for_local_and_remote(graph, local_nodes_feat)
        # propagate_end = time.perf_counter()
        # print("Time of propagate(inner)(ms) = {}".format((propagate_end - propagate_begin) * 1000.0))

        return local_out

    def forward(self, graph: DistributedGraphPre, x: Tensor) -> Tensor:
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
        # print("Time of norm(ms): {}".format((linear_begin - norm_begin) * 1000.0))
        print("Time of linear(ms): {}".format((propagate_begin -linear_begin) * 1000.0))
        print("Time of propagate(ms): {}".format((add_bias_begin - propagate_begin) * 1000.0))
        # print("Time of add_bias(ms): {}".format((add_bias_end - add_bias_begin) * 1000.0))
        # print("Time of 1 dist conv forward(ms): {}".format((add_bias_end - norm_begin) * 1000.0))
        print("**************")

        return out

