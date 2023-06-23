import torch
import torch.distributed as dist
from torch_geometric.nn import DistSAGEConvGrad
from torch_geometric.nn import DistributedGraphPre
from torch_geometric.nn import DistSAGEConvGradWithPre
from torch_sparse import SparseTensor, fill_diag, matmul, mul, matmul_with_cached_transposed
from torch_geometric.nn.dense.linear import Linear
import time
import numpy as np
import pandas as pd
import gc

import os

import argparse

try:
    import torch_ccl
except ImportError as e:
    print(e)

def create_comm_buffer(in_channels, hidden_channels, out_channels, num_send_nodes, num_recv_nodes, is_fp16=False):
    max_feat_len = max(in_channels, hidden_channels, out_channels)

    send_nodes_feat_buf = torch.zeros((num_send_nodes, max_feat_len), dtype=torch.float32)
    send_nodes_feat_buf_fp16 = None
    if is_fp16:
        send_nodes_feat_buf_fp16 = torch.zeros((num_send_nodes, max_feat_len), dtype=torch.float16)

    recv_nodes_feat_buf = torch.zeros((num_recv_nodes, max_feat_len), dtype=torch.float32)
    recv_nodes_feat_buf_fp16 = None
    if is_fp16:
        recv_nodes_feat_buf_fp16 = torch.zeros((num_recv_nodes, max_feat_len), dtype=torch.float16)

    return send_nodes_feat_buf, send_nodes_feat_buf_fp16, recv_nodes_feat_buf, recv_nodes_feat_buf_fp16

def init_dist_group():
    if dist.is_mpi_available():
        print("mpi in torch.distributed is available!")
        dist.init_process_group(backend="mpi")
    else:
        world_size = int(os.environ.get("PMI_SIZE", -1))
        rank = int(os.environ.get("PMI_RANK", -1))
        # print("PMI_SIZE = {}".format(world_size))
        # print("PMI_RANK = {}".format(rank))
        # print("use ccl backend for torch.distributed package on x86 cpu.")
        dist_url = "env://"
        dist.init_process_group(backend="ccl", init_method="env://", 
                                world_size=world_size, rank=rank)
    assert torch.distributed.is_initialized()
    # print(f"dist_info RANK: {dist.get_rank()}, SIZE: {dist.get_world_size()}")
    # number of process in this MPI group
    world_size = dist.get_world_size() 
    # mpi rank in this MPI group
    rank = dist.get_rank()
    return (rank, world_size)

# to remap the nodes id in remote_nodes_list to local nodes id (from 0)
# the remote nodes list must be ordered
def remap_remote_nodes_id(remote_nodes_list, begin_node_on_each_partition):
    local_node_idx = -1
    for rank in range(begin_node_on_each_partition.shape[0]-1):
        prev_node = -1
        num_nodes = begin_node_on_each_partition[rank+1] - begin_node_on_each_partition[rank]
        begin_idx = begin_node_on_each_partition[rank]
        for i in range(num_nodes):
            # Attention !!! remote_nodes_list[i] must be transformed to scalar !!!
            cur_node = remote_nodes_list[begin_idx+i].item()
            if cur_node != prev_node:
                local_node_idx += 1
            prev_node = cur_node
            remote_nodes_list[begin_idx+i] = local_node_idx
    return local_node_idx + 1

def load_graph_data(dir_path, graph_name, rank, world_size):
    # load vertices on subgraph
    load_nodes_start = time.perf_counter()
    local_nodes_list = np.load(os.path.join(dir_path, "p{:0>3d}-{}_nodes.npy".format(rank, graph_name)))
    node_idx_begin = local_nodes_list[0][0]
    node_idx_end = local_nodes_list[local_nodes_list.shape[0]-1][0]
    print("nodes_id_range: {} - {}".format(node_idx_begin, node_idx_end))
    num_local_nodes = node_idx_end - node_idx_begin + 1
    load_nodes_end = time.perf_counter()
    time_load_nodes = load_nodes_end - load_nodes_start

    # ----------------------------------------------------------

    # load features of vertices on subgraph
    # code for loading features is moved to the location before the training loop for saving memory
    # nodes_feat_list = np.load(os.path.join(dir_path, "p{:0>3d}-{}_nodes_feat.npy".format(rank, graph_name)))
    # nodes_feat_list = np.array([0,1,2], dtype=np.int64)
    # print("nodes_feat_list.shape:")
    # print(nodes_feat_list.shape)
    # print(nodes_feat_list.dtype)
    load_nodes_feats_end = time.perf_counter()
    time_load_nodes_feats = load_nodes_feats_end - load_nodes_end

    # ----------------------------------------------------------

    # load labels of vertices on subgraph
    nodes_label_list = np.load(os.path.join(dir_path, "p{:0>3d}-{}_nodes_label.npy".format(rank, graph_name)))
    print(nodes_label_list.dtype)
    load_nodes_labels_end = time.perf_counter()
    time_load_nodes_labels = load_nodes_labels_end - load_nodes_feats_end

    # ----------------------------------------------------------

    # load number of nodes on each subgraph
    begin_node_on_each_subgraph = np.loadtxt(os.path.join(dir_path, "begin_node_on_each_partition.txt"), dtype=np.int64, delimiter=' ')
    load_number_nodes_end = time.perf_counter()
    time_load_number_nodes = load_number_nodes_end - load_nodes_labels_end

    # ----------------------------------------------------------

    # divide the global edges list into the local edges list and the remote edges list
    local_edges_list = np.load(os.path.join(dir_path, "p{:0>3d}-{}_local_edges.npy".format(rank, graph_name)))
    remote_edges_list = np.load(os.path.join(dir_path, "p{:0>3d}-{}_remote_edges.npy".format(rank, graph_name)))

    # local nodes in local_edges_list and remote_edges_list has been localized
    # in order to perform pre_aggregation, the id of local nodes in remote_edges_list must be recover to global id
    remote_edges_list[1] += node_idx_begin

    print(local_edges_list)
    print(local_edges_list.shape)
    print(remote_edges_list)
    print(remote_edges_list.shape)
    divide_edges_list_end = time.perf_counter()
    time_divide_edges_list = divide_edges_list_end - load_number_nodes_end

    # ----------------------------------------------------------

    # sort remote_edges_list based on the src(remote) nodes' global id
    # sort_remote_edges_list_end = time.perf_counter()
    # remote_edges_list = sort_remote_edges_list_based_on_remote_nodes(remote_edges_list)
    # time_sort_remote_edges_list = sort_remote_edges_list_end - divide_edges_list_end

    # ----------------------------------------------------------

    # remove duplicated nodes
    # obtain remote nodes list and remap the global id of remote nodes to local id based on their rank
    # remote_nodes_list, range_of_remote_nodes_on_local_graph, remote_nodes_num_from_each_subgraph = \
    #                             obtain_remote_nodes_list(remote_edges_list, num_local_nodes, begin_node_on_each_subgraph, world_size)
    obtain_remote_nodes_list_end = time.perf_counter()
    time_obtain_remote_nodes_list = obtain_remote_nodes_list_end - divide_edges_list_end

    # ----------------------------------------------------------

    time_load_and_preprocessing_graph = obtain_remote_nodes_list_end - load_nodes_start

    print("elapsed time of loading nodes(ms) = {}".format(time_load_nodes * 1000))
    print("elapsed time of loading nodes feats(ms) = {}".format(time_load_nodes_feats * 1000))
    print("elapsed time of loading nodes labels(ms) = {}".format(time_load_nodes_labels * 1000))
    print("elapsed time of loading number of nodes(ms) = {}".format(time_load_number_nodes * 1000))
    print("elapsed time of dividing edges(ms) = {}".format(time_divide_edges_list * 1000))
    # print("elapsed time of sorting edges(ms) = {}".format(time_sort_remote_edges_list * 1000))
    print("elapsed time of obtaining remote nodes(ms) = {}".format(time_obtain_remote_nodes_list * 1000))
    print("elapsed time of whole process of loading graph(ms) = {}".format(time_load_and_preprocessing_graph * 1000))
    # print("number of remote nodes = {}".format(remote_nodes_list.shape[0]))

    return torch.from_numpy(local_nodes_list), torch.from_numpy(nodes_label_list), \
            torch.from_numpy(local_edges_list), torch.from_numpy(remote_edges_list), \
            torch.from_numpy(begin_node_on_each_subgraph)

def process_remote_edges_pre_post_aggr_to(is_pre_post_aggr_to, remote_edges_pre_post_aggr_to, world_size):
    remote_edges_list_pre_post_aggr_to = [torch.empty((0), dtype=torch.int64), torch.empty((0), dtype=torch.int64)]
    begin_edge_on_each_partition_to = torch.zeros(world_size+1, dtype=torch.int64)
    pre_aggr_to_splits = []
    post_aggr_to_splits = []
    for part_id in range(world_size):
        # post-aggregate
        if is_pre_post_aggr_to[part_id][0].item() == 0:
            # collect the number of local nodes current MPI rank needs
            post_aggr_to_splits.append(is_pre_post_aggr_to[part_id][2].item())
            pre_aggr_to_splits.append(0)
            # collect the local node id required by other MPI ranks, group them to edges list in which they will point to themselves
            remote_edges_list_pre_post_aggr_to[0] = torch.cat((remote_edges_list_pre_post_aggr_to[0], \
                                                               remote_edges_pre_post_aggr_to[part_id]), \
                                                               dim=0)
            remote_edges_list_pre_post_aggr_to[1] = torch.cat((remote_edges_list_pre_post_aggr_to[1], \
                                                               remote_edges_pre_post_aggr_to[part_id]), \
                                                               dim=0)
            # collect it for remapping nodes id
            begin_edge_on_each_partition_to[part_id+1] = begin_edge_on_each_partition_to[part_id] + remote_edges_pre_post_aggr_to[part_id].shape[0]
        # pre_aggregate
        else:
            # collect the number of post remote nodes current MPI rank needs
            pre_aggr_to_splits.append(is_pre_post_aggr_to[part_id][2].item())
            post_aggr_to_splits.append(0)
            # collect the subgraph sent from other MPI ranks for pre-aggregation
            num_remote_edges = int(is_pre_post_aggr_to[part_id][1].item() / 2)
            src_in_remote_edges = remote_edges_pre_post_aggr_to[part_id][:num_remote_edges]
            dst_in_remote_edges = remote_edges_pre_post_aggr_to[part_id][num_remote_edges:]
            
            # sort the remote edges based on the remote nodes (dst nodes)
            sort_index = torch.argsort(dst_in_remote_edges)
            remote_edges_list_pre_post_aggr_to[0] = torch.cat((remote_edges_list_pre_post_aggr_to[0], \
                                                               src_in_remote_edges[sort_index]), \
                                                               dim=0)
            remote_edges_list_pre_post_aggr_to[1] = torch.cat((remote_edges_list_pre_post_aggr_to[1], \
                                                               dst_in_remote_edges[sort_index]), \
                                                               dim=0)
            # collect it for remapping nodes id
            begin_edge_on_each_partition_to[part_id+1] = begin_edge_on_each_partition_to[part_id] + dst_in_remote_edges.shape[0]

        begin_edge_on_each_partition_to[world_size] = remote_edges_list_pre_post_aggr_to[0].shape[0]

    return remote_edges_list_pre_post_aggr_to, begin_edge_on_each_partition_to, \
           post_aggr_to_splits, pre_aggr_to_splits

def divide_remote_edges_list(begin_node_on_each_subgraph, remote_edges_list, world_size):
    is_pre_post_aggr_from = [torch.zeros((3), dtype=torch.int64) for i in range(world_size)]
    remote_edges_pre_post_aggr_from = []
    # remote_edges_list_post_aggr_from = [[], []]
    # local_nodes_idx_pre_aggr_from = []
    remote_edges_list_pre_post_aggr_from = [torch.empty((0), dtype=torch.int64), torch.empty((0), dtype=torch.int64)]
    begin_edge_on_each_partition_from = torch.zeros(world_size+1, dtype=torch.int64)
    remote_nodes_num_from_each_subgraph = torch.zeros(world_size, dtype=torch.int64)
    pre_aggr_from_splits = []
    post_aggr_from_splits = []
    num_diff_nodes = 0
    for i in range(world_size):
        # set the begin node idx and end node idx on current rank i
        begin_idx = begin_node_on_each_subgraph[i]
        end_idx = begin_node_on_each_subgraph[i+1]
        print("begin_idx = {}, end_idx = {}".format(begin_idx, end_idx))
        
        src_in_remote_edges = remote_edges_list[0]
        dst_in_remote_edges = remote_edges_list[1]

        # get the remote edges which are from current rank i
        edge_idx = ((src_in_remote_edges >= begin_idx) & (src_in_remote_edges < end_idx))
        src_in_remote_edges = src_in_remote_edges[edge_idx]
        dst_in_remote_edges = dst_in_remote_edges[edge_idx]

        # to get the number of remote nodes and local nodes to determine this rank is pre_aggr or post_aggr
        ids_src_nodes = torch.unique(src_in_remote_edges, sorted=True)
        ids_dst_nodes = torch.unique(dst_in_remote_edges, sorted=True)

        num_src_nodes = ids_src_nodes.shape[0]
        num_dst_nodes = ids_dst_nodes.shape[0]

        # print("total number of remote src nodes = {}".format(num_src_nodes))
        # print("total number of remote dst nodes = {}".format(num_dst_nodes))

        # accumulate the differences of remote src nodes and local dst nodes
        num_diff_nodes += abs(num_src_nodes - num_dst_nodes)
        remote_nodes_num_from_each_subgraph[i] = min(num_src_nodes, num_dst_nodes)

        # when the number of remote src_nodes > the number of local dst_nodes
        # pre_aggr is necessary to decrease the volumn of communication 
        # so pre_aggr  --> pre_post_aggr_from = 1 --> send the remote edges to src mpi rank
        #    post_aggr --> pre_post_aggr_from = 0 --> send the idx of src nodes to src mpi rank
        if num_src_nodes > num_dst_nodes:
            # pre_aggr
            # collect graph structure and send them to other MPI ransk to perform pre-aggregation
            tmp = torch.cat((src_in_remote_edges, \
                             dst_in_remote_edges), \
                             dim=0)
            remote_edges_pre_post_aggr_from.append(tmp)
            is_pre_post_aggr_from[i][0] = 1
            # number of remote edges = is_pre_post_aggr_from[i][1] / 2
            is_pre_post_aggr_from[i][1] = tmp.shape[0]
            # push the number of remote nodes current MPI rank needs
            is_pre_post_aggr_from[i][2] = ids_dst_nodes.shape[0]
            # collect number of nodes sent from other subgraphs for all_to_all_single
            pre_aggr_from_splits.append(ids_dst_nodes.shape[0])
            post_aggr_from_splits.append(0)
            # collect local node id sent from other MPI ranks, group them to edges list in which they will point to themselves
            remote_edges_list_pre_post_aggr_from[0] = torch.cat((remote_edges_list_pre_post_aggr_from[0], \
                                                                 ids_dst_nodes), \
                                                                 dim=0)
            remote_edges_list_pre_post_aggr_from[1] = torch.cat((remote_edges_list_pre_post_aggr_from[1], \
                                                                 ids_dst_nodes), \
                                                                 dim=0)
            # collect it for remapping nodes id
            begin_edge_on_each_partition_from[i+1] = begin_edge_on_each_partition_from[i] + ids_dst_nodes.shape[0]
        else:
            # post_aggr
            is_pre_post_aggr_from[i][0] = 0
            is_pre_post_aggr_from[i][1] = num_src_nodes
            # push the number of remote nodes current MPI rank needs
            is_pre_post_aggr_from[i][2] = ids_src_nodes.shape[0]
            # collect remote node id sent from other MPI ranks to notify other MPI ranks
            # which nodes current MPI rank needs
            remote_edges_pre_post_aggr_from.append(ids_src_nodes)
            # collect number of nodes sent from other subgraphs for all_to_all_single
            post_aggr_from_splits.append(ids_src_nodes.shape[0])
            pre_aggr_from_splits.append(0)

            # sort remote edges based on the remote nodes (src nodes)
            sort_index = torch.argsort(src_in_remote_edges)

            # collect remote edges for aggregation with SPMM later
            remote_edges_list_pre_post_aggr_from[0] = torch.cat((remote_edges_list_pre_post_aggr_from[0], \
                                                                 src_in_remote_edges[sort_index]), \
                                                                 dim=0)
            remote_edges_list_pre_post_aggr_from[1] = torch.cat((remote_edges_list_pre_post_aggr_from[1], \
                                                                 dst_in_remote_edges[sort_index]), \
                                                                 dim=0)
            # collect it for remapping nodes id
            begin_edge_on_each_partition_from[i+1] = begin_edge_on_each_partition_from[i] + src_in_remote_edges.shape[0]

    begin_edge_on_each_partition_from[world_size] = remote_edges_list_pre_post_aggr_from[0].shape[0]
    print("num_diff_nodes = {}".format(num_diff_nodes))

    # communicate with other mpi ranks to get the status of pre_aggr or post_aggr 
    # and number of remote edges(pre_aggr) or remote src nodes(post_aggr)
    is_pre_post_aggr_to = [torch.zeros((3), dtype=torch.int64) for i in range(world_size)]
    dist.all_to_all(is_pre_post_aggr_to, is_pre_post_aggr_from)

    # communicate with other mpi ranks to get the remote edges(pre_aggr) 
    # or remote src nodes(post_aggr)
    remote_edges_pre_post_aggr_to = [torch.empty((indices[1]), dtype=torch.int64) for indices in is_pre_post_aggr_to]
    dist.all_to_all(remote_edges_pre_post_aggr_to, remote_edges_pre_post_aggr_from)

    remote_edges_list_pre_post_aggr_to, begin_edge_on_each_partition_to, \
    post_aggr_to_splits, pre_aggr_to_splits = \
        process_remote_edges_pre_post_aggr_to(is_pre_post_aggr_to, remote_edges_pre_post_aggr_to, world_size)

    del is_pre_post_aggr_from
    del is_pre_post_aggr_to
    del remote_edges_pre_post_aggr_from
    del remote_edges_pre_post_aggr_to

    '''
    # collect communication pattern
    global_list = [torch.zeros(world_size, dtype=torch.int64) for _ in range(world_size)]
    dist.gather(remote_nodes_num_from_each_subgraph, global_list if dist.get_rank() == 0 else None, 0)
    if dist.get_rank() == 0:
        global_comm_tensor = torch.cat((global_list))
        global_comm_array = global_comm_tesosr.reshape(world_size, world_size).numpy()
        print(global_comm_array)
        np.save('./move_communication_pattern/global_comm_{}.npy'.format(world_size), global_comm_array)
    '''
    '''
    print("local_nodes_idx_pre_post, remote_edges_list_pre_post_aggr:")
    print(local_nodes_idx_pre_aggr_from)
    print(local_nodes_idx_post_aggr_to)
    print(remote_edges_list_post_aggr_from)
    print(remote_edges_list_pre_aggr_to)
    '''

    return remote_edges_list_pre_post_aggr_from, remote_edges_list_pre_post_aggr_to, \
        begin_edge_on_each_partition_from, begin_edge_on_each_partition_to, \
        pre_aggr_from_splits, post_aggr_from_splits, \
        post_aggr_to_splits, pre_aggr_to_splits

def transform_edge_index_to_sparse_tensor(local_edges_list, \
                                          remote_edges_list_pre_post_aggr_from, \
                                          remote_edges_list_pre_post_aggr_to, \
                                          begin_edge_on_each_partition_from, \
                                          begin_edge_on_each_partition_to, \
                                          num_local_nodes, \
                                          local_node_begin_idx):
    # construct local sparse tensor for local aggregation
    # localize nodes
    # local_edges_list[0] -= local_node_begin_idx
    # local_edges_list[1] -= local_node_begin_idx

    # local_edges_list has been localized
    local_adj_t = SparseTensor(row=local_edges_list[1], \
                               col=local_edges_list[0], \
                               value=None, \
                               sparse_sizes=(num_local_nodes, num_local_nodes))

    del local_edges_list
    gc.collect()

    # ----------------------------------------------------------

    print("-----before remote_edges_list_pre_post_aggr_from[0]:-----")
    print(remote_edges_list_pre_post_aggr_from[0])
    print("-----before remote_edges_list_pre_post_aggr_from[1]:-----")
    print(remote_edges_list_pre_post_aggr_from[1])
    # localize the dst nodes id (local nodes id)
    remote_edges_list_pre_post_aggr_from[1] -= local_node_begin_idx
    # remap (localize) the sorted src nodes id (remote nodes id) for construction of SparseTensor
    num_remote_nodes_from = remap_remote_nodes_id(remote_edges_list_pre_post_aggr_from[0], begin_edge_on_each_partition_from)

    print("-----after remote_edges_list_pre_post_aggr_from[0]:-----")
    print(remote_edges_list_pre_post_aggr_from[0])
    print("-----after remote_edges_list_pre_post_aggr_from[1]:-----")
    print(remote_edges_list_pre_post_aggr_from[1])

    adj_t_pre_post_aggr_from = SparseTensor(row=remote_edges_list_pre_post_aggr_from[1], \
                                            col=remote_edges_list_pre_post_aggr_from[0], \
                                            value=None, \
                                            sparse_sizes=(num_local_nodes, num_remote_nodes_from))
    
    del remote_edges_list_pre_post_aggr_from
    del begin_edge_on_each_partition_from
    gc.collect()

    # ----------------------------------------------------------

    print("-----before remote_edges_list_pre_post_aggr_to[0]:-----")
    print(remote_edges_list_pre_post_aggr_to[0])
    print("-----before remote_edges_list_pre_post_aggr_to[1]:-----")
    print(remote_edges_list_pre_post_aggr_to[1])
    # localize the src nodes id (local nodes id)
    remote_edges_list_pre_post_aggr_to[0] -= local_node_begin_idx
    # remap (localize) the sorted dst nodes id (remote nodes id) for construction of SparseTensor
    num_remote_nodes_to = remap_remote_nodes_id(remote_edges_list_pre_post_aggr_to[1], begin_edge_on_each_partition_to)

    print("-----after remote_edges_list_pre_aggr_to[0]:-----")
    print(remote_edges_list_pre_post_aggr_to[0])
    print("-----after remote_edges_list_pre_aggr_to[1]:-----")
    print(remote_edges_list_pre_post_aggr_to[1])

    adj_t_pre_post_aggr_to = SparseTensor(row=remote_edges_list_pre_post_aggr_to[1], \
                                          col=remote_edges_list_pre_post_aggr_to[0], \
                                          value=None, \
                                          sparse_sizes=(num_remote_nodes_to, num_local_nodes))
    del remote_edges_list_pre_post_aggr_to
    del begin_edge_on_each_partition_to
    gc.collect()
    # ----------------------------------------------------------

    return local_adj_t, adj_t_pre_post_aggr_from, adj_t_pre_post_aggr_to

def init_adj_t(graph: DistributedGraphPre):
    if isinstance(graph.local_adj_t, SparseTensor) and \
      not graph.local_adj_t.has_value():
        graph.local_adj_t.fill_value_(1.)

    if isinstance(graph.adj_t_pre_post_aggr_from, SparseTensor) and \
      not graph.adj_t_pre_post_aggr_from.has_value():
        graph.adj_t_pre_post_aggr_from.fill_value_(1.)

    if isinstance(graph.adj_t_pre_post_aggr_to, SparseTensor) and \
      not graph.adj_t_pre_post_aggr_to.has_value():
        graph.adj_t_pre_post_aggr_to.fill_value_(1.)

def get_in_degrees(local_edges_list, remote_edges_list, num_local_nodes, begin_idx_local_nodes):
    local_degs = torch.zeros((num_local_nodes), dtype=torch.int64)
    source = torch.ones((local_edges_list[1].shape[0]), dtype=torch.int64)
    # tmp_index = local_edges_list[1] - begin_idx_local_nodes
    # local_degs.index_add_(dim=0, index=tmp_index, source=source)
    local_degs.index_add_(dim=0, index=local_edges_list[1], source=source)
    source = torch.ones((remote_edges_list[1].shape[0]), dtype=torch.int64)
    tmp_index = remote_edges_list[1] - begin_idx_local_nodes
    local_degs.index_add_(dim=0, index=tmp_index, source=source)
    return local_degs.unsqueeze(-1)

def run_distributed_propogate(input_dir, graph_name):
    rank, world_size = init_dist_group()
    num_part = world_size
    is_fp16 = False
    print("Rank = {}, Number of threads = {}".format(rank, torch.get_num_threads()))

    # obtain graph information
    local_nodes_list, nodes_label_list, \
    local_edges_list, remote_edges_list, begin_node_on_each_subgraph = \
        load_graph_data(input_dir,
                        graph_name, 
                        rank, 
                        world_size)

    num_local_nodes = local_nodes_list.shape[0]
    local_in_degrees = get_in_degrees(local_edges_list, remote_edges_list, \
                                      num_local_nodes, begin_node_on_each_subgraph[rank])

    divide_remote_edges_begin = time.perf_counter()
    remote_edges_list_pre_post_aggr_from, remote_edges_list_pre_post_aggr_to, \
    begin_edge_on_each_partition_from, begin_edge_on_each_partition_to, \
    pre_aggr_from_splits, post_aggr_from_splits, \
    post_aggr_to_splits, pre_aggr_to_splits = \
        divide_remote_edges_list(begin_node_on_each_subgraph, \
                                 remote_edges_list, \
                                 world_size)

    divide_remote_edges_end = time.perf_counter()

    print("elapsed time of dividing remote edges(ms) = {}".format( \
            (divide_remote_edges_end - divide_remote_edges_begin) * 1000))

    pre_post_aggr_from_splits = []
    pre_post_aggr_to_splits = []
    for i in range(world_size):
        pre_post_aggr_from_splits.append(pre_aggr_from_splits[i] + post_aggr_from_splits[i])
        pre_post_aggr_to_splits.append(pre_aggr_to_splits[i] + post_aggr_to_splits[i])
    transform_remote_edges_begin = time.perf_counter()
    local_adj_t, adj_t_pre_post_aggr_from, adj_t_pre_post_aggr_to = \
        transform_edge_index_to_sparse_tensor(local_edges_list, \
                                              remote_edges_list_pre_post_aggr_from, \
                                              remote_edges_list_pre_post_aggr_to, \
                                              begin_edge_on_each_partition_from, \
                                              begin_edge_on_each_partition_to, \
                                              num_local_nodes, \
                                              begin_node_on_each_subgraph[rank])
    transform_remote_edges_end = time.perf_counter()
    print("elapsed time of transforming remote edges(ms) = {}".format( \
            (transform_remote_edges_end - transform_remote_edges_begin) * 1000))

    del local_nodes_list
    del local_edges_list
    del remote_edges_list
    del remote_edges_list_pre_post_aggr_from
    del remote_edges_list_pre_post_aggr_to
    gc.collect()

    # load features
    nodes_feat_list = np.load(os.path.join(input_dir, "p{:0>3d}-{}_nodes_feat.npy".format(rank, graph_name)))
    nodes_feat_list = torch.from_numpy(nodes_feat_list)
    # print("nodes_feat_list.shape:")
    # print(nodes_feat_list.shape)
    # print(nodes_feat_list.dtype)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    in_channels = nodes_feat_list.shape[1]
    nodes_feat_list.requires_grad = True

    hidden_channels = 16
    out_channels = 16

    buf_pre_post_aggr_to, buf_pre_post_aggr_to_fp16, \
    buf_pre_post_aggr_from, buf_pre_post_aggr_from_fp16 = \
        create_comm_buffer(in_channels, hidden_channels, out_channels, \
                           sum(pre_post_aggr_to_splits), sum(pre_post_aggr_from_splits), \
                           is_fp16)

    g = DistributedGraphPre(local_adj_t, \
                            adj_t_pre_post_aggr_from, \
                            adj_t_pre_post_aggr_to, \
                            buf_pre_post_aggr_from, \
                            buf_pre_post_aggr_to, \
                            buf_pre_post_aggr_from_fp16, \
                            buf_pre_post_aggr_to_fp16, \
                            pre_post_aggr_from_splits, \
                            pre_post_aggr_to_splits, \
                            local_in_degrees)

    init_adj_t(g)

    sage_conv = DistSAGEConvGradWithPre(in_channels, out_channels)
    out = sage_conv.propagate(g, x=nodes_feat_list)

    return nodes_feat_list, out, rank, world_size


def run_local_propogate(input_dir, graph_name):
    # load data 
    edges_list = pd.read_csv(os.path.join(input_dir, "..", "{}_edges.txt".format(graph_name)), sep=" ", header=None).values
    nodes_list = pd.read_csv(os.path.join(input_dir, "..", "{}_nodes.txt".format(graph_name)), sep=" ", header=None, usecols=[2]).values
    # use numpy to load nodes feat with faster speed
    nodes_feat_list = torch.from_numpy(np.load(os.path.join(input_dir, "..", "{}_nodes_feat.npy".format(graph_name))).astype(np.float32))
    nodes_feat_list.requires_grad = True

    num_nodes = nodes_list.shape[0]

    # construct sparse tensor
    adj_t = SparseTensor(row=torch.from_numpy(edges_list[:, 1]), col=torch.from_numpy(edges_list[:, 0]), \
                         value=torch.ones(edges_list[:, 1].shape[0], dtype=torch.float32), \
                         sparse_sizes=(num_nodes, num_nodes))
    
    del nodes_list
    del edges_list

    # run sage conv's propogate function
    aggr = "sum"
    out = matmul(adj_t, nodes_feat_list, reduce=aggr)
    # print("local propogate out = {}".format(out))
    return nodes_feat_list, out 

def check_input(local_feats, global_feats, cur_nodes_list, rank, rtol=1e-05, atol=1e-08):
    # compare the nodes feat list in each rank with the initial nodes feat list
    is_close = torch.allclose(local_feats[cur_nodes_list[:, 0]], global_feats[cur_nodes_list[:, 1]], rtol=rtol, atol=atol)
    return is_close

def check_output(out, ref_out, cur_nodes_list, rank, rtol=1e-05, atol=1e-08):
    print("rank = {}, out = {}".format(rank, out))
    print("rank = {}, ref_out = {}".format(rank, ref_out))
    print("rank = {}, cur_nodes_list = {}".format(rank, cur_nodes_list))
    is_close = torch.allclose(out[cur_nodes_list[:, 0]], ref_out[cur_nodes_list[:, 1]], atol=atol, rtol=rtol)
    # count the number of elements in out which are greater than 1e-5
    num_out_greater_than_1e_5 = torch.sum(torch.abs(out[cur_nodes_list[:, 0]]) > 1e-5)
    print("rank = {}, num_out_greater_than_1e_5 = {}".format(rank, num_out_greater_than_1e_5))

    # print the not equal elements
    if not is_close:
        torch.set_printoptions(precision=10)
        # print the index of not equal elements
        idx_of_diff = torch.where(torch.abs(out[cur_nodes_list[:, 0]] - ref_out[cur_nodes_list[:, 1]]) > \
                                 (atol + rtol * torch.abs(ref_out[cur_nodes_list[:, 1]])))
        # print("rank = {}, torch.where(torch.abs(out[cur_nodes_list[:, 0]] - ref_out[cur_nodes_list[:, 1]]) > (atol + rtol * torch.abs(ref_out[cur_nodes_list[:, 1]]))) = {}" \
        #       .format(rank, idx_of_diff))
        # print the value of not equal elements
        print("rank = {}, out[cur_nodes_list[:, 0]][idx_of_diff] = {}".format(rank, out[cur_nodes_list[:, 0]][idx_of_diff]))
        print("rank = {}, ref_out[cur_nodes_list[:, 1]][idx_of_diff] = {}".format(rank, ref_out[cur_nodes_list[:, 1]][idx_of_diff]))

    return is_close

    
# test the correctness of distributed sage conv's propogate function
def test_distributed_sage_conv_grad(input_dir, graph_name):
    print("run distributed sage conv's propogate function")
    local_feats, out, rank, world_size = run_distributed_propogate(input_dir, graph_name)
    print("rank = {}, finish run distributed sage conv's propogate function".format(rank))

    print("run local sage conv's propogate function")
    global_feats, ref_out = run_local_propogate(input_dir, graph_name)
    print("rank = {}, finish run local sage conv's propogate function".format(rank))

    # load the mapping between global id and local id
    cur_nodes_list = np.load(os.path.join(input_dir, "p{:0>3d}-{}_nodes.npy".format(rank, graph_name)))
    # localize the id in cur nodes list
    cur_nodes_list[:, 0] -= cur_nodes_list[0, 0]

    # check nodes feat list
    if check_input(local_feats, global_feats, cur_nodes_list, rank):
        print("rank = {}, check input passed".format(rank))
    else:
        print("rank = {}, check input failed".format(rank))

    # check output
    rtol = 1e-04 # relative tolerance
    atol = 1e-04 # absolute tolerance
    if check_output(out, ref_out, cur_nodes_list, rank, rtol, atol):
        print("rank = {}, rtol = {}, atol = {}, check output passed".format(rank, rtol, atol))
    else:
        print("rank = {}, rtol = {}, atol = {}, check output failed".format(rank, rtol, atol))

    # ------ test backward ------
    print("run distributed sage conv's backward function")
    grad_out = torch.ones_like(ref_out)
    ref_out.backward(grad_out)
    # expected_grad_value = value.grad
    # value.grad = None
    expected_grad_other = global_feats.grad
    print("rank = {}, expected_grad_other = {}".format(rank, expected_grad_other))

    grad_out = torch.ones_like(out)
    out.backward(grad_out)
    grad_other = local_feats.grad
    print("rank = {}, grad_other = {}".format(rank, grad_other))

    if check_output(grad_other, expected_grad_other, cur_nodes_list, rank, rtol, atol):
        print("rank = {}, rtol = {}, atol = {}, check output's gradient passed".format(rank, rtol, atol))
    else:
        print("rank = {}, rtol = {}, atol = {}, check output's gradient failed".format(rank, rtol, atol))


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='test distributed sage conv')
    parser.add_argument('--input_dir', type=str, help='input directory')
    parser.add_argument('--graph_name', type=str, help='graph name')

    args = parser.parse_args()
    input_dir = args.input_dir
    graph_name = args.graph_name

    test_distributed_sage_conv_grad(input_dir, graph_name)
