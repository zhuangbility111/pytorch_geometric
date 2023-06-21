import torch
import torch.distributed as dist
from torch_geometric.nn import DistSAGEConvGrad
from torch_sparse import SparseTensor, fill_diag, matmul, mul, matmul_with_cached_transposed
from torch_geometric.nn.dense.linear import Linear
import time
import numpy as np
import pandas as pd

import os

import argparse

try:
    import torch_ccl
except ImportError as e:
    print(e)


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

def sort_remote_edges_list_based_on_remote_nodes(remote_edges_list):
    remote_edges_row, remote_edges_col = remote_edges_list[0], remote_edges_list[1]
    sort_index = np.argsort(remote_edges_row)
    remote_edges_list[0] = remote_edges_row[sort_index]
    remote_edges_list[1] = remote_edges_col[sort_index]
    return remote_edges_list


def obtain_remote_nodes_list(remote_edges_list, num_local_nodes, num_nodes_on_each_subgraph, world_size):
    remote_nodes_list = []
    range_of_remote_nodes_on_local_graph = torch.zeros(world_size+1, dtype=torch.int64)
    remote_nodes_num_from_each_subgraph = torch.zeros(world_size, dtype=torch.int64)
    remote_edges_row = remote_edges_list[0]

    part_idx = 0
    local_node_idx = num_local_nodes - 1
    prev_node = -1
    tmp_len = len(remote_edges_row)
    for i in range(0, tmp_len):
        cur_node = remote_edges_row[i]
        if cur_node != prev_node:
            remote_nodes_list.append(cur_node)
            local_node_idx += 1
            while cur_node >= num_nodes_on_each_subgraph[part_idx+1]:
                part_idx += 1
                range_of_remote_nodes_on_local_graph[part_idx+1] = range_of_remote_nodes_on_local_graph[part_idx]
            range_of_remote_nodes_on_local_graph[part_idx+1] += 1
            remote_nodes_num_from_each_subgraph[part_idx] += 1
        prev_node = cur_node
        remote_edges_row[i] = local_node_idx

    for i in range(part_idx+1, world_size):
        range_of_remote_nodes_on_local_graph[i+1] = range_of_remote_nodes_on_local_graph[i]

    remote_nodes_list = np.array(remote_nodes_list, dtype=np.int64)
    # print("local remote_nodes_num_from_each_subgraph:")
    # print(remote_nodes_num_from_each_subgraph)

    '''
    # collect communication pattern
    global_list = [torch.zeros(world_size, dtype=torch.int64) for _ in range(world_size)]
    dist.gather(remote_nodes_num_from_each_subgraph, global_list if dist.get_rank() == 0 else None, 0)
    if dist.get_rank() == 0:
        global_comm_tesosr = torch.cat((global_list))

        global_comm_array = global_comm_tesosr.reshape(world_size, world_size).numpy()
        print(global_comm_array)
        np.save('./move_communication_pattern/global_comm_{}.npy'.format(world_size), global_comm_array)
    '''

    return remote_nodes_list, range_of_remote_nodes_on_local_graph, remote_nodes_num_from_each_subgraph

def load_graph_data(dir_path, graph_name, rank, world_size):
    # load vertices on subgraph
    load_nodes_start = time.perf_counter()
    # local_nodes_list = pd.read_csv(os.path.join(dir_path, "p{:0>3d}-{}_nodes.txt".format(rank, graph_name)), sep=" ", header=None, usecols=[0, 3]).values
    local_nodes_list = np.load(os.path.join(dir_path, "p{:0>3d}-{}_nodes.npy".format(rank, graph_name)))
    # print(local_nodes_list.dtype)
    node_idx_begin = local_nodes_list[0][0]
    node_idx_end = local_nodes_list[local_nodes_list.shape[0]-1][0]
    # print("nodes_id_range: {} - {}".format(node_idx_begin, node_idx_end))
    num_local_nodes = node_idx_end - node_idx_begin + 1
    load_nodes_end = time.perf_counter()
    time_load_nodes = load_nodes_end - load_nodes_start

    # load features of vertices on subgraph
    # nodes_feat_list = pd.read_csv(os.path.join(dir_path, "p{:0>3d}-{}_nodes_feat.txt".format(rank, graph_name)), sep=" ", header=None, dtype=np.float32).values
    nodes_feat_list = np.load(os.path.join(dir_path, "p{:0>3d}-{}_nodes_feat.npy".format(rank, graph_name)))
    # print("nodes_feat_list.shape:")
    # print(nodes_feat_list.shape)
    # print(nodes_feat_list.dtype)
    load_nodes_feats_end = time.perf_counter()
    time_load_nodes_feats = load_nodes_feats_end - load_nodes_end

    # load labels of vertices on subgraph
    # nodes_label_list = pd.read_csv(os.path.join(dir_path, "p{:0>3d}-{}_nodes_label.txt".format(rank, graph_name)), sep=" ", header=None).values
    nodes_label_list = np.load(os.path.join(dir_path, "p{:0>3d}-{}_nodes_label.npy".format(rank, graph_name)))
    # print(nodes_label_list.dtype)
    load_nodes_labels_end = time.perf_counter()
    time_load_nodes_labels = load_nodes_labels_end - load_nodes_feats_end

    # load edges on subgraph
    # edges_list = pd.read_csv(os.path.join(dir_path, "p{:0>3d}-{}_edges.txt".format(rank, graph_name)), sep=" ", header=None).values
    # edges_list = np.load(os.path.join(dir_path, "p{:0>3d}-{}_edges.npy".format(rank, graph_name)))
    load_edges_list_end = time.perf_counter()
    time_load_edges_list = load_edges_list_end - load_nodes_labels_end

    # load number of nodes on each subgraph
    num_nodes_on_each_subgraph = np.loadtxt(os.path.join(dir_path, "begin_node_on_each_partition.txt"), dtype='int64', delimiter=' ')
    load_number_nodes_end = time.perf_counter()
    time_load_number_nodes = load_number_nodes_end - load_edges_list_end

    # divide the global edges list into the local edges list and the remote edges list
    # local_edges_list, remote_edges_list = divide_edges_into_local_and_remote(edges_list, node_idx_begin, node_idx_end)
    local_edges_list = np.load(os.path.join(dir_path, "p{:0>3d}-{}_local_edges.npy".format(rank, graph_name)))
    remote_edges_list = np.load(os.path.join(dir_path, "p{:0>3d}-{}_remote_edges.npy".format(rank, graph_name)))
    '''
    print(local_edges_list)
    print(local_edges_list.shape)
    print(local_edges_list.dtype)
    print(remote_edges_list)
    print(remote_edges_list.shape)
    print(remote_edges_list.dtype)
    '''
    # print(local_edges_list)
    divide_edges_list_end = time.perf_counter()
    time_divide_edges_list = divide_edges_list_end - load_number_nodes_end

    # sort remote_edges_list based on the src(remote) nodes' global id
    sort_remote_edges_list_end = time.perf_counter()
    remote_edges_list = sort_remote_edges_list_based_on_remote_nodes(remote_edges_list)
    time_sort_remote_edges_list = sort_remote_edges_list_end - divide_edges_list_end

    # remove duplicated nodes
    # obtain remote nodes list and remap the global id of remote nodes to local id based on their rank
    remote_nodes_list, range_of_remote_nodes_on_local_graph, remote_nodes_num_from_each_subgraph = \
                                obtain_remote_nodes_list(remote_edges_list, num_local_nodes, num_nodes_on_each_subgraph, world_size)
    obtain_remote_nodes_list_end = time.perf_counter()
    time_obtain_remote_nodes_list = obtain_remote_nodes_list_end - sort_remote_edges_list_end

    time_load_and_preprocessing_graph = obtain_remote_nodes_list_end - load_nodes_start

    '''
    print("elapsed time of loading nodes(ms) = {}".format(time_load_nodes * 1000))
    print("elapsed time of loading nodes feats(ms) = {}".format(time_load_nodes_feats * 1000))
    print("elapsed time of loading nodes labels(ms) = {}".format(time_load_nodes_labels * 1000))
    print("elapsed time of loading edges(ms) = {}".format(time_load_edges_list * 1000))
    print("elapsed time of loading number of nodes(ms) = {}".format(time_load_number_nodes * 1000))
    print("elapsed time of dividing edges(ms) = {}".format(time_divide_edges_list * 1000))
    print("elapsed time of sorting edges(ms) = {}".format(time_sort_remote_edges_list * 1000))
    print("elapsed time of obtaining remote nodes(ms) = {}".format(time_obtain_remote_nodes_list * 1000))
    print("elapsed time of whole process of loading graph(ms) = {}".format(time_load_and_preprocessing_graph * 1000))
    print("number of remote nodes = {}".format(remote_nodes_list.shape[0]))
    '''

    return torch.from_numpy(local_nodes_list), torch.from_numpy(nodes_feat_list), \
           torch.from_numpy(nodes_label_list), torch.from_numpy(remote_nodes_list), \
           range_of_remote_nodes_on_local_graph, remote_nodes_num_from_each_subgraph, \
           torch.from_numpy(local_edges_list), torch.from_numpy(remote_edges_list)

def obtain_local_nodes_required_by_other(local_nodes_list, remote_nodes_list, range_of_remote_nodes_on_local_graph, \
                                         remote_nodes_num_from_each_subgraph, world_size):
    # send the number of remote nodes we need to obtain from other subgrpah
    obtain_number_remote_nodes_start = time.perf_counter()
    send_num_nodes = [torch.tensor([remote_nodes_num_from_each_subgraph[i]], dtype=torch.int64) for i in range(world_size)]
    recv_num_nodes = [torch.zeros(1, dtype=torch.int64) for i in range(world_size)]
    dist.all_to_all(recv_num_nodes, send_num_nodes)
    num_local_nodes_required_by_other = recv_num_nodes
    num_local_nodes_required_by_other = torch.cat(num_local_nodes_required_by_other, dim=0)
    obtain_number_remote_nodes_end = time.perf_counter()
    # print("elapsed time of obtaining number of remote nodes(ms) = {}".format( \
    #         (obtain_number_remote_nodes_end - obtain_number_remote_nodes_start) * 1000))

    # then we need to send the nodes_list which include the id of remote nodes we want
    # and receive the nodes_list from other subgraphs
    obtain_remote_nodes_list_start = time.perf_counter()
    send_nodes_list = [remote_nodes_list[range_of_remote_nodes_on_local_graph[i]: \
                       range_of_remote_nodes_on_local_graph[i+1]] for i in range(world_size)]
    recv_nodes_list = [torch.zeros(num_local_nodes_required_by_other[i], dtype=torch.int64) for i in range(world_size)]
    dist.all_to_all(recv_nodes_list, send_nodes_list)
    local_node_idx_begin = local_nodes_list[0][0]
    local_nodes_required_by_other = [i - local_node_idx_begin for i in recv_nodes_list]
    local_nodes_required_by_other = torch.cat(local_nodes_required_by_other, dim=0)
    obtain_remote_nodes_list_end = time.perf_counter()
    # print("elapsed time of obtaining list of remote nodes(ms) = {}".format( \
    #         (obtain_remote_nodes_list_end - obtain_remote_nodes_list_start) * 1000))
    return local_nodes_required_by_other, num_local_nodes_required_by_other

def transform_edge_index_to_sparse_tensor(local_edges_list, remote_edges_list, num_local_nodes, num_remote_nodes):
    local_edges_list = SparseTensor(row=local_edges_list[1], col=local_edges_list[0], value=torch.ones(local_edges_list[1].shape[0], dtype=torch.float32), sparse_sizes=(num_local_nodes, num_local_nodes))
    tmp_col = remote_edges_list[0] - num_local_nodes
    remote_edges_list = SparseTensor(row=remote_edges_list[1], col=tmp_col, value=torch.ones(remote_edges_list[1].shape[0], dtype=torch.float32), sparse_sizes=(num_local_nodes, num_remote_nodes))
    return local_edges_list, remote_edges_list

# run distributed sage conv's propogate function
def run_distributed_propogate(input_dir, graph_name):
    rank, world_size = init_dist_group()
    num_part = world_size
    print("Rank = {}, Number of threads = {}".format(rank, torch.get_num_threads()))

    # obtain graph information
    local_nodes_list, nodes_feat_list, nodes_label_list, remote_nodes_list, \
        range_of_remote_nodes_on_local_graph, remote_nodes_num_from_each_subgraph, \
        local_edges_list, remote_edges_list = load_graph_data(input_dir, graph_name, rank, world_size)

    nodes_feat_list.requires_grad = True

    # obtain the idx of local nodes required by other subgraph
    local_nodes_required_by_other, num_local_nodes_required_by_other = \
        obtain_local_nodes_required_by_other(local_nodes_list, remote_nodes_list, range_of_remote_nodes_on_local_graph, \
                                             remote_nodes_num_from_each_subgraph, world_size)

    # transform the local edges list and remote edges list(both are edge_index) to SparseTensor if it needs
    local_edges_list, remote_edges_list = transform_edge_index_to_sparse_tensor(local_edges_list, remote_edges_list, local_nodes_list.size(0), remote_nodes_list.size(0))
    
    in_channels = nodes_feat_list.size(-1)

    out_channels = 16
    max_feat_len = max(in_channels, out_channels)
    num_send_nodes = local_nodes_required_by_other.size(0)
    send_nodes_feat_buf = torch.zeros((num_send_nodes, max_feat_len), dtype=torch.float32)

    num_recv_nodes = remote_nodes_list.size(0)
    recv_nodes_feat_buf = torch.zeros((num_recv_nodes, max_feat_len), dtype=torch.float32)

    # init distributed sage conv
    sage_conv = DistSAGEConvGrad(in_channels, out_channels,
                                 local_nodes_required_by_other,
                                 num_local_nodes_required_by_other,
                                 remote_nodes_list,
                                 remote_nodes_num_from_each_subgraph,
                                 range_of_remote_nodes_on_local_graph,
                                 rank,
                                 num_part,
                                 send_nodes_feat_buf,
                                 recv_nodes_feat_buf,
                                 None,
                                 None)

    out = sage_conv.propagate(local_edges_list, x=nodes_feat_list, remote_edge_index=remote_edges_list, size=None)
    # print("rank = {}, out = {}".format(rank, out))
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
