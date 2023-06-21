import torch
import torch.distributed as dist
import pandas as pd
from torch_geometric.nn import DistSAGEConvGrad
import time
import numpy as np
import argparse

import os

try:
    import torch_ccl
except ImportError as e:
    print(e)

def is_array_increasing_by_1(arr):
    # check if the diff of each array elements is 1
    diff = np.diff(arr)  
    return np.all(diff == 1)

def test_nodes_list(global_input_dir, local_input_dir, graph_name, world_size):
    print("check the correctness of the nodes list...")
    # use pandas to load initial nodes list (txt file)
    initial_nodes_list = pd.read_csv(os.path.join(global_input_dir, "{}_nodes.txt".format(graph_name)), sep=" ", header=None, usecols=[2]).values
    cur_nodes_list_list = []
    # to check the data in each rank
    for rank in range(world_size):
        print("rank: {}".format(rank))
        # use numpy to load local nodes list (npy file)
        # use numpy to load nodes list (npy file)
        cur_nodes_list_list.append(np.load(os.path.join(local_input_dir, "p{:0>3d}-{}_nodes.npy".format(rank, graph_name))))
    # concatenate nodes list in each rank
    cur_nodes_list = np.concatenate(cur_nodes_list_list, axis=0)
    # release memory of nodes list in each rank
    del cur_nodes_list_list
    # assert the number of nodes
    assert cur_nodes_list.shape[0] == initial_nodes_list.shape[0]
    # to check if the data is increasing by 1
    assert is_array_increasing_by_1(cur_nodes_list[:, 0])
    del initial_nodes_list

    return cur_nodes_list

def test_edges_list(global_input_dir, local_input_dir, graph_name, world_size, global_nodes_list):
    # use pandas to load initial edges list (txt file)
    initial_edges_list = pd.read_csv(os.path.join(global_input_dir, "{}_edges.txt".format(graph_name)), sep=" ", header=None).values
    # convert src id and dst id to unique id based on some hash function 
    print("check the correctness of the edges list...")
    global_num_nodes = global_nodes_list.shape[0]
    initial_edges_ids = initial_edges_list[:, 0] * global_num_nodes + initial_edges_list[:, 1]
    current_edges_list_list = []

    begin_node_idx_in_each_part = np.loadtxt(os.path.join(local_input_dir, "begin_node_on_each_partition.txt"), dtype='int64', delimiter=' ')
    
    # to check the data in each rank
    for rank in range(world_size):
        print("rank: {}".format(rank))
        # use numpy to load local edges list (npy file)
        local_edges_list = np.load(os.path.join(local_input_dir, "p{:0>3d}-{}_local_edges.npy".format(rank, graph_name)))
        remote_edges_list = np.load(os.path.join(local_input_dir, "p{:0>3d}-{}_remote_edges.npy".format(rank, graph_name)))
        
        local_nodes_list = np.load(os.path.join(local_input_dir, "p{:0>3d}-{}_nodes.npy".format(rank, graph_name)))

        cur_num_nodes = begin_node_idx_in_each_part[rank+1] - begin_node_idx_in_each_part[rank]

        print("local_edges_list[0, :]: {}".format(local_edges_list[0, :]))
        print("local_edges_list[1, :]: {}".format(local_edges_list[1, :]))
        print("remote_edges_list[0, :]: {}".format(remote_edges_list[0, :]))
        print("remote_edges_list[1, :]: {}".format(remote_edges_list[1, :]))

        # assert the src id and dst id of edges are in the range of nodes id
        assert np.all(local_edges_list[0, :] < cur_num_nodes) and np.all(local_edges_list[1, :] < cur_num_nodes) and \
            np.all(remote_edges_list[1, :] < cur_num_nodes)

        # # recover the local id to global id
        # local_edges_list[:, 0] += begin_node_idx
        # local_edges_list[:, 1] += begin_node_idx
        # remote_edges_list[:, 0] += begin_node_idx

        # remap the src id and dst id to the global id before remapping according to the local nodes list
        local_edges_list[0, :] = local_nodes_list[local_edges_list[0, :], 1]
        local_edges_list[1, :] = local_nodes_list[local_edges_list[1, :], 1]
        # remote src id need to be mapped based on the global nodes list
        remote_edges_list[0, :] = global_nodes_list[remote_edges_list[0, :], 1]
        remote_edges_list[1, :] = local_nodes_list[remote_edges_list[1, :], 1]

        # convert src id and dst id of edges to unique id based on some hash function
        local_edges_ids = local_edges_list[0, :] * global_num_nodes + local_edges_list[1, :]
        remote_edges_ids = remote_edges_list[0, :] * global_num_nodes + remote_edges_list[1, :]

        # concatenate edges list in each rank
        current_edges_list_list.append(np.concatenate([local_edges_ids, remote_edges_ids], axis=0))

        # release memory of edges list in each rank
        del local_edges_list
        del remote_edges_list
        del local_edges_ids
        del remote_edges_ids

    # concatenate edges list in each rank
    current_edges_ids = np.concatenate(current_edges_list_list, axis=0)
    # compare the edges list in each rank with the initial edges list
    assert np.array_equal(np.sort(current_edges_ids), np.sort(initial_edges_ids))

    del current_edges_ids
    del current_edges_list_list
    del initial_edges_ids

def test_nodes_feat_list(global_input_dir, local_input_dir, graph_name, world_size, global_nodes_list):
    print("check the correctness of the nodes feat list...")
    # allocate a numpy array to set the visited flag
    is_visited = np.zeros(global_nodes_list.shape[0], dtype=np.int32)
    # use numpy to load initial nodes feat list (npy file)
    initial_nodes_feat_list = np.load(os.path.join(global_input_dir, "{}_nodes_feat.npy".format(graph_name)))
    # to check the data in each rank
    for rank in range(world_size):
        print("rank: {}".format(rank))
        # use numpy to load local nodes feat list (npy file)
        cur_nodes_feat_list = np.load(os.path.join(local_input_dir, "p{:0>3d}-{}_nodes_feat.npy".format(rank, graph_name)))
        # use numpy to load local nodes list (npy file)
        cur_nodes_list = np.load(os.path.join(local_input_dir, "p{:0>3d}-{}_nodes.npy".format(rank, graph_name)))
        # localize the id in cur nodes list
        cur_nodes_list[:, 0] -= cur_nodes_list[0, 0]
        # compare the nodes feat list in each rank with the initial nodes feat list
        assert np.allclose(cur_nodes_feat_list[cur_nodes_list[:, 0]], initial_nodes_feat_list[cur_nodes_list[:, 1]])
        # set the visited flag
        is_visited[cur_nodes_list[:, 1]] += 1

        del cur_nodes_feat_list
        del cur_nodes_list

    # check if all elements in is_visited are 1
    assert np.all(is_visited == 1)

def test_nodes_mask(global_input_dir, local_input_dir, graph_name, world_size, label):
    print("check the correctness of the {} mask...".format(label))
    # use pandas to load initial training mask (txt file)
    initial_training_mask = pd.read_csv(os.path.join(global_input_dir, "{}_nodes_{}_idx.txt".format(graph_name, label)), sep=" ", header=None).values
    initial_training_mask = initial_training_mask.reshape(-1)

    cur_training_mask_list = []
    for rank in range(world_size):
        print("rank: {}".format(rank))
        # use numpy to load local training mask (npy file)
        cur_training_mask = np.load(os.path.join(local_input_dir, "p{:0>3d}-{}_nodes_{}_idx.npy".format(rank, graph_name, label)))

        # use numpy to load local nodes list (npy file)
        cur_nodes_list = np.load(os.path.join(local_input_dir, "p{:0>3d}-{}_nodes.npy".format(rank, graph_name)))

        # localize the id in cur nodes list
        cur_nodes_list[:, 0] -= cur_nodes_list[0, 0]

        # remap the id in training mask to the id before remapping according to the local nodes list
        cur_training_mask = cur_nodes_list[cur_training_mask, 1]

        cur_training_mask_list.append(cur_training_mask)

    # concatenate training mask in each rank
    cur_training_mask = np.concatenate(cur_training_mask_list, axis=0)

    # compare the training mask in each rank with the initial training mask
    assert np.array_equal(np.sort(cur_training_mask), np.sort(initial_training_mask))

def test_nodes_label(global_input_dir, local_input_dir, graph_name, world_size):
    print("check the correctness of the nodes label...")
    # use numpy to load initial nodes label list (npy file) 
    initial_nodes_label_list = pd.read_csv(os.path.join(global_input_dir, "{}_nodes_label.txt".format(graph_name)), sep=" ", header=None).values.reshape(-1)
    # initial_nodes_label_list = np.load(os.path.join(global_input_dir, "{}_nodes_label.npy".format(graph_name))).reshape(-1)
    # allocate a numpy array to set the visited flag
    is_visited = np.zeros(global_nodes_list.shape[0], dtype=np.int32)
    # to check the data in each rank
    for rank in range(world_size):
        print("rank: {}".format(rank))
        # use numpy to load local nodes label list (npy file)
        cur_nodes_label_list = np.load(os.path.join(local_input_dir, "p{:0>3d}-{}_nodes_label.npy".format(rank, graph_name)))
        # use numpy to load local nodes list (npy file)
        cur_nodes_list = np.load(os.path.join(local_input_dir, "p{:0>3d}-{}_nodes.npy".format(rank, graph_name)))
        # localize the id in cur nodes list
        cur_nodes_list[:, 0] -= cur_nodes_list[0, 0]
        # compare the nodes label list in each rank with the initial nodes label list
        assert np.array_equal(cur_nodes_label_list[cur_nodes_list[:, 0]], initial_nodes_label_list[cur_nodes_list[:, 1]])
        # set the visited flag
        is_visited[cur_nodes_list[:, 1]] += 1

    # check if all elements in is_visited are 1
    assert np.all(is_visited == 1)


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='Test the correctness of the data loading')
    parser.add_argument('--global_input_dir', type=str, help='the initial input directory')
    parser.add_argument('--local_input_dir', type=str, help='the current input directory')
    parser.add_argument('--graph_name', type=str, help='the graph name')
    parser.add_argument('--world_size', type=int, help='the world size')

    args = parser.parse_args()
    global_input_dir = args.global_input_dir
    local_input_dir = args.local_input_dir
    graph_name = args.graph_name
    world_size = args.world_size

    # test the correctness of the nodes list
    global_nodes_list = test_nodes_list(global_input_dir, local_input_dir, graph_name, world_size)

    # # test the correctness of the edges list
    test_edges_list(global_input_dir, local_input_dir, graph_name, world_size, global_nodes_list)

    # # test the correctness of the nodes feat list
    test_nodes_feat_list(global_input_dir, local_input_dir, graph_name, world_size, global_nodes_list)

    # # test the correctness of the training mask
    test_nodes_mask(global_input_dir, local_input_dir, graph_name, world_size, "train")

    # # test the correctness of the validation mask
    test_nodes_mask(global_input_dir, local_input_dir, graph_name, world_size, "valid")

    # # test the correctness of the test mask
    test_nodes_mask(global_input_dir, local_input_dir, graph_name, world_size, "test")

    # # test the correctness of the nodes label
    test_nodes_label(global_input_dir, local_input_dir, graph_name, world_size)

    print("Test passed!")
