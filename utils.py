import os
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, Amazon, Reddit, WikiCS, Flickr
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import subgraph, k_hop_subgraph
from torch_geometric.data import Data
import pickle

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("create folder {}".format(path))
    else:
        print("folder exists! {}".format(path))

def act(x=None, act_type='leakyrelu'):
    if act_type == 'leakyrelu':
        return torch.nn.LeakyReLU() if x is None else F.leaky_relu(x)
    elif act_type == 'tanh':
        return torch.nn.Tanh() if x is None else torch.tanh(x)
    elif act_type == 'relu':
        return torch.nn.ReLU() if x is None else F.relu(x)
    elif act_type == 'sigmoid':
        return torch.nn.Sigmoid() if x is None else torch.sigmoid(x)
    elif act_type == 'softmax':
        # 注意：softmax 需要指定维度；这里假设对最后一个维度进行softmax
        return torch.nn.Softmax(dim=-1) if x is None else F.softmax(x, dim=-1)
    else:
        raise ValueError(f"Unsupported activation type: {act_type}")


def load4node(dataname):
    print(dataname)
    if dataname in ['PubMed', 'CiteSeer', 'Cora']:
        dataset = Planetoid(root='data/Planetoid', name=dataname, transform=NormalizeFeatures())
        data = dataset[0]
        input_dim = dataset.num_features
        out_dim = dataset.num_classes
    elif dataname in ['Computers', 'Photo']:
        dataset = Amazon(root='data/amazon', name=dataname)
        data = dataset[0]
        input_dim = dataset.num_features
        out_dim = dataset.num_classes
    elif dataname == 'Reddit':
        dataset = Reddit(root='data/Reddit')
        data = dataset[0]
        input_dim = dataset.num_features
        out_dim = dataset.num_classes
    elif dataname == 'WikiCS':
        dataset = WikiCS(root='data/WikiCS')
        data = dataset[0]
        input_dim = dataset.num_features
        out_dim = dataset.num_classes
    elif dataname == 'Flickr':
        dataset = Flickr(root='data/Flickr')
        data = dataset[0]
        input_dim = dataset.num_features
        out_dim = dataset.num_classes

    return data, input_dim, out_dim


def split_induced_graphs(data, dir_path, device, smallest_size=10, largest_size=30):
    induced_graph_list = []
    saved_graph_list = []
    from copy import deepcopy

    for index in range(data.x.size(0)):
        current_label = data.y[index].item()

        current_hop = 2
        subset, _, _, _ = k_hop_subgraph(node_idx=index, num_hops=current_hop,
                                         edge_index=data.edge_index, relabel_nodes=True)
        subset = subset

        while len(subset) < smallest_size and current_hop < 5:
            current_hop += 1
            subset, _, _, _ = k_hop_subgraph(node_idx=index, num_hops=current_hop,
                                             edge_index=data.edge_index)

        if len(subset) < smallest_size:
            need_node_num = smallest_size - len(subset)
            pos_nodes = torch.argwhere(data.y == int(current_label))
            pos_nodes = pos_nodes.to('cpu')
            subset = subset.to('cpu')
            candidate_nodes = torch.from_numpy(np.setdiff1d(pos_nodes.numpy(), subset.numpy()))
            candidate_nodes = candidate_nodes[torch.randperm(candidate_nodes.shape[0])][0:need_node_num]
            subset = torch.cat([torch.flatten(subset), torch.flatten(candidate_nodes)])

        if len(subset) > largest_size:
            subset = subset[torch.randperm(subset.shape[0])][0:largest_size - 1]
            subset = torch.unique(torch.cat([torch.LongTensor([index]).to(device), torch.flatten(subset)]))

        subset = subset.to(device)
        sub_edge_index, _ = subgraph(subset, data.edge_index, relabel_nodes=True)
        sub_edge_index = sub_edge_index.to(device)

        x = data.x[subset]

        induced_graph = Data(x=x, edge_index=sub_edge_index, y=current_label, index=index)
        saved_graph_list.append(deepcopy(induced_graph).to('cpu'))
        induced_graph_list.append(induced_graph)
        if index % 500 == 0:
            print(index)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    file_path = os.path.join(dir_path, 'induced_graph_min' + str(smallest_size) + '_max' + str(largest_size) + '.pkl')
    with open(file_path, 'wb') as f:
        # Assuming 'data' is what you want to pickle
        # pickle.dump(induced_graph_list, f)
        pickle.dump(saved_graph_list, f)
        print("induced graph data has been write into " + file_path)