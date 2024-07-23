import os
import torch.nn.functional as F
from ogb.nodeproppred import NodePropPredDataset
from torch_geometric.datasets import Planetoid, Amazon, Reddit, WikiCS, Flickr
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import subgraph, k_hop_subgraph
from torch_geometric.data import Data
import pickle
import numpy as np
import scipy.sparse as sp
import torch
import scipy.io as sio
from scipy.sparse import csr_matrix
import random
from sklearn import preprocessing
from sklearn.metrics import f1_score
import json

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

def load_npz_to_sparse_graph(file_name):
    """Load a SparseGraph from a Numpy binary file.
    Parameters
    ----------
    file_name : str
        Name of the file to load.
    Returns
    -------
    sparse_graph : SparseGraph
        Graph in sparse matrix format.
    """
    with np.load(file_name) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
                                   shape=loader['adj_shape'])

        if 'attr_data' in loader:
            # Attributes are stored as a sparse CSR matrix
            attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'], loader['attr_indptr']),
                                        shape=loader['attr_shape'])
        elif 'attr_matrix' in loader:
            # Attributes are stored as a (dense) np.ndarray
            attr_matrix = loader['attr_matrix']
        else:
            attr_matrix = None

        if 'labels_data' in loader:
            # Labels are stored as a CSR matrix
            labels = sp.csr_matrix((loader['labels_data'], loader['labels_indices'], loader['labels_indptr']),
                                   shape=loader['labels_shape'])
        elif 'labels' in loader:
            # Labels are stored as a numpy array
            labels = loader['labels']
        else:
            labels = None

        node_names = loader.get('node_names')
        attr_names = loader.get('attr_names')
        class_names = loader.get('class_names')
        metadata = loader.get('metadata')

    return adj_matrix, attr_matrix, labels, node_names, attr_names, class_names, metadata


valid_num_dic = {'Amazon_clothing': 17, 'Amazon_eletronics': 36, 'dblp': 27}


def load_data(dataset_source):
    class_list_train, class_list_valid, class_list_test = json.load(
        open('./data/{}_class_split.json'.format(dataset_source)))
    if dataset_source in valid_num_dic.keys():
        n1s = []
        n2s = []
        for line in open("data/{}_network".format(dataset_source)):
            n1, n2 = line.strip().split('\t')
            n1s.append(int(n1))
            n2s.append(int(n2))

        num_nodes = max(max(n1s), max(n2s)) + 1
        adj = sp.coo_matrix((np.ones(len(n1s)), (n1s, n2s)),
                            shape=(num_nodes, num_nodes))

        data_train = sio.loadmat("data/{}_train.mat".format(dataset_source))
        train_class = list(set(data_train["Label"].reshape((1, len(data_train["Label"])))[0]))

        data_test = sio.loadmat("data/{}_test.mat".format(dataset_source))
        class_list_test = list(set(data_test["Label"].reshape((1, len(data_test["Label"])))[0]))

        labels = np.zeros((num_nodes, 1))
        labels[data_train['Index']] = data_train["Label"]
        labels[data_test['Index']] = data_test["Label"]

        features = np.zeros((num_nodes, data_train["Attributes"].shape[1]))
        features[data_train['Index']] = data_train["Attributes"].toarray()
        features[data_test['Index']] = data_test["Attributes"].toarray()

        class_list = []
        for cla in labels:
            if cla[0] not in class_list:
                class_list.append(cla[0])  # unsorted

        id_by_class = {}
        for i in class_list:
            id_by_class[i] = []
        for id, cla in enumerate(labels):
            id_by_class[cla[0]].append(id)

        lb = preprocessing.LabelBinarizer()
        labels = lb.fit_transform(labels)

        degree = np.sum(adj, axis=1)
        degree = torch.FloatTensor(degree)

        adj = normalize_adj(adj + sp.eye(adj.shape[0]))
        adj = sparse_mx_to_torch_csr_tensor(adj)
        features = torch.FloatTensor(features)
        labels = torch.LongTensor(np.where(labels)[1])

    elif dataset_source=='cora-full':
        adj, features, labels, node_names, attr_names, class_names, metadata=load_npz_to_sparse_graph('./data/cora_full.npz')

        sparse_mx = adj.tocoo().astype(np.float32)
        indices =np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)

        n1s=indices[0].tolist()
        n2s=indices[1].tolist()

        degree = np.sum(adj, axis=1)
        degree = torch.FloatTensor(degree)

        adj = normalize(adj.tocoo() + sp.eye(adj.shape[0]))
        adj= sparse_mx_to_torch_csr_tensor(adj)
        features=features.todense()
        features = torch.FloatTensor(features)
        labels=torch.LongTensor(labels).squeeze()


        class_list =  class_list_train+class_list_valid+class_list_test

        id_by_class = {}
        for i in class_list:
            id_by_class[i] = []
        for id, cla in enumerate(labels.numpy().tolist()):
            id_by_class[cla].append(id)

    elif dataset_source=='ogbn-arxiv':

        dataset = NodePropPredDataset(name = dataset_source)

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, labels = dataset[0] # graph: library-agnostic graph object

        n1s=graph['edge_index'][0]
        n2s=graph['edge_index'][1]

        num_nodes = graph['num_nodes']
        print('nodes num',num_nodes)
        adj = sp.coo_matrix((np.ones(len(n1s)), (n1s, n2s)),
                                shape=(num_nodes, num_nodes))
        degree = np.sum(adj, axis=1)
        degree = torch.FloatTensor(degree)
        adj = normalize(adj + sp.eye(adj.shape[0]))
        adj = sparse_mx_to_torch_csr_tensor(adj)

        features=torch.FloatTensor(graph['node_feat'])
        labels=torch.LongTensor(labels).squeeze()


        class_list =  class_list_train+class_list_valid+class_list_test

        id_by_class = {}
        for i in class_list:
            id_by_class[i] = []
        for id, cla in enumerate(labels.numpy().tolist()):
            id_by_class[cla].append(id)


    return adj, features, labels, degree, class_list_train, class_list_valid, class_list_test, id_by_class


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def f1(output, labels):
    preds = output.max(1)[1].type_as(labels)
    f1 = f1_score(labels, preds, average='weighted')
    return f1


def sparse_mx_to_torch_coo_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch COO tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def sparse_mx_to_torch_csr_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse CSR tensor."""
    # Ensure the input is in CSR format
    sparse_mx = csr_matrix(sparse_mx)

    # Get the CSR format data
    crow_indices = torch.from_numpy(sparse_mx.indptr).to(dtype=torch.int32)
    col_indices = torch.from_numpy(sparse_mx.indices).to(dtype=torch.int32)
    values = torch.from_numpy(sparse_mx.data).to(dtype=torch.float32)

    # Get the shape of the matrix
    shape = torch.Size(sparse_mx.shape)

    # Create a torch sparse CSR tensor
    return torch.sparse_csr_tensor(crow_indices, col_indices, values, shape)


def task_generator(id_by_class, class_list, n_way, k_shot, m_query):
    # sample class indices
    class_selected = random.sample(class_list, n_way)
    id_support = []
    id_query = []
    for cla in class_selected:
        temp = random.sample(id_by_class[cla], k_shot + m_query)
        id_support.extend(temp[:k_shot])
        id_query.extend(temp[k_shot:])

    return np.array(id_support), np.array(id_query), class_selected


def euclidean_dist(x, y):
    # x: N x D query
    # y: M x D prototype
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)  # N x M

