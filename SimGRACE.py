import torch
import torch.optim as optim
from torch.autograd import Variable
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset,Planetoid, Amazon, Reddit, WikiCS, Flickr
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_undirected
from torch_geometric.loader.cluster import ClusterData
from torch_geometric.data import Data,Batch
from torch.optim import Adam
from copy import deepcopy
from GAT import GAT
from GCN import GCN
from GraphSAGE import GraphSAGE
from torch.optim import Adam
import os


def load4graph(dataset_name, shot_num=10, num_parts=None, pretrained=False):
    r"""A plain old python object modeling a batch of graphs as one big
        (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
        base class, all its methods can also be used here.
        In addition, single graphs can be reconstructed via the assignment vector
        :obj:`batch`, which maps each node to its respective graph identifier.
        """

    if dataset_name in ['MUTAG', 'ENZYMES', 'COLLAB', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY', 'COX2', 'BZR',
                        'PTC_MR']:
        dataset = TUDataset(root='data/TUDataset', name=dataset_name)

        torch.manual_seed(12345)
        dataset = dataset.shuffle()
        graph_list = [data for data in dataset]

        input_dim = dataset.num_features
        out_dim = dataset.num_classes

        if (pretrained == True):
            return input_dim, out_dim, graph_list
        else:
            return input_dim, out_dim, dataset

    if dataset_name in ['PubMed', 'CiteSeer', 'Cora']:
        dataset = Planetoid(root='data/Planetoid', name=dataset_name, transform=NormalizeFeatures())
        data = dataset[0]
        num_parts = 200

        x = data.x.detach()
        edge_index = data.edge_index
        edge_index = to_undirected(edge_index)
        data = Data(x=x, edge_index=edge_index)
        input_dim = dataset.num_features
        out_dim = dataset.num_classes

        dataset = list(ClusterData(data=data, num_parts=num_parts))
        graph_list = dataset
        # 这里的图没有标签

        return input_dim, out_dim, None, None, None, graph_list

def NodePretrain(dataname='CiteSeer', num_parts=200):
    if dataname in ['PubMed', 'CiteSeer', 'Cora']:
        dataset = Planetoid(root='data/Planetoid', name=dataname, transform=NormalizeFeatures())
    elif dataname in ['Computers', 'Photo']:
        dataset = Amazon(root='data/amazon', name=dataname)
    elif dataname == 'Reddit':
        dataset = Reddit(root='data/Reddit')
    elif dataname == 'WikiCS':
        dataset = WikiCS(root='data/WikiCS')
    elif dataname == 'Flickr':
        dataset = Flickr(root='data/Flickr')
    data = dataset[0]

    x = data.x.detach()
    edge_index = data.edge_index
    edge_index = to_undirected(edge_index)
    data = Data(x=x, edge_index=edge_index)
    input_dim = data.x.shape[1]
    graph_list = list(ClusterData(data=data, num_parts=num_parts))

    return graph_list, input_dim

class SimGRACE(torch.nn.Module):

    def __init__(self, gnn_type='TransformerConv', dataset_name = 'Cora', hid_dim = 128, gln = 2, num_epoch=100, device : int = 5):  # hid_dim=16
        super().__init__()
        self.device = torch.device('cuda:' + str(device) if torch.cuda.is_available() else 'cpu')
        self.dataset_name = dataset_name
        self.gnn_type = gnn_type
        self.num_layer = gln
        self.epochs = num_epoch
        self.hid_dim =hid_dim
        self.load_graph_data()
        self.initialize_gnn(self.input_dim, self.hid_dim)
        self.projection_head = torch.nn.Sequential(torch.nn.Linear(self.hid_dim, self.hid_dim),
                                                   torch.nn.ReLU(inplace=True),
                                                   torch.nn.Linear(self.hid_dim, self.hid_dim)).to(self.device)

    def initialize_gnn(self, input_dim, hid_dim):
        if self.gnn_type == 'GAT':
                self.gnn = GAT(input_dim = input_dim, hid_dim = hid_dim, num_layer = self.num_layer)
        elif self.gnn_type == 'GCN':
                self.gnn = GCN(input_dim = input_dim, hid_dim = hid_dim, num_layer = self.num_layer)
        elif self.gnn_type == 'GraphSAGE':
                self.gnn = GraphSAGE(input_dim = input_dim, hid_dim = hid_dim, num_layer = self.num_layer)
        else:
                raise ValueError(f"Unsupported GNN type: {self.gnn_type}")
        print(self.gnn)
        self.gnn.to(self.device)
        self.optimizer = Adam(self.gnn.parameters(), lr=0.001, weight_decay=0.00005)

    def load_graph_data(self):
        if self.dataset_name in ['PubMed', 'CiteSeer', 'Cora', 'Computers', 'Photo', 'Reddit', 'WikiCS', 'Flickr']:


            self.graph_list, self.input_dim = NodePretrain(dataname=self.dataset_name, num_parts=200)
        else:
            self.input_dim, self.out_dim, self.graph_list = load4graph(self.dataset_name, pretrained=True)

    def get_loader(self, graph_list, batch_size):

        if len(graph_list) % batch_size == 1:
            raise KeyError(
                "batch_size {} makes the last batch only contain 1 graph, \n which will trigger a zero bug in SimGRACE!")

        loader = DataLoader(graph_list, batch_size=batch_size, shuffle=False, num_workers=4)
        return loader

    def forward_cl(self, x, edge_index, batch):
        x = self.gnn(x, edge_index, batch)
        x = self.projection_head(x)
        return x

    def loss_cl(self, x1, x2):
        T = 0.1
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = - torch.log(pos_sim / (sim_matrix.sum(dim=1) + 1e-4)).mean()
        # loss = pos_sim / ((sim_matrix.sum(dim=1) - pos_sim) + 1e-4)
        # loss = - torch.log(loss).mean()
        return loss

    def perturbate_gnn(self, data):
        vice_model = deepcopy(self).to(self.device)

        for (vice_name, vice_model_param) in vice_model.named_parameters():
            if vice_name.split('.')[0] != 'projection_head':
                std = vice_model_param.data.std() if vice_model_param.data.numel() > 1 else torch.tensor(1.0)
                noise = 0.1 * torch.normal(0, torch.ones_like(vice_model_param.data) * std)
                vice_model_param.data += noise
        z2 = vice_model.forward_cl(data.x, data.edge_index, data.batch)
        return z2

    def train_simgrace(self, loader, optimizer):
        self.train()
        train_loss_accum = 0
        total_step = 0
        for step, data in enumerate(loader):
            optimizer.zero_grad()
            data = data.to(self.device)
            x2 = self.perturbate_gnn(data)
            x1 = self.forward_cl(data.x, data.edge_index, data.batch)
            x2 = Variable(x2.detach().data.to(self.device), requires_grad=False)
            loss = self.loss_cl(x1, x2)
            loss.backward()
            optimizer.step()
            train_loss_accum += float(loss.detach().cpu().item())
            total_step = total_step + 1

        return train_loss_accum / total_step

    def pretrain(self, batch_size=10, lr=0.01, decay=0.0001, epochs=100):

        loader = self.get_loader(self.graph_list, batch_size)
        print('start training {} | {} | {}...'.format(self.dataset_name, 'SimGRACE', self.gnn_type))
        optimizer = optim.Adam(self.gnn.parameters(), lr=lr, weight_decay=decay)

        train_loss_min = 1000000
        for epoch in range(1, epochs + 1):  # 1..100
            train_loss = self.train_simgrace(loader, optimizer)

            print("***epoch: {}/{} | train_loss: {:.8}".format(epoch, epochs, train_loss))

            if train_loss_min > train_loss:
                train_loss_min = train_loss
                folder_path = f"./Experiment/pre_trained_model/{self.dataset_name}"
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                torch.save(self.gnn.state_dict(),
                           "./Experiment/pre_trained_model/{}/{}.{}.{}.pth".format(self.dataset_name, 'SimGRACE',
                                                                                   self.gnn_type,
                                                                                   str(self.hid_dim) + 'hidden_dim'))
                print("+++model saved ! {}.{}.{}.{}.pth".format(self.dataset_name, 'SimGRACE', self.gnn_type,
                                                                str(self.hid_dim) + 'hidden_dim'))
