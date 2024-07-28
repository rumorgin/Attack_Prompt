import argparse
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.datasets import TUDataset,Planetoid, Amazon, Reddit, WikiCS, Flickr, CoraFull, DBLP
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_undirected
from torch_geometric.loader.cluster import ClusterData
from torch_geometric.data import Data,Batch
from torch.optim import Adam
from copy import deepcopy
from GAT import GAT
from GCN import GCN, Encoder
from GraphSAGE import GraphSAGE
from torch.optim import Adam
import os
import json
from utils import *

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--task', type=str)
    parser.add_argument('--dataset_name', type=str, default='cora-full',
                        help='Choose the dataset of pretrainor downstream task, option: dblp, Cora_full')
    parser.add_argument('--device', type=int, default=0,
                        help='Which gpu to use if any (default: 0)')
    parser.add_argument('--gnn_type', type=str, default="GCN",
                        help='We support gnn like \GCN\ \GAT\ \GT\ \GCov\ \GIN\ \GraphSAGE\, please read ProG.model module')
    parser.add_argument('--prompt_type', type=str, default="All-in-one",
                        help='Choose the prompt type for node or graph task, for node task,we support \GPPT\, \All-in-one\, \Gprompt\ for graph task , \All-in-one\, \Gprompt\, \GPF\, \GPF-plus\ ')
    parser.add_argument('--hid_dim', type=int, default=128,
                        help='hideen layer of GNN dimensions (default: 300)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train (default: 50)')
    parser.add_argument('--shot_num', type=int, default=1, help='Number of shots')
    parser.add_argument('--pre_train_model_path', type=str, default='None',
                        help='add pre_train_model_path to the downstream task, the model is self-supervise model if the path is None and prompttype is None.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate (default: 0.0001)')
    parser.add_argument('--decay', type=float, default=0.0001,
                        help='Weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=2,
                        help='Number of GNN message passing layers (default: 3).')

    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='Dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='Graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='How the node features across layers are combined. last, sum, max or concat')

    parser.add_argument('--seed', type=int, default=42, help="Seed for splitting dataset.")
    parser.add_argument('--runseed', type=int, default=0, help="Seed for running experiments.")
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for dataset loading')
    parser.add_argument('--num_layers', type=int, default=1, help='A range of [1,2,3]-layer MLPs with equal width')
    parser.add_argument('--pnum', type=int, default=5, help='The number of independent basis for GPF-plus')

    args = parser.parse_args()
    return args


class SimGRACE(torch.nn.Module):

    def __init__(self, gnn_type='TransformerConv', dataset_name = 'Cora', hid_dim = 64, gln = 2, num_epoch=100, device : int = 0):  # hid_dim=16
        super().__init__()
        self.args = get_args()
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
            # self.gnn = Encoder(in_channels=input_dim,hidden_channels=hid_dim,encoder_type='GCN')
                self.gnn = GCN(input_dim = input_dim, hid_dim = hid_dim, num_layer = self.num_layer)
        elif self.gnn_type == 'GraphSAGE':
                self.gnn = GraphSAGE(input_dim = input_dim, hid_dim = hid_dim, num_layer = self.num_layer)
        else:
                raise ValueError(f"Unsupported GNN type: {self.gnn_type}")
        print(self.gnn)
        self.gnn.to(self.device)
        self.optimizer = Adam(self.gnn.parameters(), lr=0.001, weight_decay=0.00005)

    def load_graph_data(self):

        data, class_list_train, class_list_valid, class_list_test, id_by_class = load_data(self.dataset_name)
        x = data.x.detach()
        edge_index = data.edge_index
        edge_index = to_undirected(edge_index)
        data = Data(x=x, edge_index=edge_index)

        self.graph_list = list(ClusterData(data=data, num_parts=200))
        self.input_dim = data.x.shape[1]


    def get_loader(self, graph_list, batch_size):

        if len(graph_list) % batch_size == 1:
            raise KeyError(
                "batch_size {} makes the last batch only contain 1 graph, \n which will trigger a zero bug in SimGRACE!")

        loader = DataLoader(graph_list, batch_size=batch_size, shuffle=False, num_workers=self.args.num_workers)
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
                folder_path = f"./pre_trained_gnn/{self.dataset_name}"
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                torch.save(self.gnn.state_dict(),
                           "./pre_trained_gnn/{}/{}.{}.{}.pth".format(self.dataset_name, 'SimGRACE',
                                                                                   self.gnn_type,
                                                                                   str(self.hid_dim) + 'hidden_dim'))
                print("+++model saved ! {}.{}.{}.{}.pth".format(self.dataset_name, 'SimGRACE', self.gnn_type,
                                                                str(self.hid_dim) + 'hidden_dim'))


if __name__ == '__main__':

    args = get_args()
    seed_everything(args.seed)
    mkdir('./pre_trained_gnn/')

    pt = SimGRACE(dataset_name = args.dataset_name, gnn_type = args.gnn_type, hid_dim = args.hid_dim, gln = args.num_layer, num_epoch=args.epochs, device=args.device)

    pt.pretrain(batch_size=args.batch_size, lr=args.lr, decay=args.decay, epochs=args.epochs)