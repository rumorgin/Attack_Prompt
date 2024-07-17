import torch
from torch_geometric.loader import DataLoader
import time
import warnings
import numpy as np
from torch import nn, optim
from GAT import GAT
from GCN import GCN
from GraphSAGE import GraphSAGE
from AllInOnePrompt import HeavyPrompt,GNNNodeEva,AllInOneEva
from Dataset import GraphDataset
from utils import *
import os

warnings.filterwarnings("ignore")


class NodeTask():
    def __init__(self, data, input_dim, output_dim, graphs_list=None, pre_train_model_path='None', gnn_type='TransformerConv',
                 hid_dim = 128, num_layer = 2, dataset_name='Cora', prompt_type='None', epochs=100, shot_num=10, device : int = 5, lr =0.001, wd = 5e-4,
                 batch_size = 16, search = False):
        super().__init__()
        self.task_type = 'NodeTask'
        self.pre_train_model_path = pre_train_model_path
        self.pre_train_type = self.return_pre_train_type(pre_train_model_path)
        self.device = torch.device('cuda:'+ str(device) if torch.cuda.is_available() else 'cpu')
        self.hid_dim = hid_dim
        self.num_layer = num_layer
        self.dataset_name = dataset_name
        self.shot_num = shot_num
        self.gnn_type = gnn_type
        self.prompt_type = prompt_type
        self.epochs = epochs
        self.lr = lr
        self.wd = wd
        self.batch_size = batch_size
        self.search = search
        self.initialize_lossfn()
        self.data = data
        if self.dataset_name == 'ogbn-arxiv':
            self.data.y = self.data.y.squeeze()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.graphs_list = graphs_list
        self.answering = torch.nn.Sequential(torch.nn.Linear(self.hid_dim, self.output_dim),
                                           torch.nn.Softmax(dim=1)).to(self.device)

        self.create_few_data_folder()

    def initialize_lossfn(self):
        self.criterion = torch.nn.CrossEntropyLoss()

    def initialize_optimizer(self):
        if self.prompt_type == 'None':
            if self.pre_train_model_path == 'None':
                model_param_group = []
                model_param_group.append({"params": self.gnn.parameters()})
                model_param_group.append({"params": self.answering.parameters()})
                self.optimizer = optim.Adam(model_param_group, lr=self.lr, weight_decay=self.wd)
            else:
                model_param_group = []
                model_param_group.append({"params": self.gnn.parameters()})
                model_param_group.append({"params": self.answering.parameters()})
                self.optimizer = optim.Adam(model_param_group, lr=self.lr, weight_decay=self.wd)
                # self.optimizer = optim.Adam(self.answering.parameters(), lr=self.lr, weight_decay=self.wd)

        elif self.prompt_type == 'All-in-one':
            self.pg_opi = optim.Adam( self.prompt.parameters(), lr=1e-6, weight_decay= self.wd)
            self.answer_opi = optim.Adam( self.answering.parameters(), lr=self.lr, weight_decay= self.wd)

    def initialize_prompt(self):
        if self.prompt_type == 'None':
            self.prompt = None
        elif self.prompt_type =='All-in-one':
            if(self.task_type=='NodeTask'):
                self.prompt = HeavyPrompt(token_dim=self.input_dim, token_num=10, cross_prune=0.1, inner_prune=0.3).to(self.device)
        else:
            raise KeyError(" We don't support this kind of prompt.")

    def initialize_gnn(self):
        if self.gnn_type == 'GAT':
            self.gnn = GAT(input_dim=self.input_dim, hid_dim=self.hid_dim, num_layer=self.num_layer)
        elif self.gnn_type == 'GCN':
            self.gnn = GCN(input_dim=self.input_dim, hid_dim=self.hid_dim, num_layer=self.num_layer)
        elif self.gnn_type == 'GraphSAGE':
            self.gnn = GraphSAGE(input_dim=self.input_dim, hid_dim=self.hid_dim, num_layer=self.num_layer)
        else:
            raise ValueError(f"Unsupported GNN type: {self.gnn_type}")
        self.gnn.to(self.device)
        print(self.gnn)

        if self.pre_train_model_path != 'None' and self.prompt_type != 'MultiGprompt':
            if self.gnn_type not in self.pre_train_model_path :
                raise ValueError(f"the Downstream gnn '{self.gnn_type}' does not match the pre-train model")
            if self.dataset_name not in self.pre_train_model_path :
                raise ValueError(f"the Downstream dataset '{self.dataset_name}' does not match the pre-train dataset")

            self.gnn.load_state_dict(torch.load(self.pre_train_model_path, map_location='cpu'))
            self.gnn.to(self.device)
            print("Successfully loaded pre-trained weights!")

    def return_pre_train_type(self, pre_train_model_path):
        names = ['None', 'DGI', 'GraphMAE','Edgepred_GPPT', 'Edgepred_Gprompt','GraphCL', 'SimGRACE']
        for name in names:
            if name  in  pre_train_model_path:
                return name

    def create_few_data_folder(self):
        # 创建文件夹并保存数据
        for k in range(1, 11):
            k_shot_folder = './Experiment/sample_data/Node/' + self.dataset_name + '/' + str(k) + '_shot'
            os.makedirs(k_shot_folder, exist_ok=True)

            for i in range(1, 6):
                folder = os.path.join(k_shot_folder, str(i))
                if not os.path.exists(folder):
                    os.makedirs(folder)
                    labels = self.data.y.to('cpu')

                    # 随机选择90%的数据作为测试集
                    num_test = int(0.9 * self.data.num_nodes)
                    if num_test < 1000:
                        num_test = int(0.7 * self.data.num_nodes)
                    test_idx = torch.randperm(self.data.num_nodes)[:num_test]
                    test_labels = labels[test_idx]

                    # 剩下的作为候选训练集
                    remaining_idx = torch.randperm(self.data.num_nodes)[num_test:]
                    remaining_labels = labels[remaining_idx]

                    # 从剩下的数据中选出k*标签数个样本作为训练集
                    train_idx = torch.cat([remaining_idx[remaining_labels == i][:k] for i in range(self.output_dim)])
                    shuffled_indices = torch.randperm(train_idx.size(0))
                    train_idx = train_idx[shuffled_indices]
                    train_labels = labels[train_idx]

                    # 保存文件
                    torch.save(train_idx, os.path.join(folder, 'train_idx.pt'))
                    torch.save(train_labels, os.path.join(folder, 'train_labels.pt'))
                    torch.save(test_idx, os.path.join(folder, 'test_idx.pt'))
                    torch.save(test_labels, os.path.join(folder, 'test_labels.pt'))
                    print(str(k) + ' shot ' + str(i) + ' th is saved!!')

    def load_data(self):
        self.data, self.input_dim, self.output_dim = load4node(self.dataset_name)

    def train(self, data, train_idx):
        self.gnn.train()
        self.answering.train()
        self.optimizer.zero_grad()
        out = self.gnn(data.x, data.edge_index, batch=None)
        out = self.answering(out)
        loss = self.criterion(out[train_idx], data.y[train_idx])
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def AllInOneTrain(self, train_loader, answer_epoch=1, prompt_epoch=1):
        # we update answering and prompt alternately.
        # tune task head
        self.answering.train()
        self.prompt.eval()
        self.gnn.eval()
        for epoch in range(1, answer_epoch + 1):
            answer_loss = self.prompt.Tune(train_loader, self.gnn, self.answering, self.criterion, self.answer_opi,
                                           self.device)
            print(("frozen gnn | frozen prompt | *tune answering function... {}/{} ,loss: {:.4f} ".format(epoch,
                                                                                                          answer_epoch,
                                                                                                          answer_loss)))

        # tune prompt
        self.answering.eval()
        self.prompt.train()
        for epoch in range(1, prompt_epoch + 1):
            pg_loss = self.prompt.Tune(train_loader, self.gnn, self.answering, self.criterion, self.pg_opi, self.device)
            print(("frozen gnn | *tune prompt |frozen answering function... {}/{} ,loss: {:.4f} ".format(epoch,
                                                                                                         prompt_epoch,
                                                                                                         pg_loss)))

        # return pg_loss
        return answer_loss


    def run(self):
        test_accs = []
        f1s = []
        rocs = []
        prcs = []
        batch_best_loss = []
        if self.prompt_type == 'All-in-one':
            self.answer_epoch = 50
            self.prompt_epoch = 50
            self.epochs = int(self.epochs / self.answer_epoch)
        for i in range(1, 6):
            self.initialize_gnn()
            self.initialize_prompt()
            self.initialize_optimizer()
            idx_train = torch.load(
                "./Experiment/sample_data/Node/{}/{}_shot/{}/train_idx.pt".format(self.dataset_name, self.shot_num,
                                                                                  i)).type(torch.long).to(self.device)
            print('idx_train', idx_train)
            train_lbls = torch.load(
                "./Experiment/sample_data/Node/{}/{}_shot/{}/train_labels.pt".format(self.dataset_name, self.shot_num,
                                                                                     i)).type(torch.long).squeeze().to(
                self.device)
            print("true", i, train_lbls)
            idx_test = torch.load(
                "./Experiment/sample_data/Node/{}/{}_shot/{}/test_idx.pt".format(self.dataset_name, self.shot_num,
                                                                                 i)).type(torch.long).to(self.device)
            test_lbls = torch.load(
                "./Experiment/sample_data/Node/{}/{}_shot/{}/test_labels.pt".format(self.dataset_name, self.shot_num,
                                                                                    i)).type(torch.long).squeeze().to(
                self.device)

            # GPPT prompt initialtion
            if self.prompt_type == 'GPPT':
                node_embedding = self.gnn(self.data.x, self.data.edge_index)
                self.prompt.weigth_init(node_embedding, self.data.edge_index, self.data.y, idx_train)

            if self.prompt_type in ['Gprompt', 'All-in-one', 'GPF', 'GPF-plus']:
                train_graphs = []
                test_graphs = []
                # self.graphs_list.to(self.device)
                print('distinguishing the train dataset and test dataset...')
                for graph in self.graphs_list:
                    if graph.index in idx_train:
                        train_graphs.append(graph)
                    elif graph.index in idx_test:
                        test_graphs.append(graph)
                print('Done!!!')

                train_dataset = GraphDataset(train_graphs)
                test_dataset = GraphDataset(test_graphs)

                # 创建数据加载器
                train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
                print("prepare induce graph data is finished!")

            if self.prompt_type == 'MultiGprompt':
                embeds, _ = self.Preprompt.embed(self.features, self.sp_adj, True, None, False)
                pretrain_embs = embeds[0, idx_train]
                test_embs = embeds[0, idx_test]

            patience = 20
            best = 1e9
            cnt_wait = 0
            best_loss = 1e9

            for epoch in range(1, self.epochs):
                t0 = time.time()

                if self.prompt_type == 'None':
                    loss = self.train(self.data, idx_train)
                elif self.prompt_type == 'GPPT':
                    loss = self.GPPTtrain(self.data, idx_train)
                elif self.prompt_type == 'All-in-one':
                    loss = self.AllInOneTrain(train_loader, self.answer_epoch, self.prompt_epoch)
                elif self.prompt_type in ['GPF', 'GPF-plus']:
                    loss = self.GPFTrain(train_loader)
                elif self.prompt_type == 'Gprompt':
                    loss, center = self.GpromptTrain(train_loader)
                elif self.prompt_type == 'MultiGprompt':
                    loss = self.MultiGpromptTrain(pretrain_embs, train_lbls, idx_train)

                if loss < best:
                    best = loss
                    # best_t = epoch
                    cnt_wait = 0
                    # torch.save(model.state_dict(), args.save_name)
                else:
                    cnt_wait += 1
                    if cnt_wait == patience:
                        print('-' * 100)
                        print('Early stopping at ' + str(epoch) + ' eopch!')
                        break

                print("Epoch {:03d} |  Time(s) {:.4f} | Loss {:.4f}  ".format(epoch, time.time() - t0, loss))
            import math
            if not math.isnan(loss):
                batch_best_loss.append(loss)

                if self.prompt_type == 'None':
                    test_acc, f1, roc, prc = GNNNodeEva(self.data, idx_test, self.gnn, self.answering, self.output_dim,
                                                        self.device)
                elif self.prompt_type == 'All-in-one':
                    test_acc, f1, roc, prc = AllInOneEva(test_loader, self.prompt, self.gnn, self.answering,
                                                         self.output_dim, self.device)

                print(
                    f"Final True Accuracy: {test_acc:.4f} | Macro F1 Score: {f1:.4f} | AUROC: {roc:.4f} | AUPRC: {prc:.4f}")
                print("best_loss", batch_best_loss)

                test_accs.append(test_acc)
                f1s.append(f1)
                rocs.append(roc)
                prcs.append(prc)

        mean_test_acc = np.mean(test_accs)
        std_test_acc = np.std(test_accs)
        mean_f1 = np.mean(f1s)
        std_f1 = np.std(f1s)
        mean_roc = np.mean(rocs)
        std_roc = np.std(rocs)
        mean_prc = np.mean(prcs)
        std_prc = np.std(prcs)
        print(" Final best | test Accuracy {:.4f}±{:.4f}(std)".format(mean_test_acc, std_test_acc))
        print(" Final best | test F1 {:.4f}±{:.4f}(std)".format(mean_f1, std_f1))
        print(" Final best | AUROC {:.4f}±{:.4f}(std)".format(mean_roc, std_roc))
        print(" Final best | AUPRC {:.4f}±{:.4f}(std)".format(mean_prc, std_prc))

        print(self.pre_train_type, self.gnn_type, self.prompt_type, " Graph Task completed")
        mean_best = np.mean(batch_best_loss)

        return mean_best, mean_test_acc, std_test_acc, mean_f1, std_f1, mean_roc, std_roc, mean_prc, std_prc


