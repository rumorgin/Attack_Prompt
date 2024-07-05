import torch
from torch import nn, optim
from torch_geometric.loader import DataLoader
from GAT import GAT
from GCN import GCN
from GraphSAGE import GraphSAGE
from AllInOnePrompt import HeavyPrompt
from prompt_graph.evaluation import GPPTEva, GNNNodeEva, GPFEva, MultiGpromptEva
from prompt_graph.pretrain import PrePrompt, prompt_pretrain_sample
from .task import BaseTask
import time
import warnings
import numpy as np
from prompt_graph.data import load4node, induced_graphs, graph_split, split_induced_graphs, node_sample_and_save
from prompt_graph.evaluation import GpromptEva, AllInOneEva
import pickle
import os
from prompt_graph.utils import process
warnings.filterwarnings("ignore")

class NodeTask(torch.nn.Module):
      def __init__(self, pre_train_model_path=None, gnn_type='TransformerConv', hid_dim = 128, num_layer = 2, dataset_name='Cora', prompt_type='GPF', epochs=100, shot_num=10, device : int = 5):
            super().__init__()
            self.task_type = 'NodeTask'
            self.pre_train_model_path = pre_train_model_path
            self.device = torch.device('cuda:' + str(device) if torch.cuda.is_available() else 'cpu')
            self.hid_dim = hid_dim
            self.num_layer = num_layer
            self.dataset_name = dataset_name
            self.shot_num = shot_num
            self.gnn_type = gnn_type
            self.prompt_type = prompt_type
            self.epochs = epochs

            self.load_data()
            self.answering =  torch.nn.Sequential(torch.nn.Linear(self.hid_dim, self.output_dim),
                                          torch.nn.Softmax(dim=1)).to(self.device)
            self.initialize_lossfn()
            self.create_few_data_folder()         
            self.initialize_gnn()
            self.initialize_prompt()
            self.initialize_optimizer()

      def initialize_optimizer(self):
            if self.prompt_type == 'None':
                  model_param_group = []
                  model_param_group.append({"params": self.gnn.parameters()})
                  model_param_group.append({"params": self.answering.parameters()})
                  self.optimizer = optim.Adam(model_param_group, lr=0.005, weight_decay=5e-4)
            elif self.prompt_type == 'All-in-one':
                  self.pg_opi = optim.Adam(filter(lambda p: p.requires_grad, self.prompt.parameters()), lr=0.001,
                                           weight_decay=0.00001)
                  self.answer_opi = optim.Adam(filter(lambda p: p.requires_grad, self.answering.parameters()), lr=0.001,
                                               weight_decay=0.00001)
      def initialize_lossfn(self):
            self.criterion = torch.nn.CrossEntropyLoss()

      def initialize_prompt(self):
            if self.prompt_type == 'None':
                  self.prompt = None
            elif self.prompt_type == 'All-in-one':
                  lr, wd = 0.001, 0.00001
                  self.prompt = HeavyPrompt(token_dim=self.input_dim, token_num=10, cross_prune=0.1,
                                            inner_prune=0.3).to(self.device)
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

            if self.pre_train_model_path != 'None' and self.prompt_type != 'MultiGprompt':
                  if self.gnn_type not in self.pre_train_model_path:
                        raise ValueError(f"the Downstream gnn '{self.gnn_type}' does not match the pre-train model")
                  if self.dataset_name not in self.pre_train_model_path:
                        raise ValueError(
                              f"the Downstream dataset '{self.dataset_name}' does not match the pre-train dataset")

                  self.gnn.load_state_dict(torch.load(self.pre_train_model_path, map_location=self.device))
                  print("Successfully loaded pre-trained weights!")

      def create_few_data_folder(self):
            # 创建文件夹并保存数据
            for k in range(1, 11):
                  k_shot_folder = './Experiment/sample_data/Node/'+ self.dataset_name +'/' + str(k) +'_shot'
                  os.makedirs(k_shot_folder, exist_ok=True)
                  
                  for i in range(1, 6):
                        folder = os.path.join(k_shot_folder, str(i))
                        os.makedirs(folder, exist_ok=True)
                        node_sample_and_save(self.data, k, folder, self.output_dim)
                        print(str(k) + ' shot ' + str(i) + ' th is saved!!')

      def load_multigprompt_data(self):
            adj, features, labels, idx_train, idx_val, idx_test = process.load_data(self.dataset_name)  
            self.input_dim = features.shape[1]
            features, _ = process.preprocess_features(features)
            self.sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj).to(self.device)
            self.labels = torch.FloatTensor(labels[np.newaxis])
            self.features = torch.FloatTensor(features[np.newaxis]).to(self.device)
            self.idx_train = torch.LongTensor(idx_train)
            # print("labels",labels)
            print("adj",self.sp_adj.shape)
            print("feature",features.shape)
            self.idx_val = torch.LongTensor(idx_val)
            self.idx_test = torch.LongTensor(idx_test)

      def load_induced_graph(self):
            self.data, self.dataset = load4node(self.dataset_name, shot_num = self.shot_num)
            # self.data.to('cpu')
            self.input_dim = self.dataset.num_features
            self.output_dim = self.dataset.num_classes
            file_path = 'D:/ProG-main/Experiment/induced_graph/' + self.dataset_name + '/induced_graph.pkl'
            if os.path.exists(file_path):
                  with open(file_path, 'rb') as f:
                        graphs_list = pickle.load(f)
            else:
                  print('Begin split_induced_graphs.')
                  split_induced_graphs(self.dataset_name, self.data, smallest_size=10, largest_size=30)
                  with open(file_path, 'rb') as f:
                        graphs_list = pickle.load(f)
            return graphs_list

      
      def load_data(self):
            self.data, self.dataset = load4node(self.dataset_name, shot_num = self.shot_num)
            self.data.to(self.device)
            self.input_dim = self.dataset.num_features
            self.output_dim = self.dataset.num_classes
      
      def train(self, data, train_idx):
            self.gnn.train()
            self.optimizer.zero_grad() 
            out = self.gnn(data.x, data.edge_index, batch=None) 
            out = self.answering(out)
            loss = self.criterion(out[train_idx], data.y[train_idx])
            loss.backward()  
            self.optimizer.step()  
            return loss.item()
      
      def SUPTtrain(self, data):
            self.gnn.train()
            self.optimizer.zero_grad() 
            data.x = self.prompt.add(data.x)
            out = self.gnn(data.x, data.edge_index, batch=None) 
            out = self.answering(out)
            loss = self.criterion(out[data.train_mask], data.y[data.train_mask])  
            orth_loss = self.prompt.orthogonal_loss()
            loss += orth_loss
            loss.backward()  
            self.optimizer.step()  
            return loss

      def AllInOneTrain(self, train_loader):
            #we update answering and prompt alternately.
            
            answer_epoch = 1  # 50
            prompt_epoch = 1  # 50
            
            # tune task head
            self.answering.train()
            self.prompt.eval()
            for epoch in range(1, answer_epoch + 1):
                  answer_loss = self.prompt.Tune(train_loader, self.gnn,  self.answering, self.criterion, self.answer_opi, self.device)
                  print(("frozen gnn | frozen prompt | *tune answering function... {}/{} ,loss: {:.4f} ".format(epoch, answer_epoch, answer_loss)))

            # tune prompt
            self.answering.eval()
            self.prompt.train()
            for epoch in range(1, prompt_epoch + 1):
                  pg_loss = self.prompt.Tune( train_loader,  self.gnn, self.answering, self.criterion, self.pg_opi, self.device)
                  print(("frozen gnn | *tune prompt |frozen answering function... {}/{} ,loss: {:.4f} ".format(epoch, answer_epoch, pg_loss)))
            
            return pg_loss

      
      def run(self):
            test_accs = []
            # if self.prompt_type == 'MultiGprompt':    
            for i in range(1, 6):
                  self.dataset_name ='Cora'
                  idx_train = torch.load(r"D:/ProG-main/Experiment/sample_data/Node/{}/{}_shot/{}/train_idx.pt".format(self.dataset_name, self.shot_num, i)).type(torch.long).to(self.device)
                  print('idx_train',idx_train)
                  train_lbls = torch.load(r"D:/ProG-main/Experiment/sample_data/Node/{}/{}_shot/{}/train_labels.pt".format(self.dataset_name, self.shot_num, i)).type(torch.long).squeeze().to(self.device)
                  print("true",i,train_lbls)

                  idx_test = torch.load(r"D:/ProG-main/Experiment/sample_data/Node/{}/{}_shot/{}/test_idx.pt".format(self.dataset_name, self.shot_num, i)).type(torch.long).to(self.device)
                  test_lbls = torch.load(r"D:/ProG-main/Experiment/sample_data/Node/{}/{}_shot/{}/test_labels.pt".format(self.dataset_name, self.shot_num, i)).type(torch.long).squeeze().to(self.device)
                  
                  # for all-in-one and Gprompt we use k-hop subgraph
                  graphs_list = self.load_induced_graph()
                  train_graphs = []
                  test_graphs = []

                  for graph in graphs_list:
                        if graph.index in idx_train:
                              train_graphs.append(graph)
                        elif graph.index in idx_test:
                              test_graphs.append(graph)

                  train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
                  test_loader = DataLoader(test_graphs, batch_size=16, shuffle=False)
                  print("prepare induce graph data is finished!")


                  patience = 20
                  best = 1e9
                  cnt_wait = 0
                 


                  for epoch in range(1, self.epochs):
                        t0 = time.time()
                        if self.prompt_type == 'None':
                              loss = self.train(self.data, idx_train)                             
                        elif self.prompt_type == 'GPPT':
                              loss = self.GPPTtrain(self.data, idx_train)                
                        elif self.prompt_type == 'All-in-one':
                              loss = self.AllInOneTrain(train_loader)                           
                        elif self.prompt_type in ['GPF', 'GPF-plus']:
                              loss = self.GPFTrain(train_loader)                                                          
                        elif self.prompt_type =='Gprompt':
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
                                    print('Early stopping at '+str(epoch) +' eopch!')
                                    break
                        print("Epoch {:03d} |  Time(s) {:.4f} | Loss {:.4f}  ".format(epoch, time.time() - t0, loss))
           
                  if self.prompt_type == 'None':
                        self.gnn.eval()
                        out = self.gnn(self.data.x, self.data.edge_index, batch=None)
                        out = self.answering(out)
                        pred = out.argmax(dim=1)
                        correct = pred[idx_test] == self.data.y[idx_test]
                        test_acc = int(correct.sum()) / len(idx_test)
                  elif self.prompt_type == 'All-in-one':
                        test_acc, F1 = AllInOneEva(test_loader, self.prompt, self.gnn, self.answering, self.output_dim, self.device)

                  print("test accuracy {:.4f} ".format(test_acc))                        
                  test_accs.append(test_acc)
         

            mean_test_acc = np.mean(test_accs)
            std_test_acc = np.std(test_accs)    
            print(" Final best | test Accuracy {:.4f} | std {:.4f} ".format(mean_test_acc, std_test_acc))         
                  

