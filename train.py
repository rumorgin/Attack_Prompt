from node_task import NodeTask
from utils import *
import pickle
import os
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--task', type=str)
    parser.add_argument('--dataset_name', type=str, default='Cora',
                        help='Choose the dataset of pretrainor downstream task')
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
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs to train (default: 50)')
    parser.add_argument('--shot_num', type=int, default=100, help='Number of shots')
    parser.add_argument('--pre_train_model_path', type=str, default='./Experiment/pre_trained_model/Cora/SimGRACE.GCN.128hidden_dim.pth',
                        help='add pre_train_model_path to the downstream task, the model is self-supervise model if the path is None and prompttype is None.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate (default: 0.0001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='Weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=3,
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

def load_induced_graph(dataset_name, data, device):
    folder_path = './Experiment/induced_graph/' + dataset_name
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_path = folder_path + '/induced_graph_min100_max300.pkl'
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            print('loading induced graph...')
            graphs_list = pickle.load(f)
            print('Done!!!')
    else:
        print('Begin split_induced_graphs.')
        split_induced_graphs(data, folder_path, device, smallest_size=100, largest_size=300)
        with open(file_path, 'rb') as f:
            graphs_list = pickle.load(f)
    graphs_list = [graph.to(device) for graph in graphs_list]
    return graphs_list


if __name__ == '__main__':

    args = get_args()
    seed_everything(args.seed)
    args.task = 'NodeTask'
    args.prompt_type = 'All-in-one' #GPPT All-in-one
    args.shot_num =10
    print('dataset_name', args.dataset_name)

    if args.task == 'NodeTask':
        data, input_dim, output_dim = load4node(args.dataset_name)
        data = data.to(args.device)
        if args.prompt_type in ['Gprompt', 'All-in-one', 'GPF', 'GPF-plus']:
            graphs_list = load_induced_graph(args.dataset_name, data, args.device)
        else:
            graphs_list = None

    if args.task == 'NodeTask':
        tasker = NodeTask(pre_train_model_path=args.pre_train_model_path,
                          dataset_name=args.dataset_name, num_layer=args.num_layer,
                          gnn_type=args.gnn_type, hid_dim=args.hid_dim, prompt_type=args.prompt_type,
                          epochs=args.epochs, shot_num=args.shot_num, device=args.device, lr=args.lr, wd=args.decay,
                          batch_size=args.batch_size, data=data, input_dim=input_dim, output_dim=output_dim,
                          graphs_list=graphs_list)

    pre_train_type = tasker.pre_train_type

    _, test_acc, std_test_acc, f1, std_f1, roc, std_roc, _, _ = tasker.run()

    print("Final Accuracy {:.4f}±{:.4f}(std)".format(test_acc, std_test_acc))
    print("Final F1 {:.4f}±{:.4f}(std)".format(f1, std_f1))
    print("Final AUROC {:.4f}±{:.4f}(std)".format(roc, std_roc))





