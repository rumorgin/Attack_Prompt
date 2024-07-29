import argparse
import time

import torch.optim as optim

from AllInOnePrompt import HeavyPrompt
from GCN import GCN
from GPN import *
from utils import *

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', default=True, action='store_true', help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of episodes to train.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate, default is 0.005.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=128,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')

    parser.add_argument('--way', type=int, default=5, help='way.')
    parser.add_argument('--shot', type=int, default=5, help='shot.')
    parser.add_argument('--qry', type=int, help='k shot for query set', default=10)
    parser.add_argument('--dataset', default='cora-full',
                        help='Dataset:Amazon_clothing/Amazon_eletronics/dblp/cora-full')
    parser.add_argument('--split_induced_graph', type=bool, default=False, help='split induced graph or not')
    parser.add_argument('--pretrain', type=bool, default=True, help='copy the pretrained model parameters or not')

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_cuda:
        torch.cuda.manual_seed(args.seed)
        device = torch.device("cuda:0")

    # Load data
    dataset = args.dataset
    data, class_list_train, class_list_valid, class_list_test, id_by_class = load_data(
        dataset)

    data = data.to(device)

    if args.split_induced_graph:
        folder_path = './Experiment/induced_graph/' + args.dataset
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            split_induced_graphs(data, folder_path, device, smallest_size=10, largest_size=30)


    # encoder = Encoder(in_channels=features.shape[1], hidden_channels=args.hidden)
    encoder = GCN(input_dim=data.x.shape[1], hid_dim=args.hidden, out_dim=None, num_layer=2, JK="last", drop_ratio=0,
                  pool='mean').to(device)
    if args.pretrain:
        if args.dataset == 'cora-full':
            encoder.load_state_dict(
                torch.load('./pre_trained_gnn/cora-full/SimGRACE.GCN.128hidden_dim.pth', map_location='cuda:0'))
            print("Successfully loaded pre-trained weights!")

    answering = torch.nn.Sequential(torch.nn.Linear(args.hidden, args.hidden)).to(device)

    prompt = HeavyPrompt(token_dim=data.num_features, token_num=10, cross_prune=0.1, inner_prune=0.3).to(device)


    optimizer_encoder = optim.Adam(encoder.parameters(),
                                   lr=args.lr, weight_decay=args.weight_decay)
    optimizer_prompt = optim.Adam(prompt.parameters(), lr=1e-6, weight_decay=args.weight_decay)
    # optimizer_answering = optim.Adam(answering.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    n_way = args.way
    k_shot = args.shot
    n_query = args.qry
    meta_test_num = 50
    meta_valid_num = 50

    # Sampling a pool of tasks for validation/testing
    valid_pool = [task_generator(id_by_class, class_list_valid, n_way, k_shot, n_query) for i in range(meta_valid_num)]
    test_pool = [task_generator(id_by_class, class_list_test, n_way, k_shot, n_query) for i in range(meta_test_num)]

    # Train model
    t_total = time.time()
    meta_train_acc = []
    meta_train_loss = []
    # Initialize variables to store the best accuracy and F1 score
    best_test_acc = 0.0
    best_test_f1 = 0.0
    best_epoch = 0

    for episode in range(args.episodes):

        id_support, id_query, class_selected = \
            task_generator(id_by_class, class_list_train, n_way, k_shot, n_query)

        # Load subgraphs for support and query sets
        support_subgraphs = load_induced_graphs(id_support,args.dataset)
        query_subgraphs = load_induced_graphs(id_query,args.dataset)

        # Batch the subgraphs
        support_batch = Batch.from_data_list(support_subgraphs).to(device)
        query_batch = Batch.from_data_list(query_subgraphs).to(device)

        # answering.train()
        encoder.train()

        optimizer_encoder.zero_grad()
        # optimizer_answering.zero_grad()

        support_embeddings = encoder(support_batch.x, support_batch.edge_index, support_batch.batch)
        z_dim = support_embeddings.size()[1]
        # support_embeddings = answering(support_embeddings)

        support_embeddings = support_embeddings.view([n_way, k_shot, z_dim])

        query_embeddings = encoder(query_batch.x, query_batch.edge_index,query_batch.batch)

        # query_embeddings = answering(query_embeddings)

        # compute loss
        prototype_embeddings = support_embeddings.sum(1)

        dists = cosine_dist(query_embeddings, prototype_embeddings)

        # dists = euclidean_dist(query_embeddings, prototype_embeddings)
        output = F.log_softmax(-dists, dim=1)

        labels_new = torch.LongTensor([class_selected.index(i) for i in data.y[id_query]])
        if args.use_cuda:
            labels_new = labels_new.cuda()
        loss_train = F.nll_loss(output, labels_new)

        loss_train.backward()
        optimizer_encoder.step()
        # optimizer_answering.step()

        if args.use_cuda:
            output = output.cpu().detach()
            labels_new = labels_new.cpu().detach()
        acc_train = accuracy(output, labels_new)
        f1_train = f1(output, labels_new)
        meta_train_acc.append(acc_train)
        meta_train_loss.append(loss_train.cpu().detach())
        if episode > 0 and episode % 10 == 0:
            print("-------Episode {}-------".format(episode))
            print("Meta-Train_Accuracy: {}  Meta-Train_Loss: {}".format(np.array(meta_train_acc).mean(axis=0),
                                                                        np.array(meta_train_loss).mean(axis=0)))

            # # validation
            # meta_test_acc = []
            # meta_test_f1 = []
            # for idx in range(meta_valid_num):
            #     id_support, id_query, class_selected = valid_pool[idx]
            #     acc_test, f1_test = test(class_selected, id_support, id_query, n_way, k_shot)
            #     meta_test_acc.append(acc_test)
            #     meta_test_f1.append(f1_test)
            # print("Meta-valid_Accuracy: {}, Meta-valid_F1: {}".format(np.array(meta_test_acc).mean(axis=0),
            #                                                             np.array(meta_test_f1).mean(axis=0)))
            # testing
            meta_test_acc = []
            meta_test_f1 = []
            for idx in range(meta_test_num):
                id_support, id_query, class_selected = test_pool[idx]

                # Load subgraphs for support and query sets
                support_subgraphs = load_induced_graphs(id_support, args.dataset)
                query_subgraphs = load_induced_graphs(id_query, args.dataset)

                # Batch the subgraphs
                support_batch = Batch.from_data_list(support_subgraphs).to(device)
                query_batch = Batch.from_data_list(query_subgraphs).to(device)

                # answering.eval()
                encoder.eval()


                support_embeddings = encoder(support_batch.x, support_batch.edge_index, support_batch.batch)
                z_dim = support_embeddings.size()[1]
                # support_embeddings = answering(support_embeddings)

                support_embeddings = support_embeddings.view([n_way, k_shot, z_dim])

                query_embeddings = encoder(query_batch.x, query_batch.edge_index, query_batch.batch)

                # query_embeddings = answering(query_embeddings)

                # compute loss
                prototype_embeddings = support_embeddings.sum(1)

                dists = cosine_dist(query_embeddings, prototype_embeddings)

                # dists = euclidean_dist(query_embeddings, prototype_embeddings)
                output = F.log_softmax(-dists, dim=1)

                labels_new = torch.LongTensor([class_selected.index(i) for i in data.y[id_query]])
                if args.use_cuda:
                    labels_new = labels_new.cuda()
                loss_test = F.nll_loss(output, labels_new)

                if args.use_cuda:
                    output = output.cpu().detach()
                    labels_new = labels_new.cpu().detach()
                acc_test = accuracy(output, labels_new)
                f1_test = f1(output, labels_new)
                meta_test_acc.append(acc_test)
                meta_test_f1.append(f1_test)

            # Calculate the mean accuracy and F1 score for the current test
            mean_test_acc = np.array(meta_test_acc).mean(axis=0)
            mean_test_f1 = np.array(meta_test_f1).mean(axis=0)
            print("Meta-Test_Accuracy: {}, Meta-Test_F1: {}".format(np.array(meta_test_acc).mean(axis=0),
                                                                    np.array(meta_test_f1).mean(axis=0)))

            # Update the best test results if the current results are better
            if mean_test_acc > best_test_acc:
                best_test_acc = mean_test_acc
                best_epoch = episode
                best_test_f1 = mean_test_f1

    # After the loop, print the best test results
    print("Best Meta-Test_Accuracy: {}".format(best_test_acc))
    print("Best Meta-Test_F1: {}".format(best_test_f1))
    print("Best epoch: {}".format(best_epoch))
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
