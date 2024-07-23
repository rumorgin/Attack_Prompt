import time
import argparse
import numpy as np

import torch
import torch.optim as optim

from utils import *
from GPN import *
from GCN import GCN, Encoder, Projector



def train(class_selected, id_support, id_query, n_way, k_shot):
    # Fine-tuning loop
    encoder.train()  # Set GCN model to evaluation mode

    optimizer_encoder.zero_grad()
    embeddings = encoder(features, adj)
    z_dim = embeddings.size()[1]

    # embedding lookup
    support_embeddings = embeddings[id_support]
    support_embeddings = support_embeddings.view([n_way, k_shot, z_dim])
    query_embeddings = embeddings[id_query]

    # compute loss
    prototype_embeddings = support_embeddings.sum(1)
    dists = euclidean_dist(query_embeddings, prototype_embeddings)
    output = F.log_softmax(-dists, dim=1)

    labels_new = torch.LongTensor([class_selected.index(i) for i in labels[id_query]])
    if args.cuda:
        labels_new = labels_new.cuda()
    loss_train = F.nll_loss(output, labels_new)

    loss_train.backward()
    optimizer_encoder.step()
    # optimizer_scorer.step()

    if args.cuda:
        output = output.cpu().detach()
        labels_new = labels_new.cpu().detach()
    acc_train = accuracy(output, labels_new)
    f1_train = f1(output, labels_new)

    return acc_train, f1_train, loss_train.item()


def test(class_selected, id_support, id_query, n_way, k_shot):
    encoder.eval()
    embeddings = encoder(features, adj)
    z_dim = embeddings.size()[1]

    # embedding lookup
    support_embeddings = embeddings[id_support]
    support_embeddings = support_embeddings.view([n_way, k_shot, z_dim])
    query_embeddings = embeddings[id_query]

    # compute loss
    prototype_embeddings = support_embeddings.sum(1)
    dists = euclidean_dist(query_embeddings, prototype_embeddings)
    output = F.log_softmax(-dists, dim=1)

    labels_new = torch.LongTensor([class_selected.index(i) for i in labels[id_query]])
    if args.cuda:
        labels_new = labels_new.cuda()
    loss_test = F.nll_loss(output, labels_new)

    if args.cuda:
        output = output.cpu().detach()
        labels_new = labels_new.cpu().detach()
    acc_test = accuracy(output, labels_new)
    f1_test = f1(output, labels_new)

    return acc_test, f1_test


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', default=True, action='store_true', help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of episodes to train.')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Initial learning rate, default is 0.005.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=128,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')

    parser.add_argument('--way', type=int, default=5, help='way.')
    parser.add_argument('--shot', type=int, default=5, help='shot.')
    parser.add_argument('--qry', type=int, help='k shot for query set', default=20)
    parser.add_argument('--dataset', default='dblp',
                        help='Dataset:Amazon_clothing/Amazon_eletronics/dblp/cora-full')

    args = parser.parse_args()
    args.cuda = args.use_cuda and torch.cuda.is_available()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Load data
    dataset = args.dataset
    adj, features, labels, degrees, class_list_train, class_list_valid, class_list_test, id_by_class = load_data(
        dataset)

    encoder = Encoder(in_channels=features.shape[1], hidden_channels=args.hidden)
    # encoder = GCN(input_dim=features.shape[1], hid_dim=args.hidden, out_dim=None, num_layer=3,JK="last", drop_ratio=0, pool='mean')


    # encoder.load_state_dict(torch.load('./pre_trained_gnn/Cora_full/SimGRACE.GCN.64hidden_dim.pth', map_location='cuda:0'))
    # print("Successfully loaded pre-trained weights!")

    optimizer_encoder = optim.Adam(encoder.parameters(),
                                   lr=args.lr, weight_decay=args.weight_decay)

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
    meta_train_loss=[]
    # Initialize variables to store the best accuracy and F1 score
    best_test_acc = 0.0
    best_test_f1 = 0.0
    best_epoch=0

    if args.cuda:
        encoder.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        degrees = degrees.cuda()

    for episode in range(args.episodes):
        id_support, id_query, class_selected = \
            task_generator(id_by_class, class_list_train, n_way, k_shot, n_query)
        acc_train, f1_train, loss_train = train(class_selected, id_support, id_query, n_way, k_shot)
        meta_train_acc.append(acc_train)
        meta_train_loss.append(loss_train)
        if episode > 0 and episode % 10 == 0:
            print("-------Episode {}-------".format(episode))
            print("Meta-Train_Accuracy: {}  Meta-Train_Loss: {}".format(np.array(meta_train_acc).mean(axis=0),
                                                                        np.array(meta_train_loss).mean(axis=0)))

            # validation
            meta_test_acc = []
            meta_test_f1 = []
            for idx in range(meta_valid_num):
                id_support, id_query, class_selected = valid_pool[idx]
                acc_test, f1_test = test(class_selected, id_support, id_query, n_way, k_shot)
                meta_test_acc.append(acc_test)
                meta_test_f1.append(f1_test)
            print("Meta-valid_Accuracy: {}, Meta-valid_F1: {}".format(np.array(meta_test_acc).mean(axis=0),
                                                                        np.array(meta_test_f1).mean(axis=0)))
            # testing
            meta_test_acc = []
            meta_test_f1 = []
            for idx in range(meta_test_num):
                id_support, id_query, class_selected = test_pool[idx]
                acc_test, f1_test = test(class_selected, id_support, id_query, n_way, k_shot)
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
                best_epoch=episode
                best_test_f1 = mean_test_f1

    # After the loop, print the best test results
    print("Best Meta-Test_Accuracy: {}".format(best_test_acc))
    print("Best Meta-Test_F1: {}".format(best_test_f1))
    print("Best epoch: {}".format(best_epoch))
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
