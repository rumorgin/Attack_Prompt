from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
import collections
import torch.nn.functional as F
import numpy as np
import random
import glob
from PIL import Image
import argparse
import warnings
import os
import torch
import sys
from utils import *

class OmniglotDataset(Dataset):

    def __init__(self, data_path, batch_size, n_way=10, k_shot=2, q_query=1):

        self.file_list = self.get_file_list(data_path)
        self.batch_size = batch_size
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query

    def __len__(self):
        return len(self.file_list) // self.batch_size

    def __getitem__(self, index):
        return self.get_one_task_data()

    def get_file_list(self, data_path):
        """
        Get all fonts list.
        Args:
            data_path: Omniglot Data path

        Returns: fonts list

        """
        return [f for f in glob.glob(data_path + "**/character*", recursive=True)]

    def get_one_task_data(self):
        """
        Get ones task maml data, include one batch support images and labels, one batch query images and labels.
        Returns: support_data, query_data

        """
        img_dirs = random.sample(self.file_list, self.n_way)
        support_data = []
        query_data = []

        support_image = []
        support_label = []
        query_image = []
        query_label = []

        for label, img_dir in enumerate(img_dirs):
            img_list = [f for f in glob.glob(img_dir + "**/*.png", recursive=True)]
            images = random.sample(img_list, self.k_shot + self.q_query)

            # Read support set
            for img_path in images[:self.k_shot]:
                image = Image.open(img_path)
                image = np.array(image)
                image = np.expand_dims(image / 255., axis=0)
                support_data.append((image, label))

            # Read query set
            for img_path in images[self.k_shot:]:
                image = Image.open(img_path)
                image = np.array(image)
                image = np.expand_dims(image / 255., axis=0)
                query_data.append((image, label))

        # shuffle support set
        random.shuffle(support_data)
        for data in support_data:
            support_image.append(data[0])
            support_label.append(data[1])

        # shuffle query set
        random.shuffle(query_data)
        for data in query_data:
            query_image.append(data[0])
            query_label.append(data[1])

        return np.array(support_image), np.array(support_label), np.array(query_image), np.array(query_label)

class ConvBlock(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv2d = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.max_pool(x)

        return x


def ConvBlockFunction(input, w, b, w_bn, b_bn):
    x = F.conv2d(input, w, b, padding=1)
    x = F.batch_norm(x, running_mean=None, running_var=None, weight=w_bn, bias=b_bn, training=True)
    x = F.relu(x)
    output = F.max_pool2d(x, kernel_size=2, stride=2)

    return output


class Classifier(nn.Module):
    def __init__(self, in_ch, n_way):
        super(Classifier, self).__init__()
        self.conv1 = ConvBlock(in_ch, 64)
        self.conv2 = ConvBlock(64, 64)
        self.conv3 = ConvBlock(64, 64)
        self.conv4 = ConvBlock(64, 64)
        self.logits = nn.Linear(64, n_way)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.shape[0], -1)
        x = self.logits(x)

        return x

    def functional_forward(self, x, params):
        x = ConvBlockFunction(x, params[f'conv1.conv2d.weight'], params[f'conv1.conv2d.bias'],
                              params.get(f'conv1.bn.weight'), params.get(f'conv1.bn.bias'))
        x = ConvBlockFunction(x, params[f'conv2.conv2d.weight'], params[f'conv2.conv2d.bias'],
                              params.get(f'conv2.bn.weight'), params.get(f'conv2.bn.bias'))
        x = ConvBlockFunction(x, params[f'conv3.conv2d.weight'], params[f'conv3.conv2d.bias'],
                              params.get(f'conv3.bn.weight'), params.get(f'conv3.bn.bias'))
        x = ConvBlockFunction(x, params[f'conv4.conv2d.weight'], params[f'conv4.conv2d.bias'],
                              params.get(f'conv4.bn.weight'), params.get(f'conv4.bn.bias'))

        x = x.view(x.shape[0], -1)
        x = F.linear(x, params['logits.weight'], params['logits.bias'])

        return x


def maml_train(model, support_images, support_labels, query_images, query_labels, inner_step, args, optimizer, is_train=True):
    """
    Train the model using MAML method.
    Args:
        model: Any model
        support_images: several task support images
        support_labels: several  support labels
        query_images: several query images
        query_labels: several query labels
        inner_step: support data training step
        args: ArgumentParser
        optimizer: optimizer
        is_train: whether train

    Returns: meta loss, meta accuracy

    """
    meta_loss = []
    meta_acc = []

    for support_image, support_label, query_image, query_label in zip(support_images, support_labels, query_images, query_labels):

        fast_weights = collections.OrderedDict(model.named_parameters())
        for _ in range(inner_step):
            # Update weight
            support_logit = model.functional_forward(support_image, fast_weights)
            support_loss = nn.CrossEntropyLoss().cuda()(support_logit, support_label)
            grads = torch.autograd.grad(support_loss, fast_weights.values(), create_graph=True)
            fast_weights = collections.OrderedDict((name, param - args.inner_lr * grad)
                                                   for ((name, param), grad) in zip(fast_weights.items(), grads))

        # Use trained weight to get query loss
        query_logit = model.functional_forward(query_image, fast_weights)
        query_prediction = torch.max(query_logit, dim=1)[1]

        query_loss = nn.CrossEntropyLoss().cuda()(query_logit, query_label)
        query_acc = torch.eq(query_label, query_prediction).sum() / len(query_label)

        meta_loss.append(query_loss)
        meta_acc.append(query_acc.data.cpu().numpy())

    # Zero the gradient
    optimizer.zero_grad()
    meta_loss = torch.stack(meta_loss).mean()
    meta_acc = np.mean(meta_acc)

    if is_train:
        meta_loss.backward()
        optimizer.step()

    return meta_loss, meta_acc

def get_dataset(args):
    """
    Get maml dataset.
    Args:
        args: ArgumentParser

    Returns: dataset

    """
    train_dataset = OmniglotDataset(args.train_data_dir, args.task_num,
                                    n_way=args.n_way, k_shot=args.k_shot, q_query=args.q_query)
    val_dataset = OmniglotDataset(args.val_data_dir, args.val_task_num,
                                  n_way=args.n_way, k_shot=args.k_shot, q_query=args.q_query)

    return train_dataset, val_dataset

def get_model(args, dev):
    """
    Get model.
    Args:
        args: ArgumentParser
        dev: torch dev

    Returns: model

    """
    model = Classifier(1, args.n_way).cuda()
    model.to(dev)

    return model



if __name__ == '__main__':

    sys.path.append(os.getcwd())

    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=str, default='0', help='Select gpu device.')
    parser.add_argument('--train_data_dir', type=str,
                        default="./data/Omniglot/images_background/",
                        help='The directory containing the train image data.')
    parser.add_argument('--val_data_dir', type=str,
                        default="./data/Omniglot/images_evaluation/",
                        help='The directory containing the validation image data.')
    parser.add_argument('--summary_path', type=str,
                        default="./summary",
                        help='The directory of the summary writer.')

    parser.add_argument('--task_num', type=int, default=32,
                        help='Number of task per train batch.')
    parser.add_argument('--val_task_num', type=int, default=16,
                        help='Number of task per test batch.')
    parser.add_argument('--num_workers', type=int, default=12, help='The number of torch dataloader thread.')

    parser.add_argument('--epochs', type=int, default=150,
                        help='The training epochs.')
    parser.add_argument('--inner_lr', type=float, default=0.04,
                        help='The learning rate of of the support set.')
    parser.add_argument('--outer_lr', type=float, default=0.001,
                        help='The learning rate of of the query set.')

    parser.add_argument('--n_way', type=int, default=5,
                        help='The number of class of every task.')
    parser.add_argument('--k_shot', type=int, default=1,
                        help='The number of support set image for every task.')
    parser.add_argument('--q_query', type=int, default=1,
                        help='The number of query set image for every task.')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    seed_torch(1206)

    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = get_model(args, dev)
    train_dataset, val_dataset = get_dataset(args)

    train_loader = DataLoader(train_dataset, batch_size=args.task_num, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.val_task_num, shuffle=False, num_workers=args.num_workers)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, args.outer_lr)
    best_acc = 0

    model.train()
    for epoch in range(args.epochs):
        train_acc = []
        val_acc = []
        train_loss = []
        val_loss = []

        train_bar = tqdm(train_loader)
        for support_images, support_labels, query_images, query_labels in train_bar:
            train_bar.set_description("epoch {}".format(epoch + 1))
            # Get variables
            support_images = support_images.float().to(dev)
            support_labels = support_labels.long().to(dev)
            query_images = query_images.float().to(dev)
            query_labels = query_labels.long().to(dev)

            loss, acc = maml_train(model, support_images, support_labels, query_images, query_labels,
                                   1, args, optimizer)

            train_loss.append(loss.item())
            train_acc.append(acc)
            train_bar.set_postfix(loss="{:.4f}".format(loss.item()))

        for support_images, support_labels, query_images, query_labels in val_loader:

            # Get variables
            support_images = support_images.float().to(dev)
            support_labels = support_labels.long().to(dev)
            query_images = query_images.float().to(dev)
            query_labels = query_labels.long().to(dev)

            loss, acc = maml_train(model, support_images, support_labels, query_images, query_labels,
                                   3, args, optimizer, is_train=False)

            # Must use .item()  to add total loss, or will occur GPU memory leak.
            # Because dynamic graph is created during forward, collect in backward.
            val_loss.append(loss.item())
            val_acc.append(acc)

        print("=> loss: {:.4f}   acc: {:.4f}   val_loss: {:.4f}   val_acc: {:.4f}".
              format(np.mean(train_loss), np.mean(train_acc), np.mean(val_loss), np.mean(val_acc)))

        if np.mean(val_acc) > best_acc:
            best_acc = np.mean(val_acc)
            torch.save(model, 'best.pt')