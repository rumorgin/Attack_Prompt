import os
import numpy as np
import random
import torch
import torch.nn.functional as F

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
