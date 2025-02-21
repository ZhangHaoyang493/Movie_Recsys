import os
import torch
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
import pickle
from tqdm import tqdm
import numpy as np


class BaseSort():
    def __init__(self, train_readlist_path: str):
        super().__init__()

        train_readlist = pickle.load(open(train_readlist_path, 'rb'))
        self.weight = {}
        for userid in train_readlist.keys():
            self.weight[userid] = len(train_readlist[userid])
    


    
    
    def eval_gauc(self, eval_data_path: str):
        eval_data = pickle.load(open(eval_data_path, 'rb'))

        for userid in eval_data:
