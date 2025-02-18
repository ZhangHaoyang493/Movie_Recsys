import os
import torch
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
import pickle
from tqdm import tqdm
import numpy as np
import random
# from baseRecall import BaseRecall


class DSSMDataLoader(Dataset):
    def __init__(self, 
                 train_readlist_path: str,
                 movie_info_path: str,
                 user_info_path: str,
                 data_type: str,
                 neg_sample_num: int):
        
        assert data_type in ['in_batch', 'random']

        train_readlist: dict = pickle.load(open(train_readlist_path, 'rb'))
        self.all_data = []
        self.item_info = pickle.load(open(movie_info_path, 'rb'))
        self.user_info = pickle.load(open(user_info_path, 'rb'))
        for userid in train_readlist.keys():
            for item_info in train_readlist[userid]:
                self.all_data.append([item_info[0], item_info[1], item_info[2]])

        self.gene_dict = {
            'Action': 0, 'Adventure': 1, 'Animation': 2, "Children's": 3,
            'Comedy': 4, 'Crime': 5, 'Documentary': 6, 'Drama': 7,
            'Fantasy': 8, 'Film-Noir': 9, 'Horror': 10, 'Musical': 11,
            'Mystery': 12, 'Romance': 13, 'Sci-Fi': 14, 'Thriller': 15,
            'War': 16, 'Western': 17,
        }
        
        
        self.neg_sample_num = neg_sample_num

    def __len__(self):
        return len(self.all_data)
    
    


    def __getitem__(self, index):
        if self.data_type == 'in_batch':
            return self.in_batch_data(index)
        elif self.data_type == 'random':
            return self.random_neg_sample(index)

    
