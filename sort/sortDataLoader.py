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


class SortModelDataLoader(Dataset):
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
            'Action': 18, 'Adventure': 1, 'Animation': 2, "Children's": 3,
            'Comedy': 4, 'Crime': 5, 'Documentary': 6, 'Drama': 7,
            'Fantasy': 8, 'Film-Noir': 9, 'Horror': 10, 'Musical': 11,
            'Mystery': 12, 'Romance': 13, 'Sci-Fi': 14, 'Thriller': 15,
            'War': 16, 'Western': 17,
        }
        

    def __len__(self):
        return len(self.all_data)
    
    


    def __getitem__(self, index):
        userid, itemid, score = self.all_data[index]
        user_info = self.user_info[userid]
        item_info = self.item_info[itemid]

        user_gender = user_info[1]
        user_gender = 1 if user_gender == 'M' else 0

        user_age = int(user_info[2])
        user_occupation = int(user_info[3])

        item_kind = item_info[-1]
        item_kind = [self.gene_dict[k] for k in item_kind]
        while len(item_kind) < 10:
            item_kind.append(0)
        
        return {
            'userid': torch.tensor([int(userid)]),
            'itemid': torch.tensor([int(itemid)]),
            'score': torch.tensor([float(score)]),
            'user_age': torch.tensor([user_age]),
            'user_occupation': torch.tensor([user_occupation]),
            'item_kind': torch.tensor(item_kind),
            'label': torch.tensor([1 if score >= 4 else 0])
        }



def get_sort_dataloader(batch_size: int, num_workers:int = 4):
    dataset = SortModelDataLoader(
        '/Users/zhanghaoyang/Desktop/Movie_Recsys/cache/train_readlist.pkl',
        '/Users/zhanghaoyang/Desktop/Movie_Recsys/cache/movie_info.pkl',
        '/Users/zhanghaoyang/Desktop/Movie_Recsys/cache/user_info.pkl',
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return dataloader