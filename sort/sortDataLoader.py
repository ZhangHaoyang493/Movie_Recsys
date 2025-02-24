import os
import torch
from torch import optim, nn, utils, Tensor
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
                 eval_data_path: str,
                 type='train'): #, one_hot: bool=False, user_num:int =6040, item_num:int =3952):
        

        
        self.all_data = []
        self.item_info = pickle.load(open(movie_info_path, 'rb'))
        self.user_info = pickle.load(open(user_info_path, 'rb'))
        if type == 'train':
            train_readlist: dict = pickle.load(open(train_readlist_path, 'rb'))
            for userid in train_readlist.keys():
                for item_info in train_readlist[userid]:
                    self.all_data.append([userid, item_info[0], item_info[1], item_info[2]])
        elif type == 'test':
            eval_data: dict = pickle.load(open(eval_data_path, 'rb'))
            for userid in eval_data.keys():
                for item_info in eval_data[userid]:
                    self.all_data.append([userid, item_info[0], item_info[1], item_info[2]])

        # self.one_hot = one_hot
        # self.user_num = user_num
        # self.item_num = item_num
        
        

        self.age_dict = {
            1: 0,
            18: 1,
            25: 2,
            35: 3,
            45: 4,
            50: 5,
            56: 6,
        }

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
        userid, itemid, score, _ = self.all_data[index]

        
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
        
        # if not self.one_hot:
        return {
            'userid': torch.tensor([int(userid)]),
            'itemid': torch.tensor([int(itemid)]),
            'score': torch.tensor([float(score)]),
            'gender': torch.tensor([user_gender]),
            'user_age': torch.tensor([self.age_dict[user_age]]),
            'user_occupation': torch.tensor([user_occupation]),
            'item_kind': torch.tensor(item_kind),
            'label': torch.tensor([1 if score >= 4 else 0])
        }
        # else:
        #     user_id_one_hot = torch.sparse_coo_tensor(torch.tensor([[int(userid) - 1]]), torch.tensor([1]), (self.user_num))
        #     item_id_one_hot = torch.sparse_coo_tensor(torch.tensor([[int(itemid) - 1]]), torch.tensor([1]), (self.item_num))
        #     user_gender_one_hot = torch.sparse_coo_tensor(torch.tensor([[int(user_gender) - 1]]), torch.tensor([1]), (2,))
        #     user_age_one_hot = torch.sparse_coo_tensor(torch.tensor([[int(self.age_dict[user_age]) - 1]]), torch.tensor([1]), (len(self.age_dict.keys()),))
        #     user_occupation_one_hot = torch.sparse_coo_tensor(torch.tensor([[user_occupation - 1]]), torch.tensor([1]), (21,))
        #     item_kind_one_hot = []
        #     for i in item_kind:
        #         item_id_one_hot.append(
        #             torch.sparse_coo_tensor(torch.tensor([[i - 1]]), torch.tensor([1 if i else 0]), (len(self.gene_dict.keys())))
        #         )
        #     return {
        #         'userid': user_id_one_hot,
        #         'itemid': item_id_one_hot,
        #         'score': torch.tensor([float(score)]),
        #         'gender': user_gender_one_hot,
        #         'user_age': user_age_one_hot,
        #         'user_occupation': user_occupation_one_hot,
        #         'item_kind': torch.tensor(item_id_one_hot),
        #         'label': torch.tensor([1 if score >= 4 else 0])
        #     } 


def get_sort_dataloader(batch_size: int=1, num_workers:int = 4, type: str='train'):
    dataset = SortModelDataLoader(
        '/Users/zhanghaoyang04/Desktop/Movie_Recsys/cache/train_readlist.pkl',
        '/Users/zhanghaoyang04/Desktop/Movie_Recsys/cache/movie_info.pkl',
        '/Users/zhanghaoyang04/Desktop/Movie_Recsys/cache/user_info.pkl',
        '/Users/zhanghaoyang04/Desktop/Movie_Recsys/cache/val_data.pkl',
        type=type
    )
    if type == 'test':
        return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    elif type == 'train':
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)