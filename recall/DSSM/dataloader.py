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
                self.all_data.append((userid, item_info[0]))

        self.gene_dict = {
            'Action': 0, 'Adventure': 1, 'Animation': 2, "Children's": 3,
            'Comedy': 4, 'Crime': 5, 'Documentary': 6, 'Drama': 7,
            'Fantasy': 8, 'Film-Noir': 9, 'Horror': 10, 'Musical': 11,
            'Mystery': 12, 'Romance': 13, 'Sci-Fi': 14, 'Thriller': 15,
            'War': 16, 'Western': 17,
        }

        self.data_type = data_type

        # 正常的负采样，困难负样本、简单负样本
        if self.data_type == 'random':
            self.pos_sample = []
            self.neg_sample = []
            for userid in train_readlist.keys():
                for item_info in train_readlist[userid]:
                    if item_info[1] >= 4.0:
                        # if userid not in self.pos_sample.keys():
                        #     self.pos_sample[userid] = []
                        self.pos_sample.append((userid, item_info[0]))
                    self.neg_sample.append(item_info[0])
        
        # 发现movies.dat里面的item id不是连续的，我们将其映射为连续的
        # all_items_id = [int(i[0]) for i in self.item_info.values()]
        # self.item_id_2_int = dict(list(zip(all_items_id, range(len(all_items_id)))))
        # self.int_2_item_id = dict(list(zip(range(len(all_items_id)), all_items_id)))
        # pickle.dump(self.item_id_2_int, open('/Users/zhanghaoyang/Desktop/Movie_Recsys/cache/item_id_2_int.pkl', 'wb'))
        # pickle.dump(self.int_2_item_id, open('/Users/zhanghaoyang/Desktop/Movie_Recsys/cache/int_2_item_id.pkl', 'wb'))

        # self.item_id_2_int = pickle.load(open('/Users/zhanghaoyang/Desktop/Movie_Recsys/cache/item_id_2_int.pkl', 'rb'))
        self.neg_sample_num = neg_sample_num

    def __len__(self):
        return len(self.all_data) if self.data_type == 'in_batch' else len(self.pos_sample)
    
    def in_batch_data(self, index):
        # user_info : '1': ['1', 'F', '1', '10', '48067']
        # movie_info: '1': ['1', 'Toy Story (1995)', ['Animation', "Children's", 'Comedy']]
        userid, itemid = self.all_data[index]
        user_info = self.user_info[userid]
        # 对user的info进行数字化
        # zip-code暂不知怎么处理
        user_info = torch.tensor([int(user_info[0]), 0 if user_info[1] == 'F' else 1, int(user_info[2]), int(user_info[3])])
        item_info = self.item_info[itemid]
        # item info数字化
        # 电影名和电影类型并没有处理
        item_info = torch.tensor([int(item_info[0])])
        return {
            'userid': torch.tensor([int(userid)], dtype=torch.int),
            'itemid': torch.tensor([int(itemid)], dtype=torch.int),
            'user_feature': user_info,
            'item_feature': item_info
        }

    def random_neg_sample(self, index):
        userid, itemid = self.pos_sample[index]
        user_info = self.user_info[userid]
        # 对user的info进行数字化
        # zip-code暂不知怎么处理
        user_info = torch.tensor([int(user_info[0]), 0 if user_info[1] == 'F' else 1, int(user_info[2]), int(user_info[3])])
        item_info = self.item_info[itemid]
        # item info数字化
        # 电影名和电影类型并没有处理
        item_info = torch.tensor([int(item_info[0])])

        neg_sample = random.sample(self.neg_sample, self.neg_sample_num)
        neg_sample = [int(k) for k in neg_sample]

        return {
            'userid': torch.tensor([int(userid)], dtype=torch.int),
            'itemid': torch.tensor([int(itemid)], dtype=torch.int),
            'neg_sample': torch.tensor(neg_sample),
            'user_feature': user_info,
            'item_feature': item_info
        }


    def __getitem__(self, index):
        if self.data_type == 'in_batch':
            return self.in_batch_data(index)
        elif self.data_type == 'random':
            return self.random_neg_sample(index)

    
