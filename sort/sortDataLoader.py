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
                 user_fe_path: str,
                 history_len:int=5,
                 kind_len:int=10,
                 type='train'): #, one_hot: bool=False, user_num:int =6040, item_num:int =3952):
        

        
        self.all_data = []
        self.item_info = pickle.load(open(movie_info_path, 'rb'))
        self.user_info = pickle.load(open(user_info_path, 'rb'))
        self.user_fe = pickle.load(open(user_fe_path, 'rb'))
        self.user_pos_history = {}

        self.history_len = history_len
        self.kind_len = kind_len

        train_readlist: dict = pickle.load(open(train_readlist_path, 'rb'))
        # 获取用户正向的阅读历史
        for userid in train_readlist.keys():
            if userid not in self.user_pos_history.keys():
                self.user_pos_history[userid] = []
            for item_info in train_readlist[userid]:
                if item_info[1] >= 4:
                    self.user_pos_history[userid].append((item_info[0], item_info[1], item_info[2]))

        if type == 'train':
            
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
    
    # 返回大于当前时间戳的第一个物品的索引
    def binary_search_history(self, userid, ts_now):
        readlist = self.user_pos_history[userid]
        left = 0
        right = len(readlist) - 1
        ans_index = 0
        while left <= right:
            mid = (left + right) // 2
            if readlist[mid][2] > ts_now:
                right = mid - 1
                ans_index = right + 1
            elif readlist[mid][2] < ts_now:
                left = mid + 1
                ans_index = left
            elif readlist[mid][2] == ts_now:
                return mid
        return ans_index
        
    def get_item_kind(self, itemid):
        item_info = self.item_info[itemid]
        item_kind = item_info[-1]
        item_kind = [self.gene_dict[k] for k in item_kind]
        while len(item_kind) < self.kind_len:
            item_kind.append(0)
        return item_kind
    
    # 返回用户历史阅读列表
    def get_user_his_list(self, userid, ts_now):
        user_his_item_id = []
        user_his_item_kind = []
        history_index = self.binary_search_history(userid, ts_now)
        his_list = self.user_pos_history[userid][:history_index]
        for his in his_list:
            user_his_item_id.append(int(his[0]))
            user_his_item_kind.append(self.get_item_kind(his[0]))
        while len(user_his_item_id) < self.history_len:
            user_his_item_id = [0] + user_his_item_id
            user_his_item_kind = [[0] * self.kind_len] + user_his_item_kind
        user_his_item_id = user_his_item_id[-self.history_len:]
        user_his_item_kind = user_his_item_kind[-self.history_len:]
        return user_his_item_id, user_his_item_kind
    
    def user_actice_equal_frequence_split(self, active):
        split_interval = [[3, 25], [25, 35], [35, 48], [48, 67], [67, 92], [92, 123], [123, 169], [169, 250], [251, 396], [396, 2312]]
        for i, interval in enumerate(split_interval):
            if active >= interval[0] and active < interval[1]:
                return i
        return -1
    
    def user_mean_score_equal_interval_split(self, mean_score):
        # 按照0.5分桶
        return int(mean_score / 0.5)


    def __getitem__(self, index):
        userid, itemid, score, ts_now = self.all_data[index]

        
        user_info = self.user_info[userid]
        item_info = self.item_info[itemid]

        user_gender = user_info[1]
        user_gender = 1 if user_gender == 'M' else 0

        user_age = int(user_info[2])
        user_occupation = int(user_info[3])

        item_kind = self.get_item_kind(itemid)

        his_id, his_kind = self.get_user_his_list(userid, ts_now)

        # {'active': 51, 
        # 'mean_score': 4.176470588235294, 
        # 'std_score': 0.45905420991926166, 
        # 'like_kinds': [['Drama', 21], ["Children's", 18], ['Animation', 16]]}
        user_fe = self.user_fe[userid]
        user_act = self.user_fe[userid]['active']
        user_act = self.user_actice_equal_frequence_split(user_act)
        # assert user_act >= 0

        user_mean_score = self.user_fe[userid]['mean_score']
        user_mean_score = self.user_mean_score_equal_interval_split(user_mean_score)
        
        user_std_score = self.user_fe[userid]['std_score']
        user_like_kinds = self.user_fe[userid]['like_kinds']
        user_like_kinds = [self.gene_dict[i[0]] for i in user_like_kinds]
        
        
        # if not self.one_hot:
        return {
            'userid': torch.tensor([int(userid)]),
            'itemid': torch.tensor([int(itemid)]),
            'score': torch.tensor([float(score)]),
            'gender': torch.tensor([user_gender]),
            'user_age': torch.tensor([self.age_dict[user_age]]),
            'user_occupation': torch.tensor([user_occupation]),
            'item_kind': torch.tensor(item_kind),
            'item_id_his': torch.tensor(his_id),
            'item_kind_his': torch.tensor(his_kind),
            'label': torch.tensor([1 if score >= 4 else 0]),
            'label_lower_3': torch.tensor([1 if score <= 3 else 0]),
            'label_4': torch.tensor([1 if score == 4 else 0]),
            'label_5': torch.tensor([1 if score == 5 else 0]),
            # 'user_act'
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


def get_sort_dataloader(batch_size: int=1, num_workers:int = 4, type: str='train', his_len=5, kind_len=10):
    dataset = SortModelDataLoader(
        '/Users/zhanghaoyang04/Desktop/Movie_Recsys/cache/train_readlist.pkl',
        '/Users/zhanghaoyang04/Desktop/Movie_Recsys/cache/movie_info.pkl',
        '/Users/zhanghaoyang04/Desktop/Movie_Recsys/cache/user_info.pkl',
        '/Users/zhanghaoyang04/Desktop/Movie_Recsys/cache/val_data.pkl',
        '/Users/zhanghaoyang04/Desktop/Movie_Recsys/cache/user_fe.pkl',
        type=type,
        history_len=his_len,
        kind_len=kind_len,
    )
    if type == 'test':
        return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    elif type == 'train':
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)