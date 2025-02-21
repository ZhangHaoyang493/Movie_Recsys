import os
import torch
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
import pickle
from tqdm import tqdm
import numpy as np

import sys
sys.append('..')
from sortDataLoader import get_sort_dataloader
from baseSort import BaseSortModel

# return {
#             'userid': torch.tensor([int(userid)]),
#             'itemid': torch.tensor([int(itemid)]),
#             'score': torch.tensor([float(score)]),
#             'user_age': torch.tensor([user_age]),
#             'user_occupation': torch.tensor([user_occupation]),
#             'item_kind': torch.tensor(item_kind),
#             'label': torch.tensor([1 if score >= 4 else 0])
#         }

class LRSortModel(BaseSortModel):
    def __init__(self, 
                user_num,
                item_num, 
                user_age_num, 
                user_occupation_num, 
                item_kind_num):
        super().__init__()

        self.user_id_para = nn.Embedding(user_num, 1)
        self.item_id_para = nn.Embedding(item_num, 1)
        self.age_para = nn.Embedding(user_age_num, 1)
        self.occupation_para = nn.Embedding(user_occupation_num, 1)
        self.kind_para = nn.Embedding(item_kind_num, 1)

    def forward(self, data):
        user_weight = self.user_id_para(data['userid'])
        item_weight = self.item_id_para(data['itemid'])
        age_weight = self.age_para(data['user_age'])
        occ_weight = self.age_para(data['user_occupation'])
        kind_weight = self.kind_para(data['item_kind'])
        label = data['label']

        logit = torch.sigmoid(user_weight + item_weight + age_weight + occ_weight + kind_weight)

        return self.binary_cross_entropy_loss(logit, label)

        
        