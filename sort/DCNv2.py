import os
import torch
from torch import optim, nn, utils, Tensor
from torch.utils.data import Dataset, DataLoader
import pickle
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score

import sys
sys.path.append('/Users/zhanghaoyang/Desktop/Movie_Recsys/sort')
from sortDataLoader import get_sort_dataloader
from trainer import Trainer

# {
#             'userid': torch.tensor([int(userid)]),
#             'itemid': torch.tensor([int(itemid)]),
#             'score': torch.tensor([float(score)]),
#             'gender': torch.tensor([user_gender]),
#             'user_age': torch.tensor([self.age_dict[user_age]]),
#             'user_occupation': torch.tensor([user_occupation]),
#             'item_kind': torch.tensor(item_kind),
#             'label': torch.tensor([1 if score >= 4 else 0])
#         }
class DCNBlock(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.mlp = nn.Linear(in_dim, in_dim)

    def forward(self, x0, xl):
        # x0: bxdimx1   xl: bxdimx1
        cross_fea = x0 * self.mlp(xl) + xl
        return x0, cross_fea



class DCNv2SortModel(nn.Module):
    def __init__(self, 
                user_num,
                item_num,
                user_gender_num,
                user_age_num, 
                user_occupation_num, 
                item_kind_num, dim:int=8):
        super().__init__()

        self.user_id_para = nn.Embedding(user_num, dim)
        self.item_id_para = nn.Embedding(item_num, dim)
        self.age_para = nn.Embedding(user_age_num, dim)
        self.gender_para = nn.Embedding(user_gender_num, dim)
        self.occupation_para = nn.Embedding(user_occupation_num, dim)
        self.kind_para = nn.Embedding(item_kind_num, dim)

        self.dim = dim
        all_dim = dim * 15
        self.mlp = nn.Sequential(
            nn.Linear(all_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.dcn = nn.ModuleList(
            [
                DCNBlock(all_dim),
                DCNBlock(all_dim),
                DCNBlock(all_dim),
            ]
        )


    def binary_cross_entropy_loss(self, logit, label):
        return (-(label * torch.log(logit + 1e-6) + (1 - label) * torch.log(1 - logit + 1e-6))).sum() / label.shape[0]

    def get_logit(self, data):
        b, _ = data['userid'].shape

        user_weight = self.user_id_para(data['userid']) # bx1x8
        item_weight = self.item_id_para(data['itemid']) # bx1x8
        age_weight = self.age_para(data['user_age']) # bx1x8
        gender_weight = self.gender_para(data['gender']) # bx1x8
        occ_weight = self.occupation_para(data['user_occupation']) # bx1x8
        kind_weight = self.kind_para(data['item_kind']) #bx10x8

        kind_data = data['item_kind'] # bx10
        weight_tensor = torch.ones_like(kind_data)
        weight_tensor[kind_data == 0] = 0
        weight_tensor = weight_tensor.unsqueeze(-1)
        kind_weight = kind_weight * weight_tensor

        all_feature = torch.concat([user_weight, item_weight, age_weight, gender_weight, occ_weight, kind_weight], dim=1) #bx15x8
        all_feature = all_feature.view(b, -1)

        x0, xl = all_feature, all_feature
        for dcn in self.dcn:
            x0, xl = dcn(x0, xl)

        logit = self.mlp(xl) # bx1
        return logit
    
    def forward(self, data):
        
        logit = torch.sigmoid(self.get_logit(data))


        label = data['label']

        # logit = torch.sigmoid(user_weight + item_weight + age_weight + occ_weight + kind_weight + gender_weight)


        loss = self.binary_cross_entropy_loss(logit, label)
        auc = self.auc(logit.view(-1, 1), label)
        return {'loss': loss, 'auc': auc}
    
    def eval_(self, data):
        logit = torch.sigmoid(self.get_logit(data))
        label = data['label']
        return logit.view(1).detach().cpu().numpy()[0], label.view(1).detach().cpu().numpy()[0]
    
    def auc(self, logit, label):
        return roc_auc_score(label.detach().cpu().numpy(), logit.detach().cpu().numpy())
    


if __name__ == '__main__':
    item_num = 3952 + 1
    user_num = 6040 + 1
    epoch_num = 15
    batch_size = 256
    lr = 1e-3
    lr_min = 1e-4
    device = 'cpu'
    # data_type = 'in_batch'
    neg_sample_num = 20

    model = DCNv2SortModel(user_num, item_num, user_age_num=7 + 1, user_gender_num=2 + 1, user_occupation_num=21 + 1, item_kind_num=18 + 1)
    dataloader = get_sort_dataloader(batch_size=batch_size, num_workers=0)
    eval_dataloader = get_sort_dataloader(type='test', num_workers=0)
    trainer = Trainer(model, 'DCNv1', './log')
    trainer.set_config(epoch_num, lr, 'adam', dataloader, eval_dataloader, 'cosin', lr_min, save_epoch=epoch_num, eval_epoch=1)
    trainer.train()
    trainer.eval()


    
  