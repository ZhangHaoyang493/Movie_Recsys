import os
import torch
from torch import optim, nn, utils, Tensor
from torch.utils.data import Dataset, DataLoader
import pickle
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score

import sys
sys.path.append('/data2/zhy/Movie_Recsys/sort')
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

class ActUnit(nn.Module):
    def __init__(self, in_dim):
        super().__init__()

        self.mlps = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, i1, i2):
        # i1: b x seq_len x dim 历史物品 i2: b x seq_len x dim 目标物品
        i3 = i1 - i2
        i_in = torch.concat([i1, i3, i2], dim=-1)
        i_out = self.mlps(i_in)  # b x seq_len x 1
        return i1 * i_out


class DINSortModel(nn.Module):
    def __init__(self, 
                user_num,
                item_num,
                user_gender_num,
                user_age_num, 
                user_occupation_num, 
                item_kind_num, his_len:int=5, 
                dim:int=8, kind_len:int=10):
        super().__init__()

        self.user_id_para = nn.Embedding(user_num, dim)
        self.item_id_para = nn.Embedding(item_num, dim)
        self.age_para = nn.Embedding(user_age_num, dim)
        self.gender_para = nn.Embedding(user_gender_num, dim)
        self.occupation_para = nn.Embedding(user_occupation_num, dim)
        self.kind_para = nn.Embedding(item_kind_num, dim)

        self.his_len = his_len
        self.kind_len = kind_len

        self.his_item_dim = (1 + self.kind_len) * dim

        self.dim = dim
        all_dim = dim * (5 + self.kind_len) + self.his_item_dim
        self.mlp = nn.Sequential(
            nn.Linear(all_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.act_unit = ActUnit(self.his_item_dim * 3)


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


        his_item_id = data['item_id_his']  # b x his_len
        his_item_kind = data['item_kind_his'] # b x his_len x kind_len
        his_feature = []
        for i in range(self.his_len):
            his_item_id_now = his_item_id[:, i].view(-1, 1)
            his_item_kind_now = his_item_kind[:, i].view(-1, self.kind_len)
            his_item_weight = self.item_id_para(his_item_id_now) # bx1x8
            his_item_kind_weight = self.kind_para(his_item_kind_now) #bx10x8

            weight_tensor = torch.ones_like(his_item_kind_now)
            weight_tensor[his_item_kind_now == 0] = 0
            weight_tensor = weight_tensor.unsqueeze(-1)
            his_item_kind_weight = his_item_kind_weight * weight_tensor

            id_kind_feature = torch.concat([his_item_weight, his_item_kind_weight], dim=-2).view(b, -1)
            his_feature.append(id_kind_feature.view(-1, 1, self.his_item_dim))
        his_feature = torch.cat(his_feature, dim=1) # bx his_len x his_dim
        item_feature_now = torch.concat([item_weight, kind_weight], dim=-2).view(b, 1, -1).repeat(1, self.his_len, 1)

        act_weight = self.act_unit(his_feature, item_feature_now) # b x his_len x 1
        his_feature = his_feature * act_weight
        his_feature_pooling = torch.sum(his_feature, dim=-2)  # bx 1 x his_dim



        all_feature = torch.concat([user_weight, item_weight, age_weight, gender_weight, occ_weight, kind_weight], dim=1) #bx15x8
        all_feature = all_feature.view(b, -1)
        all_feature = torch.concat([all_feature, his_feature_pooling], dim=-1)
        logit = self.mlp(all_feature) # bx1
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
    batch_size = 128
    lr = 1e-3
    lr_min = 1e-4
    device = 'cpu'
    # data_type = 'in_batch'
    neg_sample_num = 20
    his_len = 5
    kind_len = 10

    model = DINSortModel(user_num, item_num, user_age_num=7 + 1, user_gender_num=2 + 1, user_occupation_num=21 + 1, item_kind_num=18 + 1, his_len=his_len, kind_len=kind_len)
    dataloader = get_sort_dataloader(batch_size=batch_size, num_workers=0, his_len=his_len, kind_len=kind_len)
    eval_dataloader = get_sort_dataloader(type='test', num_workers=0, his_len=his_len, kind_len=kind_len)
    trainer = Trainer(model, 'Deep', './log')
    trainer.set_config(epoch_num, lr, 'adam', dataloader, eval_dataloader, 'cosin', lr_min, save_epoch=epoch_num, eval_epoch=1)
    trainer.train()
    trainer.eval()


    
  