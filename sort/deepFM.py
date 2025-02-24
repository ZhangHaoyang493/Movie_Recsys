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

class DeepFMSortModel(nn.Module):
    def __init__(self, 
                user_num,
                item_num,
                user_gender_num,
                user_age_num, 
                user_occupation_num, 
                item_kind_num, dim:int=8):
        super().__init__()

        self.user_id_para = nn.Embedding(user_num, 1 + dim)
        self.item_id_para = nn.Embedding(item_num, 1 + dim)
        self.age_para = nn.Embedding(user_age_num, 1 + dim)
        self.gender_para = nn.Embedding(user_gender_num, 1 + dim)
        self.occupation_para = nn.Embedding(user_occupation_num, 1 + dim)
        self.kind_para = nn.Embedding(item_kind_num, 1 + dim)

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


    def binary_cross_entropy_loss(self, logit, label):
        return (-(label * torch.log(logit + 1e-6) + (1 - label) * torch.log(1 - logit + 1e-6))).sum() / label.shape[0]

    def get_logit(self, data):
        b, _ = data['userid'].shape
        user_weight = self.user_id_para(data['userid']) # bx1x9
        item_weight = self.item_id_para(data['itemid'])
        age_weight = self.age_para(data['user_age'])
        gender_weight = self.gender_para(data['gender'])
        occ_weight = self.occupation_para(data['user_occupation'])
        kind_weight = self.kind_para(data['item_kind']) # bx10x9
        kind_weight_tensor = torch.ones_like(data['item_kind']) # bx10
        kind_weight_tensor[data['item_kind'] == 0] = 0
        kind_weight_tensor = kind_weight_tensor.unsqueeze(-1) # bx10x1
        kind_weight = kind_weight * kind_weight_tensor
        one_order_weight = user_weight[..., 0] + item_weight[..., 0] + age_weight[..., 0] +\
              occ_weight[..., 0] + torch.sum(kind_weight[..., 0], dim=-1, keepdim=True) + gender_weight[..., 0]
        two_order_emb = torch.concat([user_weight[..., 1:], item_weight[..., 1:], age_weight[..., 1:],
                                      occ_weight[..., 1:], kind_weight[..., 1:], gender_weight[..., 1:]], dim=1)  # bx15xdim
        two_order_weight = torch.matmul(two_order_emb, torch.transpose(two_order_emb, 1, 2)) # bx15x15
        b, weight_dim, _ = two_order_weight.shape
        two_order_weight = two_order_weight * ((torch.ones((weight_dim, weight_dim)) - torch.eye(weight_dim)).unsqueeze(0))
        two_order_weight = torch.sum(two_order_weight, dim=-1)
        two_order_weight = torch.sum(two_order_weight, dim=-1)
        two_order_weight = two_order_weight / 2.0

        mlp_weight = self.mlp(two_order_emb.view(b, -1))

        return one_order_weight.view(-1, 1) + two_order_weight.view(-1, 1) + mlp_weight.view(-1, 1)
    
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

    model = DeepFMSortModel(user_num, item_num, user_age_num=7 + 1, user_gender_num=2 + 1, user_occupation_num=21 + 1, item_kind_num=18 + 1)
    dataloader = get_sort_dataloader(batch_size=batch_size, num_workers=0)
    eval_dataloader = get_sort_dataloader(type='test', num_workers=0)
    trainer = Trainer(model, 'Deep', './log')
    trainer.set_config(epoch_num, lr, 'adam', dataloader, eval_dataloader, 'cosin', lr_min, save_epoch=epoch_num, eval_epoch=1)
    trainer.train()
    trainer.eval()


    
  