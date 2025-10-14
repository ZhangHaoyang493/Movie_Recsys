import os
import torch
from torch import optim, nn, utils, Tensor
from torch.utils.data import Dataset, DataLoader
import pickle
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

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

class MMoESortModel(nn.Module):
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
        
        expert_dims = [all_dim, 64, 32]
        self.expert_score_4_1 = self.deep_net(dims=expert_dims)
        self.expert_score_4_2 = self.deep_net(dims=expert_dims)
        self.expert_score_5_1 = self.deep_net(dims=expert_dims)
        self.expert_score_5_2 = self.deep_net(dims=expert_dims)
        self.expert_like_1 = self.deep_net(dims=expert_dims)
        self.expert_like_2 = self.deep_net(dims=expert_dims)
        self.expert_share_1 = self.deep_net(dims=expert_dims)
        self.expert_share_2 = self.deep_net(dims=expert_dims)

        tower_dim = [32, 16, 1]
        self.tower_score4 = self.deep_net(tower_dim)
        self.tower_score5 = self.deep_net(tower_dim)
        self.tower_like = self.deep_net(tower_dim)

        gate_dim = [all_dim, 32, 4]
        self.gate_expert_4 = self.deep_net(gate_dim)
        self.gate_expert_5 = self.deep_net(gate_dim)
        self.gate_expert_like = self.deep_net(gate_dim)
        self.gate_expert_share = self.deep_net([all_dim, 32, 8])


        self.loss_weight = nn.Parameter(torch.randn((3,)))


        
    def deep_net(self, dims):
        model_list = []
        for i in range(1, len(dims)):
            model_list.append(nn.Linear(dims[i - 1], dims[i]))
            model_list.append(nn.ReLU())
        model_list.pop()
        return nn.Sequential(*model_list)

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
        # logit = self.mlp(all_feature) # bx1

        expert_41 = self.expert_score_4_1(all_feature).view(b, 1, -1)
        expert_42 = self.expert_score_4_2(all_feature).view(b, 1, -1)
        expert_51 = self.expert_score_5_1(all_feature).view(b, 1, -1)
        expert_52 = self.expert_score_5_2(all_feature).view(b, 1, -1)
        expert_like1 = self.expert_like_1(all_feature).view(b, 1, -1)
        expert_like2 = self.expert_like_2(all_feature).view(b, 1, -1) # bx32
        expert_share1 = self.expert_share_1(all_feature).view(b, 1, -1)
        expert_share2 = self.expert_share_1(all_feature).view(b, 1, -1)

        fea_score_4 = torch.cat([expert_41, expert_42, expert_share1, expert_share2], dim=1) # bx4x32
        fea_score_5 = torch.cat([expert_51, expert_52, expert_share1, expert_share2], dim=1)
        fea_score_like = torch.cat([expert_41, expert_42, expert_share1, expert_share2], dim=1)
        fea_score_share = torch.cat([expert_41, expert_42, expert_51, expert_52, expert_like1, expert_like2, expert_share1, expert_share2], dim=1)


        gate1_out = F.softmax(self.gate_expert_4(all_feature), dim=-1).view(b, -1, 1) # bx4x1
        gate2_out = F.softmax(self.gate_expert_5(all_feature), dim=-1).view(b, -1, 1) # bx4x1
        gate3_out = F.softmax(self.gate_expert_like(all_feature), dim=-1).view(b, -1, 1) # bx4x1
        gate3_out = F.softmax(self.gate_expert_share(all_feature), dim=-1).view(b, -1, 1) # bx8x1

        # expert_fea = torch.cat([expert1_out, expert2_out, expert3_out, expert4_out, expert5_out, expert6_out], dim=1) # bx6x32
        
        tower4_fea = torch.sum(gate1_out * fea_score_4, dim=1)
        tower5_fea = torch.sum(gate2_out * fea_score_5, dim=1)
        towerlike_fea = torch.sum(gate3_out * fea_score_like, dim=1)
        towershare_fea = torch.sum(gate3_out * fea_score_5, dim=1)

        score_4 = self.tower_score4(tower4_fea)
        score_5 = self.tower_score5(tower5_fea)
        like = self.tower_like(towerlike_fea)
        


        return (score_4, score_5, like)
    
    def forward(self, data):
        
        logit = self.get_logit(data)


        label, label4, label5 = data['label'], data['label_4'], data['label_5']

        # logit = torch.sigmoid(user_weight + item_weight + age_weight + occ_weight + kind_weight + gender_weight)
        score_4, score_5, like = logit

        score_4 = torch.sigmoid(score_4)
        score_5 = torch.sigmoid(score_5)
        like = torch.sigmoid(like)


        loss_4 = self.binary_cross_entropy_loss(score_4, label4)
        loss_5 = self.binary_cross_entropy_loss(score_5, label5)
        loss_like = self.binary_cross_entropy_loss(like, label)

        loss = torch.exp(-self.loss_weight[1]) * loss_4 + \
                torch.exp(-self.loss_weight[2]) * loss_5 + torch.exp(-self.loss_weight[3]) * loss_like + \
                (torch.sum(self.loss_weight))
        
        auc_4 = self.auc(score_4, label4)
        auc_5 = self.auc(score_5, label5)
        auc = self.auc(like, label)
        return {'loss': loss, 'auc': auc, 'auc4': auc_4, 'auc5': auc_5}
    
    def eval_(self, data):
        logit = self.get_logit(data)
        score_4, score_5, like = logit
        score_4 = torch.sigmoid(score_4)
        score_5 = torch.sigmoid(score_5)
        like = torch.sigmoid(like)
        
        label = data['label']
        logit = (score_4 ** 0.9) * (like ** 0.95) * (score_5) 
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

    model = MMoESortModel(user_num, item_num, user_age_num=7 + 1, user_gender_num=2 + 1, user_occupation_num=21 + 1, item_kind_num=18 + 1)
    dataloader = get_sort_dataloader(batch_size=batch_size, num_workers=0)
    eval_dataloader = get_sort_dataloader(type='test', num_workers=0)
    trainer = Trainer(model, 'Deep', './log')
    trainer.set_config(epoch_num, lr, 'adam', dataloader, eval_dataloader, 'cosin', lr_min, save_epoch=epoch_num, eval_epoch=1)
    trainer.train()
    trainer.eval()


    
  