import os
import torch
from torch import nn
from sklearn.metrics import roc_auc_score

import sys
sys.path.append('/Users/zhanghaoyang/Desktop/Movie_Recsys/sort')
from sortDataLoader import get_sort_dataloader
from trainer import Trainer

# return {
#             'userid': torch.tensor([int(userid)]),
#             'itemid': torch.tensor([int(itemid)]),
#             'score': torch.tensor([float(score)]),
#             'user_age': torch.tensor([user_age]),
#             'user_occupation': torch.tensor([user_occupation]),
#             'item_kind': torch.tensor(item_kind),
#             'label': torch.tensor([1 if score >= 4 else 0])
#         }

class LRSortModel(nn.Module):
    def __init__(self, 
                user_num,
                item_num,
                user_gender_num,
                user_age_num, 
                user_occupation_num, 
                item_kind_num):
        super().__init__()

        self.user_id_para = nn.Embedding(user_num, 1)
        self.item_id_para = nn.Embedding(item_num, 1)
        self.age_para = nn.Embedding(user_age_num, 1)
        self.gender_para = nn.Embedding(user_gender_num, 1)
        self.occupation_para = nn.Embedding(user_occupation_num, 1)
        self.kind_para = nn.Embedding(item_kind_num, 1)


    def binary_cross_entropy_loss(self, logit, label):
        return (-(label * torch.log(logit + 1e-6) + (1 - label) * torch.log(1 - logit + 1e-6))).sum() / label.shape[0]

    def train_step(self, data):
        user_weight = self.user_id_para(data['userid'])
        item_weight = self.item_id_para(data['itemid'])
        age_weight = self.age_para(data['user_age'])
        gender_weight = self.gender_para(data['gender'])
        occ_weight = self.occupation_para(data['user_occupation'])
        kind_weight = self.kind_para(data['item_kind']) # bx10x1
        kind_weight[data['item_kind'] == 0] = 0
        kind_weight = torch.sum(kind_weight, dim=1).view(-1, 1, 1)

        return user_weight + item_weight + age_weight + occ_weight + kind_weight + gender_weight

    def forward(self, data):
        
        label = data['label']

        logit = torch.sigmoid(self.train_step(data))


        loss = self.binary_cross_entropy_loss(logit, label)
        auc = self.auc(logit.view(-1, 1), label)
        return {'loss': loss, 'auc': auc}
    
    def eval_(self, data):
        logit = torch.sigmoid(self.train_step(data))
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

    model = LRSortModel(user_num, item_num, user_age_num=7 + 1, user_gender_num=2 + 1, user_occupation_num=21 + 1, item_kind_num=18 + 1)
    dataloader = get_sort_dataloader(batch_size=batch_size, num_workers=0)
    eval_dataloader = get_sort_dataloader(type='test', num_workers=0)
    trainer = Trainer(model, 'LR', './log')
    trainer.set_config(epoch_num, lr, 'adam', dataloader, eval_dataloader, 'cosin', lr_min, save_epoch=epoch_num, eval_epoch=1)
    trainer.train()
    trainer.eval()


    
  