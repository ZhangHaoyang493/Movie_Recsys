import os
import torch
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
import pickle
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score

import sys
sys.path.append('/Users/zhanghaoyang04/Desktop/Movie_Recsys/sort')
from sortDataLoader import get_sort_dataloader


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
                user_age_num, 
                user_occupation_num, 
                item_kind_num):
        super().__init__()

        self.user_id_para = nn.Embedding(user_num, 1)
        self.item_id_para = nn.Embedding(item_num, 1)
        self.age_para = nn.Embedding(user_age_num, 1)
        self.occupation_para = nn.Embedding(user_occupation_num, 1)
        self.kind_para = nn.Embedding(item_kind_num, 1)

    def binary_cross_entropy_loss(self, logit, label):
        return (-(label * torch.log(logit + 1e-6) + (1 - label) * torch.log(1 - logit + 1e-6))).sum() / label.shape[0]

    def forward(self, data):
        user_weight = self.user_id_para(data['userid'])
        item_weight = self.item_id_para(data['itemid'])
        age_weight = self.age_para(data['user_age'])
        occ_weight = self.occupation_para(data['user_occupation'])
        kind_weight = self.kind_para(data['item_kind']) # bx10x1
        kind_weight[data['item_kind'] == 0] = 0
        kind_weight = torch.sum(kind_weight, dim=1).view(-1, 1, 1)
        label = data['label']

        logit = torch.sigmoid(user_weight + item_weight + age_weight + occ_weight + kind_weight)

        return self.binary_cross_entropy_loss(logit, label), self.auc(logit.view(-1, 1), label)
    
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

    model = LRSortModel(user_num, item_num, user_age_num=7 + 1, user_occupation_num=21 + 1, item_kind_num=18 + 1)
    if device != 'cpu':
        model.cuda(int(device[-1]))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    

    dataloader = get_sort_dataloader(
        batch_size=batch_size,
        num_workers=0
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=(len(dataloader)) * epoch_num, 
        eta_min=lr_min
    )

    
    for epoch in range(epoch_num):
        model.train()
        loss_epoch = 0.0
        tqdm_bar = tqdm(dataloader, ncols=100)
        data_index = 1
        for data in tqdm_bar:
            optimizer.zero_grad()
            loss, auc = model(data)
            loss.backward()
            optimizer.step()
            loss_epoch += loss
            scheduler.step()
            tqdm_bar.set_postfix_str(
                'lr: %.6f | loss: %.3f | auc: %.3f' % (optimizer.state_dict()['param_groups'][0]['lr'], loss_epoch / data_index, auc)
            )
            data_index += 1
        print('Epoch: %d, Loss: %.3f' % (epoch, loss_epoch / len(dataloader)))

        if epoch != 0 and epoch % epoch_num == 0:
            torch.save(model, './LR_epoch_%d.pth' % epoch)
    torch.save(model, './LR_final.pth')



        
        