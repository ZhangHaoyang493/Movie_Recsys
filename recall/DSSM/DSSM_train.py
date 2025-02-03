

import os
import torch
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
import pickle
from tqdm import tqdm
import numpy as np
import faiss

from dataloader import DSSMDataLoader

class DSSMModel(nn.Module):
    def __init__(self, user_num, item_num, dim: int=16):
        super().__init__()
        self.user_id_embedding = nn.Embedding(user_num, dim)
        self.item_id_embedding = nn.Embedding(item_num, dim)

    def InBatchLoss(self, user_emb, item_emb):
        dot_product_all = torch.matmul(user_emb, item_emb.T)

        b, _ = dot_product_all.shape

        # loss = dot_product_all.clone()
        help_mat = torch.ones_like(dot_product_all) - 2 * torch.eye(b)

        # for i in range(b):
        #     # i, i位置的是正样本
        #     dot_product_all[i, i] = dot_product_all[i, i].clone() * -1
        loss = dot_product_all * help_mat
        
        loss = torch.log(1 + torch.exp(loss))
        # loss = torch.log(loss)
        # loss = loss.sum(dim=-1).sum(dim=-1)
        loss = loss.sum()
        return loss / b

    def forward(self, data):
        user_id = data['userid']
        item_id = data['itemid']
        user_fea = data['user_feature']
        item_fea = data['item_feature']

        user_emb = self.user_id_embedding(user_id).squeeze(1)
        item_emb = self.item_id_embedding(item_id).squeeze(1)

        loss = self.InBatchLoss(user_emb, item_emb)
        
        return loss
    
    def save_emb(self, save_path):
        item_emb = self.item_id_embedding.weight.detach().cpu().numpy()
        item_emb = {i: item_emb[i].tolist() for i in range(item_emb.shape[0])}
        pickle.dump(item_emb, open(os.path.join(save_path, 'item_emb.pkl'), 'wb'))
        user_emb = self.user_id_embedding.weight.detach().cpu().numpy()
        user_emb = {i: user_emb[i].tolist() for i in range(user_emb.shape[0])}
        pickle.dump(user_emb, open(os.path.join(save_path, 'user_emb.pkl'), 'wb'))

    def get_user_emb(self, user_id):
        return self.user_id_embedding(user_id).detach().cpu().numpy()

if __name__ == '__main__':
    item_num = 3952 + 1
    user_num = 6040 + 1
    epoch_num = 10
    batch_size = 128
    lr = 1e-3
    lr_min = 1e-4

    model = DSSMModel(user_num, item_num, dim=16)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    workdir = '/Users/zhanghaoyang/Desktop/Movie_Recsys/recall/DSSM'

    dataset = DSSMDataLoader(
        '/Users/zhanghaoyang/Desktop/Movie_Recsys/cache/train_readlist.pkl',
        '/Users/zhanghaoyang/Desktop/Movie_Recsys/cache/movie_info.pkl',
        '/Users/zhanghaoyang/Desktop/Movie_Recsys/cache/user_info.pkl'
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)


    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(len(dataset) // batch_size) * epoch_num, eta_min=lr_min)

    
    for epoch in range(epoch_num):
        model.train()
        loss_epoch = 0.0
        tqdm_bar = tqdm(dataloader, ncols=100)
        for data in tqdm_bar:
            optimizer.zero_grad()
            loss = model(data)
            loss.backward()
            optimizer.step()
            loss_epoch += loss
            scheduler.step()
            tqdm_bar.set_postfix_str('lr: %.6f' % optimizer.state_dict()['param_groups'][0]['lr'])
        print('Epoch: %d, Loss: %.3f' % (epoch, loss_epoch / len(dataloader)))
    torch.save(model, './DSSM.pth')
    # model.save_user_item_embedding_weights(workdir)
    model.save_emb('.')
