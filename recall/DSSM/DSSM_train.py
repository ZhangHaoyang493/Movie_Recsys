

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
    def __init__(self, user_num, item_num, dim: int=16, device: str='cpu', data_type: str='random', neg_sample_num: int=1):
        super().__init__()
        self.user_id_embedding = nn.Embedding(user_num, dim)
        self.item_id_embedding = nn.Embedding(item_num, dim)

        nn.init.xavier_uniform_(self.user_id_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.device = torch.device(device)
        self.data_type = data_type
        self.neg_sample_num = neg_sample_num

    def InBatchLoss(self, user_id, item_id):
        user_emb = self.user_id_embedding(user_id).squeeze(1)
        item_emb = self.item_id_embedding(item_id).squeeze(1)

        dot_product_all = torch.matmul(user_emb, item_emb.T)

        b, _ = dot_product_all.shape

        # loss = dot_product_all.clone()
        help_mat = (torch.ones_like(dot_product_all) - 2 * torch.eye(b).to(self.device))
        # help_mat = help_mat

        # for i in range(b):
        #     # i, i位置的是正样本
        #     dot_product_all[i, i] = dot_product_all[i, i].clone() * -1
        loss = dot_product_all * help_mat
        
        loss = torch.log(1 + torch.exp(loss))
        # loss = torch.log(loss)
        # loss = loss.sum(dim=-1).sum(dim=-1)
        loss = loss.sum()
        return loss / b
    
    def RandomNegLoss(self, user_id, item_id, neg_id):
        # user_id: bx1   item_id: bx1   neg_id: bx10
        # user_emb: bx1xdim   item_emb: bx1xdim  neg_emb: bx10xdim
        b, _ = user_id.shape
        user_emb = self.user_id_embedding(user_id)  # bx1xdim
        item_emb = self.item_id_embedding(item_id)
        neg_emb = self.item_id_embedding(neg_id)

        cat_item_emb = torch.concat([item_emb, neg_emb], dim=1) # bx11xdim
        
        # 计算向量长度
        cat_item_emb_len = torch.sqrt(torch.sum((cat_item_emb * cat_item_emb), dim=-1)) # bx11
        user_emb_len = torch.sqrt(torch.sum((user_emb * user_emb), dim=-1)) # bx1
        user_item_len_dot = cat_item_emb_len * user_emb_len


        dot_product = torch.matmul(user_emb, cat_item_emb.transpose(1, 2)).squeeze(1)  # bx11
        dot_product = dot_product / (user_item_len_dot + 1e-6) # 计算相似度
 
        # 第一列是正样本
        # dot_product[:, 0] = dot_product[:, 0] * -1
        # dot_product[:, 1:] = dot_product[:, 1:] / self.neg_sample_num

        label = torch.zeros_like(dot_product)
        label[:, 0] = 1


        # loss = torch.log(1 + torch.exp(dot_product))
        # loss = loss.sum()
        dot_product = (dot_product + 1) / 2
        loss = label * torch.log(dot_product + 1e-6) + (1 - label) * torch.log(1 - dot_product + 1e-6)
        loss = -loss
        loss = loss.sum()


        return loss / b

    def forward(self, data):
        user_id = data['userid'].to(self.device)
        item_id = data['itemid'].to(self.device)
        user_fea = data['user_feature'].to(self.device)
        item_fea = data['item_feature'].to(self.device)
        if self.data_type == 'random':
            neg_id = data['neg_sample'].to(self.device)
        
        if self.data_type == 'random':
            return self.RandomNegLoss(user_id, item_id, neg_id)
        elif self.data_type == 'in_batch':
            loss = self.InBatchLoss(user_id, item_id)
        
        return loss
    
    def save_emb(self, save_path, postfix=''):
        item_emb = self.item_id_embedding.weight.detach().cpu().numpy()
        item_emb = {i: item_emb[i].tolist() for i in range(item_emb.shape[0])}
        pickle.dump(item_emb, open(os.path.join(save_path, 'item_emb_%s.pkl' % postfix), 'wb'))
        user_emb = self.user_id_embedding.weight.detach().cpu().numpy()
        user_emb = {i: user_emb[i].tolist() for i in range(user_emb.shape[0])}
        pickle.dump(user_emb, open(os.path.join(save_path, 'user_emb_%s.pkl' % postfix), 'wb'))

    def get_user_emb(self, user_id):
        return self.user_id_embedding(user_id).detach().cpu().numpy()

if __name__ == '__main__':
    item_num = 3952 + 1
    user_num = 6040 + 1
    epoch_num = 10
    batch_size = 256
    lr = 1e-2
    lr_min = 1e-4
    device = 'cpu'
    data_type = 'random'
    neg_sample_num = 20

    model = DSSMModel(user_num, item_num, dim=16, device=device, data_type=data_type, neg_sample_num=neg_sample_num)
    if device != 'cpu':
        model.cuda(int(device[-1]))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    workdir = '/Users/zhanghaoyang04/Desktop/Movie_Recsys/recall/DSSM'

    dataset = DSSMDataLoader(
        '/Users/zhanghaoyang04/Desktop/Movie_Recsys/cache/train_readlist.pkl',
        '/Users/zhanghaoyang04/Desktop/Movie_Recsys/cache/movie_info.pkl',
        '/Users/zhanghaoyang04/Desktop/Movie_Recsys/cache/user_info.pkl',
        data_type=data_type,
        neg_sample_num=neg_sample_num
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)


    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=(len(dataset) // batch_size) * epoch_num, 
        eta_min=lr_min
    )

    
    for epoch in range(epoch_num):
        model.train()
        loss_epoch = 0.0
        tqdm_bar = tqdm(dataloader, ncols=100)
        data_index = 1
        for data in tqdm_bar:
            optimizer.zero_grad()
            loss = model(data)
            loss.backward()
            optimizer.step()
            loss_epoch += loss
            scheduler.step()
            tqdm_bar.set_postfix_str(
                'lr: %.6f | loss: %.3f' % (optimizer.state_dict()['param_groups'][0]['lr'], loss_epoch / data_index)
            )
            data_index += 1
        print('Epoch: %d, Loss: %.3f' % (epoch, loss_epoch / len(dataloader)))

        if epoch != 0 and epoch % 5 == 0:
            torch.save(model, './DSSM_epoch_%d.pth' % epoch)
            model.save_emb('.', 'epoch_%d' % epoch)
    torch.save(model, './DSSM_final.pth')
    # model.save_user_item_embedding_weights(workdir)
    model.save_emb('.', 'final')
