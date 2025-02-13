

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
    def __init__(self, 
                 user_num, 
                 item_num, 
                 dim: int=16, 
                 device: str='cpu', 
                 data_type: str='random', 
                 neg_sample_num: int=1,
                 loss_fn_type: str='SS'):
        super().__init__()

        assert loss_fn_type in ['SS', 'NCE', 'BPR'], 'invalid loss function type!'

        self.user_id_embedding = nn.Embedding(user_num, dim)
        self.item_id_embedding = nn.Embedding(item_num, dim)


        # self.user_tower = nn.Sequential(
        #     nn.Linear(dim, dim * 4),
        #     nn.Linear(dim * 4, dim),
        #     # nn.Linear(dim * 4, dim)
        # )

        # self.item_tower = nn.Sequential(
        #     nn.Linear(dim, dim * 4),
        #     nn.Linear(dim * 4, dim),
        #     # nn.Linear(dim * 4, dim)
        # )
        # nn.init.xavier_uniform_(self.user_id_embedding.weight)
        # nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.device = torch.device(device)
        self.data_type = data_type
        self.neg_sample_num = neg_sample_num
        self.loss_fn_type = loss_fn_type

    # Sampled Softmax Loss
    def InBatchLoss(self, user_id, item_id):
        user_emb = self.user_id_embedding(user_id).squeeze(1)
        item_emb = self.item_id_embedding(item_id).squeeze(1)

        if hasattr(self, 'user_tower') and hasattr(self, 'item_tower'):
            user_emb = self.user_tower(user_emb)
            item_emb = self.item_tower(item_emb)
        

        dot_product_all = torch.matmul(user_emb, item_emb.T)

        b, _ = dot_product_all.shape

        # loss = dot_product_all.clone()
        help_mat = torch.eye(b).to(self.device)
        # help_mat = help_mat

        # for i in range(b):
        #     # i, i位置的是正样本
        #     dot_product_all[i, i] = dot_product_all[i, i].clone() * -1
        dot_product_all = torch.exp(dot_product_all)
        dot_product_all_sum = torch.sum(dot_product_all, dim=-1, keepdim=True)
        dot_product_all = dot_product_all / dot_product_all_sum
        dot_product_all = torch.log(dot_product_all) * help_mat

        loss = -dot_product_all.sum()# * help_mat

        
        # loss = torch.log(1 + torch.exp(loss))
        # loss = torch.log(loss)
        # loss = loss.sum(dim=-1).sum(dim=-1)
        # loss = loss.sum()
        return loss / b
    
    # Sampled Softmax Loss
    def RandomNegLoss(self, user_id, item_id, neg_id):
        # user_id: bx1   item_id: bx1   neg_id: bx10
        # user_emb: bx1xdim   item_emb: bx1xdim  neg_emb: bx10xdim
        
        user_emb = self.user_id_embedding(user_id)  # bx1xdim
        item_emb = self.item_id_embedding(item_id)
        neg_emb = self.item_id_embedding(neg_id)

        if hasattr(self, 'user_tower') and hasattr(self, 'item_tower'):
            user_emb = self.user_tower(user_emb)
            item_emb = self.item_tower(item_emb)
            neg_emb = self.item_tower(neg_emb)

        cat_item_emb = torch.concat([item_emb, neg_emb], dim=1) # bx11xdim
        
        # 计算向量长度
        # cat_item_emb_len = torch.sqrt(torch.sum((cat_item_emb * cat_item_emb), dim=-1)) # bx11
        # user_emb_len = torch.sqrt(torch.sum((user_emb * user_emb), dim=-1)) # bx1
        # user_item_len_dot = cat_item_emb_len * user_emb_len


        dot_product = torch.matmul(user_emb, cat_item_emb.transpose(1, 2)).squeeze(1)  # bx11
        # dot_product = dot_product / (user_item_len_dot + 1e-6) # 计算相似度


        label = torch.zeros_like(dot_product)
        label[:, 0] = 1
        
        # sample sigmoid loss (SS Loss)
        if self.loss_fn_type == 'SS':
            b, _ = dot_product.shape
            dot_product = torch.exp(dot_product)
            dot_sum = torch.sum(dot_product, dim=-1, keepdim=True)
            dot_product = dot_product / dot_sum
            loss = -torch.log(dot_product[:, 0]).sum()
            loss = loss / b
        elif self.loss_fn_type == 'NCE':
            # NCE loss
            b, n = dot_product.shape
            loss = torch.sigmoid(dot_product)
            loss = -(label * torch.log(loss + 1e-6) + (1 - label) * torch.log(1 - loss + 1e-6))
            loss = loss.sum()
            loss = loss / (b * n)
        elif self.loss_fn_type == 'BPR':
            # BPR Loss
            b, _ = dot_product.shape
            pos_weight = dot_product[:, 0].view(-1, 1)
            loss = dot_product - pos_weight
            loss = loss[:, 1:]
            loss = torch.log(1 + torch.exp(loss)).sum()
            loss = loss / b

        return loss

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
    epoch_num = 15
    batch_size = 256
    lr = 1e-3
    lr_min = 1e-4
    device = 'cpu'
    data_type = 'random'
    # data_type = 'in_batch'
    neg_sample_num = 20

    model = DSSMModel(user_num, item_num, dim=16, device=device, data_type=data_type, neg_sample_num=neg_sample_num)
    if device != 'cpu':
        model.cuda(int(device[-1]))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    workdir = '/Users/zhanghaoyang/Desktop/Movie_Recsys/recall/DSSM'

    dataset = DSSMDataLoader(
        '/Users/zhanghaoyang/Desktop/Movie_Recsys/cache/train_readlist.pkl',
        '/Users/zhanghaoyang/Desktop/Movie_Recsys/cache/movie_info.pkl',
        '/Users/zhanghaoyang/Desktop/Movie_Recsys/cache/user_info.pkl',
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

        if epoch != 0 and epoch % epoch_num == 0:
            torch.save(model, './DSSM_epoch_%d.pth' % epoch)
            model.save_emb('.', 'epoch_%d' % epoch)
    torch.save(model, './DSSM_final.pth')
    # model.save_user_item_embedding_weights(workdir)
    model.save_emb('.', 'final')
