import sys
sys.path.append('/data2/zhy/Movie_Recsys/')

from BaseModel.base_model import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
import numpy as np

from DataReader.data_reader import DataReader
from tqdm import tqdm
from model_utils.lr_schedule import CosinDecayLR

# BaseModel继承于LightningModule
class DSSM(BaseModel):
    def __init__(self, config_path, dataloaders={}, hparams={}):
        super(DSSM, self).__init__(config_path)

        # 保存超参数
        self.save_hyperparameters(hparams)
        self.hparams_ = hparams
        
        # 定义DSSM的网络结构
        # 假设我们有两个全连接层，分别用于用户和物品的特征处理
        self.user_fc = nn.Sequential(
            nn.Linear(self.user_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16)
        )
        
        self.item_fc = nn.Sequential(
            nn.Linear(self.item_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16)
        )
        
        self.movies_dataloader = dataloaders.get('movies_dataloader', None)
        self.val_dataloader_ = dataloaders.get('val_dataloader', None)

       

    def forward(self, x):
        user_vector = self.get_user_embedding(x)  # 获取用户特征向量
        item_vector = self.get_item_embedding(x)  # 获取物品特征向量
        
        user_emb = self.user_fc(user_vector)  # 用户特征通过全连接层  bx16
        item_emb = self.item_fc(item_vector)  # 物品特征通过全连接层  bx16

        # 构造一个负采样的item_emb，第i个batch的负样本是随机选取的其他item_emb
        batch_size = item_emb.size(0)
        neg_indices = torch.randperm(batch_size)
        neg_item_emb = item_emb[neg_indices]

        # 归一化
        user_emb = F.normalize(user_emb, p=2, dim=1)
        item_emb = F.normalize(item_emb, p=2, dim=1)
        neg_item_emb = F.normalize(neg_item_emb, p=2, dim=1)

        return user_emb, item_emb, neg_item_emb

    def triplet_loss(self, user_emb, pos_item_emb, neg_item_emb, margin=1.0, mask=None):
        pos_scores = torch.sum(user_emb * pos_item_emb, dim=1)
        neg_scores = torch.sum(user_emb * neg_item_emb, dim=1)
        losses = F.relu(margin - pos_scores + neg_scores)
        if mask is not None:
            losses = losses * mask
        return losses.mean()

    def training_step(self, batch, batch_idx):
        user_emb, item_emb, neg_item_emb = self.forward(batch)
        # 计算三元组损失, recall阶段只使用正样本，负样本是batch内随机采样的其他item
        mask = batch['label'][:, 1] # 获取mask
        loss = self.triplet_loss(user_emb, item_emb, neg_item_emb, mask=mask)
        
        self.log('train_loss', loss)
        # 记录当前学习率
        self.log('lr', self.optimizers().param_groups[0]['lr'])
        return loss


    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams_['lr'], betas=(0.9, 0.999))
        lr_scheduler = CosinDecayLR(optimizer, lrs=[self.hparams_['lr'], self.hparams_['min_lr']], milestones=self.hparams_['lr_milestones'])
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'interval': 'step',  # 每个训练步骤调用一次
                'frequency': 1
            }

        }

    def get_user_embedding(self, batch):
        user_embeddings = []
        for slot_id in self.user_slots:
            emb = self.get_slots_embedding(slot_id, batch[slot_id])
            if slot_id in self.sparse_slots:
                user_embeddings.append(emb)
            elif slot_id in self.dense_slots:
                user_embeddings.append(emb)
            elif slot_id in self.array_slots:
                mask = batch.get(f"{slot_id}_mask", None)  # bxarr_lenxdim
                if mask is not None:
                    emb = emb * mask.unsqueeze(-1)  # 应用mask
                    emb = emb.sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)  # 避免除以零，mean pooling
                user_embeddings.append(emb)
        user_feature_vector = torch.cat(user_embeddings, dim=1)  # 在特征维度上拼接
        return user_feature_vector
    
    def get_item_embedding(self, batch):
        item_embeddings = []
        for slot_id in self.item_slots:
            emb = self.get_slots_embedding(slot_id, batch[slot_id])
            if slot_id in self.sparse_slots:
                item_embeddings.append(emb)
            elif slot_id in self.dense_slots:
                item_embeddings.append(emb)
            elif slot_id in self.array_slots:
                mask = batch.get(f"{slot_id}_mask", None)  # bxarr_lenxdim
                if mask is not None:
                    emb = emb * mask.unsqueeze(-1)  # 应用mask
                    emb = emb.sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)  # 避免除以零，mean pooling
                item_embeddings.append(emb)
        item_feature_vector = torch.cat(item_embeddings, dim=1)  # 在特征维度上拼接
        return item_feature_vector
    
    @torch.no_grad()
    def hit_rate(self, k=10):
        hits_num = 0
        all_nums = 0
        for batch in tqdm(self.val_dataloader_, desc="Evaluating Hit Rate", ncols=100):
            labels = batch['label'][:, 1]  # 计算hit rate时只考虑正样本，这里获取到labels的值，后续用于过滤负样本

            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
            user_vector = self.get_user_embedding(batch)  # 获取用户特征向量
            user_emb = self.user_fc(user_vector)  # 用户特征通过全连接层  bx16
            user_emb = F.normalize(user_emb, p=2, dim=1)
            user_emb = user_emb.cpu().numpy()

            D, I = self.index.search(user_emb, k)  # 搜索top-k的物品

            # 将索引映射到item id
            I = [[self.idx_item_emb_dic[idx] for idx in user_indices] for user_indices in I]

            # 计算命中率
            targets = batch[2].cpu().numpy()  # 假设item_id是物品的真实ID
            hits = 0
            for i in range(len(targets)):
                if labels[i] == 1:  # 只计算正样本的命中率
                    if targets[i] in I[i]:
                        hits_num += 1
                    all_nums += 1
        hit_rate = (hits_num / all_nums) if all_nums > 0 else 0
        self.log('Hit_Rate_50', hit_rate)
        print(f"Hit Rate@{k}: {hit_rate}")

    @torch.no_grad()
    def on_train_epoch_end(self):
        if self.current_epoch % 1 == 0:
            all_item_embeddings = []
            self.idx_item_emb_dic = {}
            idx = 0
            for batch in tqdm(self.movies_dataloader, desc="Building Item Embeddings", ncols=100):
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                item_vector = self.get_item_embedding(batch)  # 获取物品特征向量
                item_emb = self.item_fc(item_vector)  # 物品特征通过全连接层  bx16
                item_emb = F.normalize(item_emb, p=2, dim=1)
                item_emb = item_emb.cpu().numpy()
                all_item_embeddings.append(item_emb)
                for i, item_id in enumerate(batch[2].cpu().numpy()): # 2是item id的slot_id
                    self.idx_item_emb_dic[idx] = item_id  # 记录item_id对应的embedding在all_item_embeddings中的索引
                    idx += 1

            self.all_item_embeddings = np.concatenate(all_item_embeddings, axis=0)
            self.index = faiss.IndexFlatIP(self.all_item_embeddings.shape[1])  # 内积相似度
            self.index.add(self.all_item_embeddings)  # 添加所有物品向量到索引

            # 计算验证集的命中率
            self.hit_rate(k=50)