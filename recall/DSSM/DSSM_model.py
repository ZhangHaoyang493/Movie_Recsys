import sys
sys.path.append('/Users/zhanghaoyang/Desktop/Movie_Recsys/')

from BaseModel.base_model import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F

# BaseModel继承于LightningModule
class DSSM(BaseModel):
    def __init__(self, config_path):
        super(DSSM, self).__init__(config_path)
        
        # 定义DSSM的网络结构
        # 假设我们有两个全连接层，分别用于用户和物品的特征处理
        self.user_fc = nn.Sequential(
            nn.Linear(self.user_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 16)
        )
        
        self.item_fc = nn.Sequential(
            nn.Linear(self.item_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 16)
        )

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

    def triplet_loss(self, user_emb, pos_item_emb, neg_item_emb, margin=1.0):
        pos_scores = torch.sum(user_emb * pos_item_emb, dim=1)
        neg_scores = torch.sum(user_emb * neg_item_emb, dim=1)
        losses = F.relu(margin - pos_scores + neg_scores)
        return losses.mean()

    def training_step(self, batch, batch_idx):
        user_emb, item_emb, neg_item_emb = self.forward(batch)
        # 计算三元组损失
        loss = self.triplet_loss(user_emb, item_emb, neg_item_emb)
        
        self.log('train_loss', loss)
        return loss


    def validation_step(self, batch, batch_idx):
        user_emb, item_emb, neg_item_emb = self.forward(batch)

        # 计算三元组损失
        loss = self.triplet_loss(user_emb, item_emb, neg_item_emb)

        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

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