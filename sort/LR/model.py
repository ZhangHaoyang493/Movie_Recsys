import sys
sys.path.append('/data2/zhy/Movie_Recsys/')

from BaseModel.base_model import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import faiss
import numpy as np
from math import log2
import random

from DataReader.data_reader import DataReader
from tqdm import tqdm
from model_utils.lr_schedule import CosinDecayLR

class LR(BaseModel):
    def __init__(self, config_path, dataloaders={}, hparams={}):
        super(LR, self).__init__(config_path)
        
        self.save_hyperparameters(hparams)
        self.hparams_ = hparams

        # 定义Deep模型的网络结构
        self.score_fc = torch.sum
        
        self.movies_dataloader = dataloaders.get('movies_dataloader', None)
        self.val_dataloader_ = dataloaders.get('val_dataloader', None)


    def bceLoss(self, preds, labels):
        return F.binary_cross_entropy(preds.view(-1), labels.view(-1), reduction='mean')


    def forward(self, x):
        inp_feature = self.get_inp_embedding(x)  # 获取输入特征向量
        scores = F.sigmoid(self.score_fc(inp_feature, dim=1))  # 通过全连接层计算得分
        return scores  # 返回预测分数


    # def get_user_embedding(self, batch):
    #     user_embeddings = []
    #     for feature_name in self.user_feature_names:
    #         emb = self.get_features_embedding(feature_name, batch[feature_name])
    #         if feature_name in self.sparse_feature_names:
    #             user_embeddings.append(emb)
    #         elif feature_name in self.dense_feature_names:
    #             user_embeddings.append(emb)
    #         elif feature_name in self.array_feature_names:
    #             mask = batch.get(f"{feature_name}_mask", None)  # bxarr_lenxdim
    #             if mask is not None:
    #                 emb = emb * mask.unsqueeze(-1)  # 应用mask
    #                 emb = emb.sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)  # 避免除以零，mean pooling
    #             user_embeddings.append(emb)
    #     user_feature_vector = torch.cat(user_embeddings, dim=1)  # 在特征维度上拼接
    #     return user_feature_vector
    
    # def get_item_embedding(self, batch):
    #     item_embeddings = []
    #     for feature_name in self.item_feature_names:
    #         emb = self.get_features_embedding(feature_name, batch[feature_name])
    #         if feature_name in self.sparse_feature_names:
    #             item_embeddings.append(emb)
    #         elif feature_name in self.dense_feature_names:
    #             item_embeddings.append(emb)
    #         elif feature_name in self.array_feature_names:
    #             mask = batch.get(f"{feature_name}_mask", None)  # bxarr_lenxdim
    #             if mask is not None:
    #                 emb = emb * mask.unsqueeze(-1)  # 应用mask
    #                 emb = emb.sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)  # 避免除以零，mean pooling
    #             item_embeddings.append(emb)
    #     item_feature_vector = torch.cat(item_embeddings, dim=1)  # 在特征维度上拼接
    #     return item_feature_vector
    
    def get_inp_embedding(self, batch):
        features, _, _ = self.get_embedding_from_set(batch, self.user_feature_names | self.item_feature_names)
        return features
    
    def training_step(self, batch, batch_idx):
        scores = self.forward(batch)
        labels = batch['label'][:, 1]  # 获取是否喜欢的标签
        loss = self.bceLoss(scores, labels)  # 计算二元交叉熵损失
        self.log('train_loss', loss, prog_bar=True)
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
    
    @torch.no_grad()
    def inference(self, batch):
        inp_feature = self.get_inp_embedding(batch)  # 获取输入特征向量
        scores = F.sigmoid(self.score_fc(inp_feature, dim=1))  # 通过全连接层计算得分
        return scores  # 返回预测分数


    @torch.no_grad()
    def on_train_epoch_end(self):
        if self.current_epoch % 1 == 0:
            self.eval()