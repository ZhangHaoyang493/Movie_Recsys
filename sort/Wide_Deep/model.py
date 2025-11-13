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
from model_utils.utils import MLP
from sklearn.metrics import roc_auc_score

class WideDeepModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=[32, 32, 1]):
        super(WideDeepModel, self).__init__()
        dims = [input_dim] + hidden_dims
        
        self.wide_network = torch.sum
        self.deep_network = MLP(dims=dims)
        self.bias = nn.Parameter(torch.zeros(1))

    
    def forward(self, wide_x, deep_x):
        wide_out = self.wide_network(wide_x, dim=1, keepdim=True) + self.bias  # 线性部分
        deep_out = self.deep_network(deep_x)
        return F.sigmoid(wide_out + deep_out)
    

class WideDeep(BaseModel):
    def __init__(self, config_path, dataloaders={}, hparams={}):
        super(WideDeep, self).__init__(config_path)
        
        self.save_hyperparameters(hparams)
        self.hparams_ = hparams

        wide_and_deep_config = self.config['wide_and_deep_cfg']
        self.wide_feature_names = set(wide_and_deep_config['wide_feature_names'])


        # 定义Deep模型的网络结构，包括输入维度和隐藏层维度，这里减去wide特征的维度
        self.score_fc = WideDeepModel(input_dim=self.user_input_dim + self.item_input_dim - len(self.wide_feature_names), hidden_dims=[32, 32, 1])
        
        self.movies_dataloader = dataloaders.get('movies_dataloader', None)
        self.val_dataloader_ = dataloaders.get('val_dataloader', None)
        
        


    def bceLoss(self, preds, labels):
        return F.binary_cross_entropy(preds.view(-1), labels.view(-1), reduction='mean')


    def forward(self, x):
        wide_x, deep_x = self.get_inp_embedding(x)  # 获取输入特征向量
        return self.score_fc(wide_x, deep_x), wide_x  # 返回预测分数

    
    def get_inp_embedding(self, batch):
        features, dims, fnames = self.get_embedding_from_set(batch, self.user_feature_names | self.item_feature_names)
        wide_x = []
        deep_x = []
        start_idx = 0
        for dim, fname in zip(dims, fnames):
            end_idx = start_idx + dim
            if fname in self.wide_feature_names:
                wide_x.append(features[:, start_idx:start_idx+1])  # Bx1
                deep_x.append(features[:, start_idx+1:end_idx])  # Bxdim
            else:
                deep_x.append(features[:, start_idx:end_idx])  # Bxdim
            start_idx = end_idx
        wide_x = torch.cat(wide_x, dim=1)
        deep_x = torch.cat(deep_x, dim=1)

        return wide_x, deep_x
    
    def training_step(self, batch, batch_idx):
        scores, wide_x = self.forward(batch)
        labels = batch['label'][:, 1]  # 获取是否喜欢的标签
        loss = self.bceLoss(scores, labels)# + 1e-6 * torch.mean(torch.abs(wide_x))  # 计算二元交叉熵损失
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
        wide_x, deep_x = self.get_inp_embedding(batch)  # 获取输入特征向量
        return self.score_fc(wide_x, deep_x)  # 返回预测分数


    @torch.no_grad()
    def on_train_epoch_end(self):
        if self.current_epoch % 1 == 0:
            self.eval()