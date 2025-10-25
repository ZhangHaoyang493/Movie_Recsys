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

class Deep(BaseModel):
    def __init__(self, config_path, dataloaders={}):
        super(Deep, self).__init__(config_path)
        
        # 定义Deep模型的网络结构
        self.score_fc = nn.Sequential(
            nn.Linear(self.user_input_dim + self.item_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.movies_dataloader = dataloaders.get('movies_dataloader', None)
        self.val_dataloader_ = dataloaders.get('val_dataloader', None)

    def forward(self, x):
        inp_feature = self.get_inp_embedding(x)  # 获取输入特征向量
        scores = self.score_fc(inp_feature)  # 通过全连接层计算得分

        


        return scores.squeeze(-1)  # 返回得分，去掉最后一个维度

    def get_inp_embedding(self, batch):
        embeddings = []
        all_slots = self.user_slots + self.item_slots
        for slot_id in all_slots:
            emb = self.get_slots_embedding(slot_id, batch[slot_id])
            if slot_id in self.sparse_slots:
                embeddings.append(emb)
            elif slot_id in self.dense_slots:
                embeddings.append(emb)
            elif slot_id in self.array_slots:
                mask = batch.get(f"{slot_id}_mask", None)  # bxarr_lenxdim
                if mask is not None:
                    emb = emb * mask.unsqueeze(-1)  # 应用mask
                    emb = emb.sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)  # 避免除以零，mean pooling
                embeddings.append(emb)
        feature_vector = torch.cat(embeddings, dim=1)  # 在特征维度上拼接
        return feature_vector