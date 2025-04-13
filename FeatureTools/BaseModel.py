import os
import torch
from torch import optim, nn, utils, Tensor
from torch.utils.data import Dataset, DataLoader
import yaml
from tqdm import tqdm
import numpy as np
import random
from BaseDataLoader import BaseDataloader


class BaseModel(nn.Module):
    def __init__(self, config_file):
        super().__init__()
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        self.fea_config_dict = {}
        self.user_fea_config_dict = {}
        self.item_fea_config_dict = {}
        
        self.user_fea_dim = 0
        self.item_fea_dim = 0

        for fea_name, fea_config in self.config['user_feature_config'].items():
            if 'DependEmbeddingTableName' not in fea_config:
                setattr(self, fea_name, nn.Embedding(int(fea_config['MaxIndex']), int(fea_config['Dim'])))
            self.user_fea_config_dict[fea_name] = fea_config
            self.fea_config_dict[fea_name] = fea_config
            if fea_config['AggreateMethod'] in ['avgpooling', 'none']:
                self.user_fea_dim += fea_config['Dim']
            elif fea_config['AggreateMethod'] in ['padding']:
                self.user_fea_dim += fea_config['PaddingDim'] * fea_config['Dim']
        
        for fea_name, fea_config in self.config['item_feature_config'].items():
            if 'DependEmbeddingTableName' not in fea_config:
                setattr(self, fea_name, nn.Embedding(int(fea_config['MaxIndex']), int(fea_config['Dim'])))
            self.item_fea_config_dict[fea_name] = fea_config
            self.fea_config_dict[fea_name] = fea_config

            if fea_config['AggreateMethod'] in ['avgpooling', 'none']:
                self.user_fea_dim += fea_config['Dim']
            elif fea_config['AggreateMethod'] in ['padding']:
                self.user_fea_dim += fea_config['PaddingDim'] * fea_config['Dim']
    
    def get_data_embedding(self, data):
        user_embedding_data = {}
        item_embedding_data = {}
        for key in data:
            if key != 'label':
                val, mask = data[key]
                if 'DependEmbeddingTableName' not in self.fea_config_dict[key]:
                    embedding_data = getattr(self, key)(val)
                else:
                    embedding_data = getattr(self, self.fea_config_dict[key]['DependEmbeddingTableName'])(val)
                if mask is not None:
                    embedding_data = embedding_data * mask.view(-1, 1)
                if self.fea_config_dict[key]['AggreateMethod'] == 'avgpooling':
                    embedding_data = torch.mean(embedding_data, dim=0, keepdim=True)
                if key in self.user_fea_config_dict:
                    user_embedding_data[key] = embedding_data
                else:
                    item_embedding_data[key] = embedding_data
            else:
                self.label = data[key]
        return user_embedding_data, item_embedding_data, self.label

if __name__ == '__main__':
    dataloader = BaseDataloader('./example.yaml')
    data = dataloader[1]
    model = BaseModel('./example.yaml')
    u, i, l = model.get_data_embedding(data)
    print(u, i, l)
    


