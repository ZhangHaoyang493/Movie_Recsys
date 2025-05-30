import os
import torch
from torch import optim, nn, utils, Tensor
from torch.utils.data import Dataset, DataLoader
import yaml
from tqdm import tqdm
import numpy as np
import random
import sys
sys.path.append('.')
from .BaseDataLoader import BaseDataloader


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

        fea_name_set = set()

        for fea_name, fea_config in self.config['user_feature_config'].items():
            # 判断fea name是否定义重复
            assert fea_name not in fea_name_set, 'The name of your feature is duplicated.'
            fea_name_set.add(fea_name)

            self.user_fea_config_dict[fea_name] = fea_config
            self.fea_config_dict[fea_name] = fea_config

            if fea_config['FeatureKind'] == 'number':
                self.user_fea_dim += 3
                continue

            if 'DependEmbeddingTableName' not in fea_config:
                setattr(self, fea_name, nn.Embedding(int(fea_config['MaxIndex']), int(fea_config['Dim'])))
            

            if fea_config['AggreateMethod'] in ['avgpooling', 'none']:
                self.user_fea_dim += fea_config['Dim']
            elif fea_config['AggreateMethod'] in ['padding']:
                self.user_fea_dim += fea_config['PaddingDim'] * fea_config['Dim']
        
        for fea_name, fea_config in self.config['item_feature_config'].items():
            # 判断fea name是否定义重复
            assert fea_name not in fea_name_set, 'The name of your feature is duplicated.'
            fea_name_set.add(fea_name)
            
            self.item_fea_config_dict[fea_name] = fea_config
            self.fea_config_dict[fea_name] = fea_config

            if fea_config['FeatureKind'] == 'number':
                self.user_fea_dim += 3
                continue

            if 'DependEmbeddingTableName' not in fea_config:
                setattr(self, fea_name, nn.Embedding(int(fea_config['MaxIndex']), int(fea_config['Dim'])))
            

            if fea_config['AggreateMethod'] in ['avgpooling', 'none']:
                self.item_fea_dim += fea_config['Dim']
            elif fea_config['AggreateMethod'] in ['padding']:
                self.item_fea_dim += fea_config['PaddingDim'] * fea_config['Dim']

        # 额外检查一下那些有DependEmbeddingTableName字段的特征的Dim是否和DependEmbeddingTableName字段的特征Dim相等
        for fea_k in self.fea_config_dict:
            fea_config = self.fea_config_dict[fea_k]
            if 'DependEmbeddingTableName' in fea_config:
                assert fea_config['Dim'] == self.fea_config_dict[fea_config['DependEmbeddingTableName']]['Dim'], \
                    'The dim of feature %s must equal to its DependEmbeddingTableName feature %s' % (fea_k, fea_config['DependEmbeddingTableName'])
    
    def get_data_embedding(self, data):
        user_embedding_data = {}
        item_embedding_data = {}
        user_number_data = {}
        item_number_data = {}
        label = None
        for key in data:
            if key != 'label':
                val, mask = data[key] # mask: bx1xpaddingDim

                if self.fea_config_dict[key]['FeatureKind'] == 'number':
                    if key in self.user_fea_config_dict:
                        user_number_data[key] = val
                    else:
                        item_number_data[key] = val
                    continue

                if 'DependEmbeddingTableName' not in self.fea_config_dict[key]:
                    embedding_data = getattr(self, key)(val)
                else:
                    embedding_data = getattr(self, self.fea_config_dict[key]['DependEmbeddingTableName'])(val)
                # embedding_data: bxpaddingDimxdim
                if len(embedding_data.shape) == 2:
                    embedding_data = embedding_data.unsqueeze(0)
                _, paddingDim, _ = embedding_data.shape
                if self.fea_config_dict[key]['AggreateMethod'] == 'padding':
                    embedding_data = embedding_data * mask.view(-1, paddingDim, 1)
                if self.fea_config_dict[key]['AggreateMethod'] == 'avgpooling':
                    # embedding_data = torch.mean(embedding_data, dim=0, keepdim=True)
                    embedding_data = embedding_data * mask.view(-1, paddingDim, 1)
                    embedding_data = torch.sum(embedding_data, dim=1, keepdim=True) # bx1xdim
                    embedding_data = embedding_data / torch.sum(mask, dim=-1, keepdim=True).unsqueeze(-1)
                if key in self.user_fea_config_dict:
                    user_embedding_data[key] = embedding_data
                else:
                    item_embedding_data[key] = embedding_data
            else:
                label = data[key]
        return user_embedding_data, item_embedding_data, user_number_data, item_number_data, label

    def load_model(self, model_path):
        """
        加载模型参数
        :param model_path: 模型文件路径
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件 {model_path} 不存在")
        self.load_state_dict(torch.load(model_path).state_dict())
        print('Load model successfully!')

if __name__ == '__main__':
    dataloader = BaseDataloader('./example.yaml')
    data = dataloader[1]
    model = BaseModel('./example.yaml')
    u, i, l = model.get_data_embedding(data)
    print(u, i, l)
    


