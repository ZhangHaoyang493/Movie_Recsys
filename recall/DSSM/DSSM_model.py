import sys
sys.path.append('../../')
sys.path.append('../')

from baseRecall import BaseRecallModel
from utils.model_utils import *
from utils.trainer import Trainer
from FeatureTools.BaseDataLoader import BaseDataloader
from FeatureTools.BaseModel import BaseModel
from utils.trainer import Trainer

import torch.nn.functional as F

import yaml


class DSSM(BaseModel):
    def __init__(self, config_file, model_config_file):
        super().__init__(config_file)

        with open(model_config_file, 'r') as f:
            self.model_config = yaml.safe_load(f)

        # print(self.model_config['dims'])
        user_dims = [self.user_fea_dim] + eval(str(self.model_config['dims']))
        item_dims = [self.item_fea_dim] + eval(str(self.model_config['dims']))
        self.user_tower = fc_model(user_dims)
        self.item_tower = fc_model(item_dims)


        self.random_negative_sample_ratio = 1

        self.bce_loss = nn.BCELoss()

    def forward(self, data):
        user_feature, item_feature, label = self.get_data_embedding(data)

        batch_size = label.shape[0]
        user_fea_embedding = torch.concat(list(user_feature.values()), dim=-1).view(batch_size, -1)
        item_fea_embedding = torch.concat(list(item_feature.values()), dim=0).view(batch_size, -1)

        # 用户塔和物料塔前向推理
        user_emb = self.user_tower(user_fea_embedding)  # Bx16
        item_emb = self.item_tower(item_fea_embedding)  # Bx16
        
        # 负采样
        negative_sample_emb = []
        negative_labels = []
        negative_user_emb = []
        for i in range(self.random_negative_sample_ratio):
            indices = torch.randperm(batch_size)
            negative_sample_emb.append(item_emb[indices])
            negative_user_emb.append(user_emb)
            negative_labels.append(torch.zeros(size=(batch_size, 1)))
        negative_sample_emb = torch.concat(negative_sample_emb, dim=0) # (self.random_negative_sample_ratio*B)x16
        negative_labels = torch.concat(negative_labels, dim=0) # (self.random_negative_sample_ratio*B)x1
        negative_user_emb = torch.concat(negative_user_emb, dim=0)


        all_item_emb = torch.concat([item_emb, negative_sample_emb], dim=0)
        all_labels = torch.concat([label, negative_labels], dim=0)
        all_user_emb = torch.concat([user_emb, negative_user_emb], dim=0)

        # 向量归一化
        normed_all_item_emb = F.normalize(all_item_emb, p=2, dim=1)
        normed_all_user_emb = F.normalize(all_user_emb, p=2, dim=1)
        

        # 计算相似度
        similar_degree = torch.sum(normed_all_item_emb * normed_all_user_emb, dim=-1, keepdim=True)
        similar_degree = torch.sigmoid(similar_degree)

        # 计算损失
        loss = self.bce_loss(similar_degree, all_labels)

        return {'loss': loss}
    
    def eval_(self, data):
        user_feature, item_feature, label = self.get_data_embedding(data)

        batch_size = label.shape[0]
        user_fea_embedding = torch.concat(list(user_feature.values()), dim=-1).view(batch_size, -1)
        item_fea_embedding = torch.concat(list(item_feature.values()), dim=0).view(batch_size, -1)

        # 用户塔和物料塔前向推理
        with torch.no_grad():
            user_emb = self.user_tower(user_fea_embedding)  # Bx16
            item_emb = self.item_tower(item_fea_embedding)  # Bx16
            normed_all_item_emb = F.normalize(user_emb, p=2, dim=1)
            normed_all_user_emb = F.normalize(item_emb, p=2, dim=1)
            similar_degree = torch.sum(normed_all_item_emb * normed_all_user_emb, dim=-1, keepdim=True)
            similar_degree = torch.sigmoid(similar_degree)
        return similar_degree, label

    # 只用来推理，获取用户的embedding
    def get_user_emb(self, user_data):
        batch_size = 1
        user_feature, _, _ = self.get_data_embedding(user_data)
        user_fea_embedding = torch.concat(list(user_feature.values()), dim=-1).view(batch_size, -1)
        with torch.no_grad():
            user_emb = self.user_tower(user_fea_embedding)  # Bx16
            user_emb = F.normalize(user_emb, p=2, dim=1)
        return user_emb.view(-1).detach().cpu().numpy()

    # 获取item的embedding
    def get_item_emb(self, item_data):
        batch_size = 1
        _, item_feature, _ = self.get_data_embedding(item_data)
        item_fea_embedding = torch.concat(list(item_feature.values()), dim=0).view(batch_size, -1)
        with torch.no_grad():
            item_emb = self.item_tower(item_fea_embedding)  # Bx16
            item_emb = F.normalize(item_emb, p=2, dim=1)
        return item_emb.view(-1).detach().cpu().numpy()

if __name__ == '__main__':
    model = DSSM('./feature_config.yaml', './model_config.yaml')
    trainer = Trainer('./model_config.yaml', './feature_config.yaml', model)
    trainer.train()





