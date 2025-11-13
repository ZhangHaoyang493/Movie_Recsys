import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pyjson5 as json
import os
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

class BaseModel(L.LightningModule):
    def __init__(self, config_path: str):
        super(BaseModel, self).__init__()
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # 从json文件中获取各个配置参数
        self.sparse_feature_names = self.config.get('sparse_feature_names', None)
        self.dense_feature_names = self.config.get('dense_feature_names', None)
        self.array_feature_names = self.config.get('array_feature_names', None)
        self.embedding_size = self.config.get('embedding_size', None)
        self.embedding_table_size = self.config.get('embedding_table_size', None)
        self.share_emb_table_features = self.config.get('share_emb_table_features', {})
        self.array_max_length = self.config.get('array_max_length', {})
        self.item_feature_names = self.config.get('item_feature_names', [])
        self.user_feature_names = self.config.get('user_feature_names', [])
        self.user_history_path = self.config.get('user_history_path', None)
        self.dense_feature_dim = self.config.get('dense_feature_dim', 0)
        self.emb_idx_2_val_path = self.config.get('embedding_idx_2_original_val_dict_path', None)
        self.val_2_emb_idx_path = self.config.get('original_val_2_embedding_idx_dict_path', None)

        self.sparse_feature_names = set(self.sparse_feature_names) if self.sparse_feature_names else set()
        self.dense_feature_names = set(self.dense_feature_names) if self.dense_feature_names else set()
        self.array_feature_names = set(self.array_feature_names) if self.array_feature_names else set()
        self.item_feature_names = set(self.item_feature_names) if self.item_feature_names else set()
        self.user_feature_names = set(self.user_feature_names) if self.user_feature_names else set()

        # 计算item侧和user侧的input的维度
        self.item_input_dim = 0
        self.user_input_dim = 0
        for feature_name in self.sparse_feature_names:
            if feature_name in self.share_emb_table_features:
                emb_feature_name = self.share_emb_table_features[feature_name]
            else:
                emb_feature_name = feature_name
            emb_size = self.embedding_size.get(emb_feature_name, 8)  # 默认embedding维度为8
            if feature_name in self.item_feature_names:
                self.item_input_dim += emb_size
            if feature_name in self.user_feature_names:
                self.user_input_dim += emb_size
        for feature_name in self.dense_feature_names:
            if feature_name in self.item_feature_names:
                self.item_input_dim += self.dense_feature_dim
            if feature_name in self.user_feature_names:
                self.user_input_dim += self.dense_feature_dim
        for feature_name in self.array_feature_names:
            if feature_name in self.share_emb_table_features:
                emb_feature_name = self.share_emb_table_features[feature_name]
            else:
                emb_feature_name = feature_name
            emb_size = self.embedding_size.get(emb_feature_name, 8)  # 默认embedding维度为8
            if feature_name in self.item_feature_names:
                self.item_input_dim += emb_size
            if feature_name in self.user_feature_names:
                self.user_input_dim += emb_size


        print(f"Item input dim: {self.item_input_dim}, User input dim: {self.user_input_dim}")
        # 构建embedding表
        self.build_embedding_tables()

        # 获取用户的评分历史
        if self.user_history_path:
            with open(self.user_history_path, 'r') as f:
                self.user_history = json.load(f)
        # 获取embedding索引和真实值的映射字典
        if self.emb_idx_2_val_path:
            with open(self.emb_idx_2_val_path, 'r') as f:
                self.emb_idx_2_val_dict = json.load(f)
        # 获取真实值和embedding索引的映射字典
        if self.val_2_emb_idx_path:
            with open(self.val_2_emb_idx_path, 'r') as f:
                self.val_2_emb_idx_dict = json.load(f)

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass

    def get_feature_name_embedding_size(self, feature_name: str):
        # 获取指定feature name的embedding维度
        if feature_name in self.share_emb_table_features:
            emb_feature_name = self.share_emb_table_features[feature_name]
        else:
            emb_feature_name = feature_name

        embedding_size = self.embedding_size.get(emb_feature_name, None)
        if embedding_size is None:
            raise ValueError(f"Embedding size for feature [{emb_feature_name}] is not specified in the config file")
        return embedding_size

    def build_embedding_tables(self):
        # 构建embedding表,使用ModuleDict管理多个embedding表，这样可以方便地将它们注册为模型的子模块。直接使用字典的话会导致参数无法被正确注册和更新。
        self.embedding_tables = nn.ModuleDict()
        # 处理稀疏类型的feature name
        for feature_name in self.sparse_feature_names:
            if feature_name in self.share_emb_table_features:
                emb_feature_name = self.share_emb_table_features[feature_name]
            else:
                emb_feature_name = feature_name

            if emb_feature_name in self.embedding_tables:
                continue  # 如果已经创建过共享的embedding表，则跳过


            # 获取当前feature name的embedding表大小和维度
            table_size = self.embedding_table_size.get(emb_feature_name, None)  # 默认表大小为None
            embedding_size = self.embedding_size.get(emb_feature_name, None)  # 默认embedding维度为None

            # 参数检查
            if table_size is None:
                raise ValueError(f"Embedding table size for feature [{emb_feature_name}] is not specified in the config file")
            if embedding_size is None:
                raise ValueError(f"Embedding size for feature [{emb_feature_name}] is not specified in the config file")

            # 创建当前feature name对应的embedding表
            self.embedding_tables[emb_feature_name] = nn.Embedding(table_size, embedding_size)

        # 处理array类型的feature name
        for feature_name in self.array_feature_names:
            if feature_name in self.share_emb_table_features:
                emb_feature_name = self.share_emb_table_features[str(feature_name)]
            else:
                emb_feature_name = feature_name

            if emb_feature_name in self.embedding_tables:
                continue  # 如果已经创建过共享的embedding表，则跳过

            # 获取当前feature name的embedding表大小和维度
            table_size = self.embedding_table_size.get(emb_feature_name, None)  # 默认表大小为1000
            embedding_size = self.embedding_size.get(emb_feature_name, None)  # 默认embedding维度为8

            # 参数检查
            if table_size is None:
                raise ValueError(f"Embedding table size for feature [{emb_feature_name}] is not specified in the config file")
            if embedding_size is None:
                raise ValueError(f"Embedding size for feature [{emb_feature_name}] is not specified in the config file")

            # 创建当前feature name对应的embedding表
            self.embedding_tables[emb_feature_name] = nn.Embedding(table_size, embedding_size)


        # 获取feature name对应的embedding
    def get_embedding(self, feature_name: str, idx: torch.Tensor):
        """
        获取feature name对应的embedding
        :param feature_name: feature name
        :param idx: 特征值对应的embedding索引
        :return: embedding向量
        """
        if feature_name not in self.embedding_tables:
            raise ValueError(f"Embedding table for feature [{feature_name}] does not exist")
        return self.embedding_tables[feature_name](idx)


    def get_features_embedding(self, feature_name, feature_value):
        """
        获取指定feature name对应的embedding
        :param feature_name: feature name
        :param feature_value: feature value
        :return: 指定feature name对应的embedding
        """

        if feature_name in self.sparse_feature_names:
            if str(feature_name) in self.share_emb_table_features:
                emb_feature_name = self.share_emb_table_features[feature_name]
            else:
                emb_feature_name = feature_name
            emb_idx = feature_value.long()
            return self.get_embedding(emb_feature_name, emb_idx)
        elif feature_name in self.dense_feature_names:
            dense_val = feature_value.float().unsqueeze(1)  # 数值特征，直接转换为float，并增加一个维度
            return dense_val
        elif feature_name in self.array_feature_names:
            emb_indices = feature_value  # 数组特征，已经是一个列表
            if str(feature_name) in self.share_emb_table_features:
                emb_feature_name = self.share_emb_table_features[feature_name]
            else:
                emb_feature_name = feature_name
            return self.get_embedding(emb_feature_name, emb_indices.long())
        else:
            raise ValueError(f"Feature name [{feature_name}] is not defined in sparse_feature_names, dense_feature_names or array_feature_names")
    
    # 对数组特征进行处理，可以在子类中重写，默认使用mean pooling
    def array_feature_process(self, embeddings):
        for feature_name in self.array_feature_names:
            if feature_name not in embeddings:
                continue
            emb = embeddings[feature_name]  # bxarr_lenxdim
            mask = embeddings.get(f"{feature_name}_mask", None)  # bxarr_lenxdim
            if mask is not None:
                emb = emb * mask.unsqueeze(-1)  # 应用mask
                emb = emb.sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)  # 避免除以零，mean pooling
                embeddings[feature_name] = emb


    # 根据feature name集合中包含的feature name获取对应的embedding字典
    def get_embedding_from_set(self, batch, feature_name_set):
        embeddings = {}
        for feature_name in feature_name_set:
            emb = self.get_features_embedding(feature_name, batch[feature_name])
            if feature_name in self.sparse_feature_names:
                embeddings[feature_name] = emb
            elif feature_name in self.dense_feature_names:
                embeddings[feature_name] = emb
            elif feature_name in self.array_feature_names:
                mask = batch.get(f"{feature_name}_mask", None)  # bxarr_lenxdim
                embeddings[feature_name] = emb
                embeddings[f"{feature_name}_mask"] = mask
        
        # 处理数组特征
        self.array_feature_process(embeddings)

        emb_to_cat = []
        feature_dims = []
        feature_names = []
        for feature_name in feature_name_set:
            emb = embeddings[feature_name]
            emb_to_cat.append(emb)
            feature_dims.append(emb.shape[1])
            feature_names.append(feature_name)
        
        return torch.cat(emb_to_cat, dim=1), feature_dims, feature_names  # 在特征维度上拼接
    
    @torch.no_grad()
    def inference(self, batch):
        pass

    @torch.no_grad()
    def eval(self, eval_items=['AUC']):
        if self.val_dataloader_ is None:
            print("No validation dataloader provided.")
            return

        eval_auc = 'AUC' in eval_items
        eval_gauc = 'GAUC' in eval_items

        user_score_dic = {}
        pred, truth = [], []
        print()
        for batch in tqdm(self.val_dataloader_, desc="Evaluating AUC", ncols=100):
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
            scores = self.inference(batch)

            labels = batch['label'][:, 1]  # 获取是否喜欢的标签
            scores = scores.view(-1).cpu().detach().numpy().tolist()
            labels = labels.view(-1).cpu().numpy().tolist()
            user_ids = batch['user_id'].view(-1).cpu().numpy().tolist()

            pred.extend(scores)
            truth.extend(labels)

            if eval_gauc:
                for uid, score, label in zip(user_ids, scores, labels):
                    if uid not in user_score_dic:
                        user_score_dic[uid] = {'scores': [], 'labels': [], 'labels_pos_count': 0}
                    user_score_dic[uid]['scores'].append(score)
                    user_score_dic[uid]['labels'].append(label)
                    if label == 1:
                        user_score_dic[uid]['labels_pos_count'] += 1

        avg_auc = roc_auc_score(truth, pred)

        if eval_gauc:
            sum_auc = 0.0
            num = 0
            for uid in user_score_dic:
                if user_score_dic[uid]['labels_pos_count'] == 0 or user_score_dic[uid]['labels_pos_count'] == len(user_score_dic[uid]['labels']):
                    continue  # 全为正样本或全为负样本，跳过
                sum_auc += roc_auc_score(user_score_dic[uid]['labels'], user_score_dic[uid]['scores'])
                num += 1
            print(num)
            gauc = sum_auc / num if num > 0 else 0.0

        self.log('Val_AUC', avg_auc)
        if eval_gauc:
            self.log('Val_GAUC', gauc)
        if eval_gauc:
            print(f"AUC: {avg_auc}, GAUC: {gauc}")
        else:
            print(f"AUC: {avg_auc}")




        

if __name__ == "__main__":
    import sys
    sys.path.append('/data2/zhy/Movie_Recsys')
    from DataReader.data_reader import DataReader
    from torch.utils.data import DataLoader

    data_reader = DataReader('/data2/zhy/Movie_Recsys/feature.json', '/data2/zhy/Movie_Recsys/FeatureFiles/train_ratings_features.txt')
    dataloader = DataLoader(data_reader, batch_size=8, shuffle=True)

    model = BaseModel('/data2/zhy/Movie_Recsys/feature.json')
    for batch in dataloader:
        for feature_name in batch:
            if feature_name != 'label' and feature_name[-5:] != '_mask':
                emb = model.get_features_embedding(feature_name, batch[feature_name])
                print(f"Feature: {feature_name}, Embedding shape: {emb.shape}")
        break