import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pyjson5 as json
import os

class BaseModel(L.LightningModule):
    def __init__(self, config_path: str):
        super(BaseModel, self).__init__()
        with open(config_path, 'r') as f:
            config = json.load(f)

        # 从json文件中获取各个配置参数
        self.sparse_slots = config.get('sparse_slots', None)
        self.dense_slots = config.get('dense_slots', None)
        self.array_slots = config.get('array_slots', None)
        self.embedding_size = config.get('embedding_size', None)
        self.embedding_table_size = config.get('embedding_table_size', None)
        self.share_slot_ids = config.get('share_slot_ids', {})
        self.array_max_length = config.get('array_max_length', {})
        self.item_slots = config.get('item_slots', [])
        self.user_slots = config.get('user_slots', [])
        self.dense_feature_dim = config.get('dense_feature_dim', 0)

        self.sparse_slots = set(self.sparse_slots) if self.sparse_slots else set()
        self.dense_slots = set(self.dense_slots) if self.dense_slots else set()
        self.array_slots = set(self.array_slots) if self.array_slots else set()
        self.item_slots = set(self.item_slots) if self.item_slots else set()
        self.user_slots = set(self.user_slots) if self.user_slots else set()

        # 计算item侧和user侧的input的维度
        self.item_input_dim = 0
        self.user_input_dim = 0
        for slot_id in self.sparse_slots:
            emb_size = self.embedding_size.get(str(slot_id), 8)  # 默认embedding维度为8
            if slot_id in self.item_slots:
                self.item_input_dim += emb_size
            if slot_id in self.user_slots:
                self.user_input_dim += emb_size
        for slot_id in self.dense_slots:
            if slot_id in self.item_slots:
                self.item_input_dim += self.dense_feature_dim
            if slot_id in self.user_slots:
                self.user_input_dim += self.dense_feature_dim
        for slot_id in self.array_slots:
            if str(slot_id) in self.share_slot_ids:
                emb_slot_id = self.share_slot_ids[str(slot_id)]
            else:
                emb_slot_id = slot_id
            emb_size = self.embedding_size.get(str(emb_slot_id), 8)  # 默认embedding维度为8
            if slot_id in self.item_slots:
                self.item_input_dim += emb_size
            if slot_id in self.user_slots:
                self.user_input_dim += emb_size


        print(f"Item input dim: {self.item_input_dim}, User input dim: {self.user_input_dim}")
        # 构建embedding表
        self.build_embedding_tables()

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

    def build_embedding_tables(self):
        # 构建embedding表
        self.embedding_tables = {}
        # 处理稀疏类型的slot id
        for slot_id in self.sparse_slots:
            # 获取当前slot的embedding表大小和维度
            table_size = self.embedding_table_size.get(str(slot_id), None)  # 默认表大小为1000
            embedding_size = self.embedding_size.get(str(slot_id), None)  # 默认embedding维度为8
            
            # 参数检查
            if table_size is None:
                raise ValueError(f"Embedding table size for slot_id {slot_id} is not specified in the config file")
            if embedding_size is None:
                raise ValueError(f"Embedding size for slot_id {slot_id} is not specified in the config file")

            # 创建当前slot id对应的embedding表
            self.embedding_tables[slot_id] = nn.Embedding(table_size, embedding_size)

        # 处理array类型的slot id
        for slot_id in self.array_slots:
            if str(slot_id) in self.share_slot_ids:
                emb_slot_id = self.share_slot_ids[str(slot_id)]
            else:
                emb_slot_id = slot_id
            
            if emb_slot_id in self.embedding_tables:
                continue  # 如果已经创建过共享的embedding表，则跳过

            # 获取当前slot的embedding表大小和维度
            table_size = self.embedding_table_size.get(str(emb_slot_id), None)  # 默认表大小为1000
            embedding_size = self.embedding_size.get(str(emb_slot_id), None)  # 默认embedding维度为8
            
            # 参数检查
            if table_size is None:
                raise ValueError(f"Embedding table size for slot_id {emb_slot_id} is not specified in the config file")
            if embedding_size is None:
                raise ValueError(f"Embedding size for slot_id {emb_slot_id} is not specified in the config file")

            # 创建当前slot id对应的embedding表
            self.embedding_tables[emb_slot_id] = nn.Embedding(table_size, embedding_size)


        # 获取slot id对应的embedding
    def get_embedding(self, slot_id: int, idx: torch.Tensor):
        """
        获取slot id对应的embedding
        :param slot_id: slot id
        :param idx: 特征值对应的embedding索引
        :return: embedding向量
        """
        if slot_id not in self.embedding_tables:
            raise ValueError(f"Embedding table for slot_id {slot_id} does not exist")
        return self.embedding_tables[slot_id](idx)


    # def get_slots_feature(self, batch):
    #     """
    #     获取一个batch中所有slot id对应的embedding
    #     :param batch: 一个batch的数据，格式为 {slot_id: feature_value, ...}
    #     :return: 一个batch中所有slot id对应的embedding，格式为 {slot_id: embedding_tensor, ...}
    #     """
    #     self.slot_embeddings = {}
    #     for slot_id, feature_value in batch.items():
    #         # 如果slot id是字符串，则跳过
    #         if isinstance(slot_id, str):
    #             continue  # 跳过mask字段

    #         if slot_id in self.sparse_slots:
    #             emb_idx = feature_value.long()
    #             self.slot_embeddings[slot_id] = self.get_embedding(slot_id, emb_idx)
    #         elif slot_id in self.dense_slots:
    #             dense_val = feature_value.float().unsqueeze(1)  # 数值特征，直接转换为float，并增加一个维度
    #             self.slot_embeddings[slot_id] = dense_val
    #         elif slot_id in self.array_slots:
    #             emb_indices = feature_value  # 数组特征，已经是一个列表
    #             if str(slot_id) in self.share_slot_ids:
    #                 emb_slot_id = self.share_slot_ids[str(slot_id)]
    #             else:
    #                 emb_slot_id = slot_id
    #             self.slot_embeddings[slot_id] = self.get_embedding(emb_slot_id, emb_indices.long())
    #         else:
    #             raise ValueError(f"Slot id {slot_id} is not defined in sparse_slots, dense_slots or array_slots")
            
    def get_slots_embedding(self, slot_id, slot_value):
        """
        获取指定slot id对应的embedding
        :param slot_id: slot id
        :param slot_value: slot value
        :return: 指定slot id对应的embedding
        """
        if slot_id in self.sparse_slots:
            emb_idx = slot_value.long()
            return self.get_embedding(slot_id, emb_idx)
        elif slot_id in self.dense_slots:
            dense_val = slot_value.float().unsqueeze(1)  # 数值特征，直接转换为float，并增加一个维度
            return dense_val
        elif slot_id in self.array_slots:
            emb_indices = slot_value  # 数组特征，已经是一个列表
            if str(slot_id) in self.share_slot_ids:
                emb_slot_id = self.share_slot_ids[str(slot_id)]
            else:
                emb_slot_id = slot_id
            return self.get_embedding(emb_slot_id, emb_indices.long())
        else:
            raise ValueError(f"Slot id {slot_id} is not defined in sparse_slots, dense_slots or array_slots")
            
    # def get_user_feature(self):
    #     """
    #     获取用户侧的特征向量
    #     :return: 用户侧的特征向量，形状为 (batch_size, user_input_dim)
    #     """
    #     pass
        # user_features = []
        # for slot_id in self.user_slots:
        #     if slot_id not in self.slot_embeddings:
        #         raise ValueError(f"User slot_id {slot_id} not found in the batch data")
        #     emb = self.slot_embeddings[slot_id]
        #     if slot_id in self.array_slots:
        #         mask = self.slot_embeddings.get(f"{slot_id}_mask", None)
        #         if mask is None:
        #             raise ValueError(f"Mask for array slot_id {slot_id} not found in the batch data")
        #         # 对数组类型的embedding进行mask处理，然后取平均
        #         emb = emb * mask.unsqueeze(2)  # 扩展mask维度以匹配emb的维度
        #         emb = emb.sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)  # 避免除0
        #     user_features.append(emb)
        
        # user_feature_vector = torch.cat(user_features, dim=1)  # 在特征维度上拼接
        # return user_feature_vector
            
        

if __name__ == "__main__":
    import sys
    sys.path.append('/Users/zhanghaoyang/Desktop/Movie_Recsys')
    from DataReader.data_reader import DataReader
    from torch.utils.data import DataLoader

    data_reader = DataReader('/Users/zhanghaoyang/Desktop/Movie_Recsys/feature.json', '/Users/zhanghaoyang/Desktop/Movie_Recsys/FeatureFiles/train_ratings_features.txt')
    dataloader = DataLoader(data_reader, batch_size=8, shuffle=True)

    model = BaseModel('/Users/zhanghaoyang/Desktop/Movie_Recsys/feature.json')
    for batch in dataloader:
        model.get_slots_feature(batch)
        for slot_id, emb in model.slot_embeddings.items():
            print(f"Slot ID: {slot_id}, Embedding Shape: {emb.shape}")
        break