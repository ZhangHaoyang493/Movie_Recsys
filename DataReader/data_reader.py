import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml
import os
import pyjson5 as json


# 构造一个torch的dataloader类，初始化给定一个包含各种slot id的json文件
class DataReader(Dataset):
    def __init__(self, config_path: str):
        # 读取json文件
        with open(config_path, 'r') as f:
            config = json.load(f)

        # 从json文件中获取各个配置参数
        self.sparse_slots = config.get('sparse_slots', None)
        self.dense_slots = config.get('dense_slots', None)
        self.array_slots = config.get('array_slots', None)
        self.embedding_size = config.get('embedding_size', None)
        self.embedding_table_size = config.get('embedding_table_size', None)
        self.data_path = config.get('data_path', None)
        self.share_slot_ids = config.get('share_slot_ids', {})
        self.array_max_length = config.get('array_max_length', {})
        
        # 读取数据文件
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file {self.data_path} does not exist")
        with open(self.data_path, 'r') as f:
            self.data_lines = f.readlines()
            self.data_lines = [line.strip() for line in self.data_lines]
        
        # 构建embedding表
        self.build_embedding_tables()

    # 构建embedding表
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
    def get_embedding(self, slot_id: int, idx: int):
        """
        获取slot id对应的embedding
        :param slot_id: slot id
        :param idx: 特征值对应的embedding索引
        :return: embedding向量
        """
        if slot_id not in self.embedding_tables:
            raise ValueError(f"Embedding table for slot_id {slot_id} does not exist")
        return self.embedding_tables[slot_id](idx)
    
    def __len__(self):
        return len(self.data_lines)
    
    def __getitem__(self, idx):
        line = self.data_lines[idx]
        # 解析每一行数据，格式为 slot_id:feature_hash_value slot_id:feature_hash_value ... \t label
        feature_part, label_part = line.split('\t')
        feature_items = feature_part.split(' ')
        
        ret_features = {}
        
        for item in feature_items:
            slot_id, emb_idx = item.split(':')
            slot_id = int(slot_id)
            
            if slot_id in self.sparse_slots:
                emb_idx = int(emb_idx)
                # 稀疏特征，转换为embedding索引
                ret_features[slot_id] = self.get_embedding(slot_id, torch.tensor(emb_idx))
            elif slot_id in self.dense_slots:
                dense_val = int(emb_idx)
                # 数值特征，直接转换为float
                ret_features[slot_id] = torch.tensor(float(dense_val), dtype=torch.float32)
            elif slot_id in self.array_slots:
                # 数组特征，转换为embedding索引的列表
                emb_indices = [int(i) for i in str(emb_idx).split(',')]  # 处理空字符串的情况
                # print(emb_indices)
                if str(slot_id) in self.share_slot_ids:
                    emb_slot_id = self.share_slot_ids[str(slot_id)]
                else:
                    emb_slot_id = slot_id
                embeddings = [self.get_embedding(emb_slot_id, torch.tensor(i)) for i in emb_indices]
                max_length = self.array_max_length.get(str(slot_id), None)
                if max_length is None:
                    raise ValueError(f"Max length for array slot_id {slot_id} is not specified in the config file")
                # 如果长度不足max_length，则进行padding。同时构造一个mask
                padding_mask = torch.ones(max_length, dtype=torch.float32)
                while len(embeddings) < max_length:
                    embeddings.append(torch.zeros(self.embedding_size[str(emb_slot_id)]))
                    padding_mask[len(embeddings) - 1] = 0  # padding位置的mask置0
                # 如果长度超过max_length，则进行截断
                embeddings = embeddings[:max_length]
                padding_mask = padding_mask[:max_length]

                ret_features[slot_id] = torch.stack(embeddings)  # 堆叠为一个tensor
                ret_features[f"{slot_id}_mask"] = padding_mask  # 添加mask
            else:
                raise ValueError(f"Slot id {slot_id} not found in either sparse or dense slots or array slots")

        ret_features['label'] = torch.tensor(float(label_part), dtype=torch.float32)

        return ret_features

if __name__ == "__main__":
    data_reader = DataReader('/Users/zhanghaoyang/Desktop/Movie_Recsys/feature.json')
    dataloader = DataLoader(data_reader, batch_size=8, shuffle=True)
    for batch in dataloader:
        print(batch)
        break