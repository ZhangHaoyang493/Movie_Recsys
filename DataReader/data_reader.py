import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml
import os
import pyjson5 as json


# 构造一个torch的dataloader类，初始化给定一个包含各种slot id的json文件
class DataReader(Dataset):
    def __init__(self, config_path: str, feature_file_path: str = None):
        # 读取json文件
        with open(config_path, 'r') as f:
            config = json.load(f)

        # 从json文件中获取各个配置参数
        self.sparse_slots = config.get('sparse_slots', None)
        self.dense_slots = config.get('dense_slots', None)
        self.array_slots = config.get('array_slots', None)
        self.data_path = feature_file_path
        self.array_max_length = config.get('array_max_length', {})

        if self.data_path is None:
            raise ValueError("feature_file_path must be provided either in config file or as an argument")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file {self.data_path} does not exist")
        
        # 读取数据文件
        with open(self.data_path, 'r') as f:
            self.data_lines = f.readlines()
            self.data_lines = [line.strip() for line in self.data_lines]
    
    def __len__(self):
        return len(self.data_lines)
    
    def __getitem__(self, idx):
        line = self.data_lines[idx]
        # 解析每一行数据，格式为 slot_id:feature_hash_value slot_id:feature_hash_value ... \t label
        feature_part, label_part = line.split('\t')
        feature_items = feature_part.split(' ')
        
        ret_datas = {}
        
        for item in feature_items:
            slot_id, emb_idx = item.split(':')
            slot_id = int(slot_id)
            
            if slot_id in self.sparse_slots:
                emb_idx = int(emb_idx)
                # 稀疏特征，转换为embedding索引
                ret_datas[slot_id] = emb_idx
            elif slot_id in self.dense_slots:
                dense_val = int(emb_idx)
                # 数值特征，直接转换为float
                ret_datas[slot_id] = dense_val
            elif slot_id in self.array_slots:
                # 数组特征，转换为embedding索引的列表
                emb_indices = [int(i) for i in str(emb_idx).split(',')]  # 处理空字符串的情况
                max_length = self.array_max_length.get(str(slot_id), None)
                if max_length is None:
                    raise ValueError(f"Max length for array slot_id {slot_id} is not specified in the config file")
                
                # 如果长度不足max_length，则进行padding。同时构造一个mask
                padding_mask = torch.ones(max_length, dtype=torch.float32)
                while len(emb_indices) < max_length:
                    emb_indices.append(0)  # 使用0进行padding
                    padding_mask[len(emb_indices) - 1] = 0  # padding位置的mask置0
                # 如果长度超过max_length，则进行截断
                emb_indices = emb_indices[:max_length]
                padding_mask = padding_mask[:max_length]

                emb_indices = torch.tensor(emb_indices, dtype=torch.long)
                ret_datas[slot_id] = emb_indices  # 堆叠为一个tensor
                ret_datas[f"{slot_id}_mask"] = padding_mask  # 添加mask
            else:
                raise ValueError(f"Slot id {slot_id} not found in either sparse or dense slots or array slots")

        ret_datas['label'] = torch.tensor(float(label_part), dtype=torch.float32)

        return ret_datas



if __name__ == "__main__":
    data_reader = DataReader('/Users/zhanghaoyang/Desktop/Movie_Recsys/feature.json', '/Users/zhanghaoyang/Desktop/Movie_Recsys/FeatureFiles/train_ratings_features.txt')
    dataloader = DataLoader(data_reader, batch_size=8, shuffle=True)
    for batch in dataloader:
        for key, value in batch.items():
            print(key)
            print(value.shape)
        break