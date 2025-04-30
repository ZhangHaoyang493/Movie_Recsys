import os
import torch
from torch import optim, nn, utils, Tensor
from torch.utils.data import Dataset, DataLoader
import yaml
from tqdm import tqdm
import numpy as np
import random

class HashConfig:
    def __init__(self):
        self.movie_genres_hash = {
            'Action': 0,
            'Adventure': 1,
            'Animation': 2,
            "Children's": 3,
            'Comedy': 4,
            'Crime': 5,
            'Documentary': 6,
            'Drama': 7,
            'Fantasy': 8,
            'Film-Noir': 9,
            'Horror': 10,
            'Musical': 11,
            'Mystery': 12,
            'Romance': 13,
            'Sci-Fi': 14,
            'Thriller': 15,
            'War': 16,
            'Western': 17
        }

        self.age_hash = {
            '1': 0,
            '18': 1,
            '25': 2,
            '35': 3,
            '45': 4,
            '50': 5,
            '56': 6
        }

        self.gender_hash = {
            'F': 0,
            'M': 1
        }


# FeatureConfigReader负责根据feature的配置文件读取特征
class FeatureConfigReader(Dataset):
    def __init__(self, config_file):
        super().__init__()

        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)

        self.user_feature = {}
        with open(self.config['user_feature_file_path'], 'r') as f:
            for line in f:
                line = line.strip()
                infos = line.split('::')
                userid = infos[0]
                self.user_feature[userid] = infos
        
        self.item_feature = {}
        with open(self.config['item_feature_file_path'], 'r') as f:
            for line in f:
                line = line.strip()
                infos = line.split('::')
                itemid = infos[0]
                self.item_feature[itemid] = infos

        self.hash_config = HashConfig()

    def bucketize(self, float_fea, bucket_bounds):
        for i, bound in enumerate(bucket_bounds):
            if float_fea < bound:
                return i
        return len(bucket_bounds)

    def type_convert(self, fea, type_, hashDictName=None, bucket_bounds=None):
        assert type_ in ['int', 'float', 'str']
        if type_ == 'int':
            return torch.tensor([int(fea)])
        elif type_ == 'float':
            float_fea = float(fea)
            if bucket_bounds:
                assert len(bucket_bounds) > 0
                return torch.tensor([int(self.bucketize(float_fea, bucket_bounds))])
            return torch.tensor([float_fea])
        elif type_ == 'str':   # 如果是string类型的特征输入，需要指定将string转为int的dict
            assert hashDictName is not None
            return torch.tensor([int(getattr(self.hash_config, hashDictName)[str(fea)])])
    

    def load_feature(self, feas, fea_index, fea_config: dict):
        fea_kind = fea_config['FeatureKind']
        if fea_kind == 'kind':
            return [self.type_convert(feas[fea_index], fea_config['type'], fea_config.get('hashDictName', None)), torch.tensor([-1])]
        elif fea_kind == 'kindarray':
            kind_array = eval(feas[fea_index])
            # if fea_config['AggreateMethod'] == 'padding':
            ret = []
            mask = []
            for k in kind_array:
                if len(ret) >= int(fea_config['PaddingDim']):  # ret的长度大于int(fea_config['PaddingDim'])的话，就break掉
                    break
                ret.append(self.type_convert(k, fea_config['type'], fea_config.get('hashDictName', None)))
                mask.append(torch.tensor([1]))
            while len(ret) < int(fea_config['PaddingDim']):
                ret.append(torch.tensor([0]))
                mask.append(torch.tensor([0]))
        
            return [torch.tensor(ret), torch.tensor(mask)]
        elif fea_kind == 'number':
            return [self.type_convert(feas[fea_index], 'float'), torch.tensor([-1])]
        elif fea_kind == 'number_bucket':
            assert 'BucketBounds' in fea_config
            return [self.type_convert(feas[fea_index], 'float', bucket_bounds=eval(fea_config.get('BucketBounds', None))), torch.tensor([-1])]
            # elif fea_config['AggreateMethod'] == 'avgpooling':
            #     ret = []
            #     for k in kind_array:
            #         ret.append(self.type_convert(k, fea_config['type'], fea_config.get('hashDictName', None)))
            #     return [torch.tensor(ret), torch.tensor([-1])]


class UserItemFeatureReader(FeatureConfigReader):
    def __init__(self, config_file):
        super().__init__(config_file)
    
    def get_user_feature(self, userid):
        data = {}
        for fea_name, fea_config in self.config['user_feature_config'].items():
            fea_index = int(fea_config['Depend'])
            fea = self.load_feature(self.user_feature[userid], fea_index, fea_config)
            data[fea_name] = fea
        return data
    
    def get_item_feature(self, itemid):
        data = {}
        for fea_name, fea_config in self.config['item_feature_config'].items():
            fea_index = int(fea_config['Depend'])
            fea = self.load_feature(self.item_feature[itemid], fea_index, fea_config)
            data[fea_name] = fea
        return data