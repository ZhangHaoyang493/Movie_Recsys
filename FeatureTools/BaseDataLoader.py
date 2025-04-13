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
            1: 0,
            18: 1,
            25: 2,
            35: 3,
            45: 4,
            50: 5,
            56: 6
        }

        self.gender_hash = {
            'F': 0,
            'M': 1
        }



class BaseDataloader(Dataset):
    def __init__(self, config_file, mode='train'):
        super().__init__()

        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        # print(self.config)
        self.all_rating_data = []
        with open(self.config['train_data_path'] if mode=='train' else self.config['test_data_path'], 'r') as f:
            for line in f:
                line = line.strip()
                userid, movieid, score, time = line.split('::')
                score = float(score)
                time = int(time)
                self.all_rating_data.append([userid, movieid, score, time])

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

    def __len__(self):
        return len(self.all_rating_data)
    

    def type_convert(self, fea, type_, hashDictName=None):
        assert type_ in ['int', 'float', 'str']
        if type_ == 'int':
            return torch.tensor([int(fea)])
        elif type_ == 'float':
            return torch.tensor([float(fea)])
        elif type_ == 'str':
            assert hashDictName is not None
            return torch.tensor([int(getattr(self.hash_config, hashDictName)[str(fea)])])
    

    def load_feature(self, feas, fea_index, fea_config: dict):
        fea_kind = fea_config['FeatureKind']
        if fea_kind == 'kind':
            return [self.type_convert(feas[fea_index], fea_config['type'], fea_config.get('hashDictName', None)), None]
        elif fea_kind == 'kindarray':
            kind_array = eval(feas[fea_index])
            if fea_config['AggreateMethod'] == 'padding':
                ret = []
                mask = []
                for k in kind_array:
                    ret.append(self.type_convert(k, fea_config['type'], fea_config.get('hashDictName', None)))
                    mask.append(torch.tensor([1]))
                while len(ret) < int(fea_config['PaddingDim']):
                    ret.append(torch.tensor([0]))
                    mask.append(torch.tensor([0]))
                return [torch.tensor(ret), torch.tensor(mask)]
            elif fea_config['AggreateMethod'] == 'avgpooling':
                ret = []
                for k in kind_array:
                    ret.append(self.type_convert(k, fea_config['type'], fea_config.get('hashDictName', None)))
                return [torch.tensor(ret), None]
    
    def __getitem__(self, index):
        data = {}
        userid, movieid, score, _ = self.all_rating_data[index]
        for fea_name, fea_config in self.config['user_feature_config'].items():
            fea_index = int(fea_config['Depend'])
            fea = self.load_feature(self.user_feature[userid], fea_index, fea_config)
            data[fea_name] = fea
        for fea_name, fea_config in self.config['item_feature_config'].items():
            fea_index = int(fea_config['Depend'])
            fea = self.load_feature(self.item_feature[movieid], fea_index, fea_config)
            data[fea_name] = fea
        data['label'] = torch.tensor([1]) if float(score) >= 4 else torch.tensor([0])
        return data



if __name__ == '__main__':
    dataloader = BaseDataloader('./example.yaml')
    print(dataloader[1])