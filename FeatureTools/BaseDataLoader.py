import torch
from torch.utils.data import Dataset, DataLoader
import yaml
from .FeatureConfigReader import FeatureConfigReader, UserItemFeatureReader


class BaseDataloader(UserItemFeatureReader):
    def __init__(self, config_file, mode='train'):
        super().__init__(config_file)

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


    def __len__(self):
        return len(self.all_rating_data)
    
    def __getitem__(self, index):
        data = {}
        userid, movieid, score, _ = self.all_rating_data[index]

        user_data = self.get_user_feature(userid)
        item_data = self.get_item_feature(movieid)

        data.update(user_data)
        data.update(item_data)
        data['label'] = torch.tensor([1]) if float(score) >= 4 else torch.tensor([0])

        return data

# 这个dataloader会读取所有item的特征，用于双塔模型训练好之后存储所有的item向量用
# class ItemFeatureDataloader(BaseDataloader):
#     def __init__(self, config_file):
#         super().__init__(config_file)

#         self.item_data = list(self.item_feature.values())

#     def __len__(self):
#         return len(self.item_data)
    
#     def __getitem__(self, index):
#         movieid = self.item_data[index][0]
#         data = {}
#         for fea_name, fea_config in self.config['item_feature_config'].items():
#             fea_index = int(fea_config['Depend'])
#             fea = self.load_feature(self.item_feature[movieid], fea_index, fea_config)
#             data[fea_name] = fea
#         return data

# class UserItemFeatureReader(FeatureConfigReader):
#     def __init__(self, config_file):
#         super().__init__(config_file)
    
#     def get_user_feature(self, userid):
#         data = {}
#         for fea_name, fea_config in self.config['user_feature_config'].items():
#             fea_index = int(fea_config['Depend'])
#             fea = self.load_feature(self.user_feature[userid], fea_index, fea_config)
#             data[fea_name] = fea
#         return data
    
#     def get_item_feature(self, itemid):
#         data = {}
#         for fea_name, fea_config in self.config['item_feature_config'].items():
#             fea_index = int(fea_config['Depend'])
#             fea = self.load_feature(self.item_feature[itemid], fea_index, fea_config)
#             data[fea_name] = fea
#         return data
    



def get_dataloader(fea_config_file, batch_size: int=1, num_workers:int = 4, type: str='train'):
    dataset = BaseDataloader(fea_config_file, mode=type)
    if type == 'test':
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    elif type == 'train':
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    elif type == 'only_item':
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

if __name__ == '__main__':
    dataloader = BaseDataloader('./example.yaml')
    print(dataloader[1])