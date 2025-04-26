import sys
sys.path.append('..')
sys.path.append('../..')

import pickle
from baseRecall import BaseRecallModel
import faiss
import numpy as np
# from DSSM import DSSMModel
# import torch
import numpy as np
from DSSM_model import DSSM
import sys
sys.path.append('..')
from FeatureTools.BaseDataLoader import get_dataloader
from FeatureTools.FeatureConfigReader import UserItemFeatureReader
from tqdm import tqdm

class DSSMRecall(BaseRecallModel):
    def __init__(self, config_file, model_path, val_data_path, model_config_path, item_feature_path):
        super().__init__(config_file, val_data_path)

        self.DSSMModel = DSSM(config_file, model_config_path)
        self.DSSMModel.load_model(model_path)
        self.useritemFeatureReader = UserItemFeatureReader(config_file)

        self.index2itemId = {}
        item_emb_arr = []
        index = 0
        with open(item_feature_path, 'r') as f:
            for line in f:
                movieid = line.split('::')[0]
                itememb = self.DSSMModel.get_item_emb(self.useritemFeatureReader.get_item_feature(movieid))
                self.index2itemId[index] = movieid
                item_emb_arr.append(itememb)
                index += 1
        item_emb_arr = np.array(item_emb_arr)
        dimension = item_emb_arr.shape[1]  # 获取向量的维度
        self.index = faiss.IndexFlatIP(dimension)  # 使用内积度量
        self.index.add(item_emb_arr)  # 将向量添加到Faiss索引中


    def recall(self, userid, k=5):
        # user_emb = self.model.get_user_emb(torch.tensor([int(userid)]).view(1, 1))[0]
        # if userid in self.recall_res:
        #     return self.recall_res[userid]

        user_emb = self.DSSMModel.get_user_emb(self.useritemFeatureReader.get_user_feature(userid))
        user_emb = np.expand_dims(user_emb, axis=0) 
        distances, indices = self.index.search(user_emb, k)
        results = [(str(self.index2itemId[idx]), float(dist)) for idx, dist in zip(indices[0][:], distances[0][:])]
        # print(userid, results)
        # self.recall_res[userid] = results
        return results
    
    
if __name__ == '__main__':
    dssm_recall = DSSMRecall('./feature_config.yaml', 
                             './model/DSSM_epoch_9280.pth', 
                             '../../data/test_ratings.dat', 
                             './model_config.yaml',
                             '../../data/item_feature.dat')

    dssm_recall.eval(k=10)
    dssm_recall.eval(k=20)
    dssm_recall.eval(k=30)
    dssm_recall.eval(k=40)
    dssm_recall.eval(k=50)
    # dssm_recall.recall('1', k=30)