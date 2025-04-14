import sys
sys.path.append('..')
sys.path.append('../..')

from FeatureTools.BaseModel import BaseModel
from FeatureTools.BaseDataLoader import UserItemFeatureReader
from tqdm import tqdm

class BaseRecallModel:
    """
    所有的召回方法都会继承这个类，这个类主要包含一个评估召回层命中率的函数
    """

    def __init__(self, config_file, val_data_path):

        # key是userid，value是平分正向的物料的id
        self.val_data_dict = {}
        with open(val_data_path, 'r') as f:
            for line in f:
                line = line.strip()
                userid, movieid, score, time = line.split('::')
                if userid not in self.val_data_dict:
                    self.val_data_dict[userid] = []
                if float(score) >= 4:
                    self.val_data_dict[userid].append(movieid)
        self.featureReader = UserItemFeatureReader(config_file)
        
    
    # 对召回效果进行评估，基于命中率（Hit Rate）
    def eval(self, k=5):
        """
        k：对于每个用户召回多少条样本进行评估
        """
        # recall_res = {}
        hit_num = 0
        all_num = 0
        for userid in tqdm(self.val_data_dict.keys()):
            user_data = self.featureReader.get_user_feature(userid)
            recall_for_user = self.recall(userid, user_data, k)
            recall_res = set([x[0] for x in recall_for_user])
            for itemid in self.val_data_dict[userid]:
                if itemid in recall_res:
                    hit_num += 1
            all_num += len(self.val_data_dict[userid])
        
        print('Recall Hit Rate : %.2f%s' % ((hit_num / all_num) * 100, '%'))

    