import sys
sys.path.append('/Users/zhanghaoyang/Desktop/Movie_Recsys/recall')

import pickle
from baseRecall import BaseRecall
import faiss
import numpy as np
# from DSSM import DSSMModel
# import torch
import numpy as np


class DSSMRecall(BaseRecall):
    def __init__(self, item_emb_path, user_emb_path):
        super().__init__()
        
        self.user_emb = pickle.load(open(user_emb_path, 'rb'))
        self.item_emb = pickle.load(open(item_emb_path, 'rb'))

        # 将item emb存入faiss数据库
        self.index2itemId = {}
        index = 0
        item_emb_np = []
        for k in self.item_emb.keys():
            self.index2itemId[index] = k
            index += 1
            item_emb_np.append(self.item_emb[k])
        item_emb_np = np.array(item_emb_np)

        dimension = item_emb_np.shape[1]  # 获取向量的维度
        self.index = faiss.IndexFlatIP(dimension)  # 使用内积度量
        self.index.add(item_emb_np)  # 将向量添加到Faiss索引中



        # self.model: DSSMModel = torch.load(model_path) #DSSMModel(item_num=item_num, user_num=user_num, dim=16)  
        # torch.load(model_path)
        # self.model.load_state_dict(torch.load(model_path))
        # self.model.eval()


    def recall(self, userid, k=5):
        # user_emb = self.model.get_user_emb(torch.tensor([int(userid)]).view(1, 1))[0]

        user_emb = np.array(self.user_emb[int(userid)]).astype('float32')
        user_emb = np.expand_dims(user_emb, axis=0) 
        distances, indices = self.index.search(user_emb, k)
        results = [(str(self.index2itemId[idx]), float(dist)) for idx, dist in zip(indices[0][:], distances[0][:])]
        # print(userid, results)
        return results
    
    
if __name__ == '__main__':
    dssm_recall = DSSMRecall(
        '/Users/zhanghaoyang/Desktop/Movie_Recsys/recall/DSSM/item_emb_final.pkl',
        '/Users/zhanghaoyang/Desktop/Movie_Recsys/recall/DSSM/user_emb_final.pkl',
        # '/Users/zhanghaoyang/Desktop/Movie_Recsys/cache/int_2_item_id.pkl'
        # '/Users/zhanghaoyang/Desktop/Movie_Recsys/recall/DSSM/DSSM.pth',
    )

    # dssm_recall.eval(val_data_path='/Users/zhanghaoyang/Desktop/Movie_Recsys/cache/val_data.pkl', k=50)
    dssm_recall.recall('1', k=30)