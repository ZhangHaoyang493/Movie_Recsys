import sys
sys.path.append('..')

import pickle
from baseRecall import BaseRecall
import faiss
import numpy as np

class MFRecall(BaseRecall):
    def __init__(self, user_emb_path, item_emb_path):
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

    def recall(self, userid, k=5):
        user_emb = np.array(self.user_emb[int(userid)]).astype('float32')
        user_emb = np.expand_dims(user_emb, axis=0) 
        distances, indices = self.index.search(user_emb, k + 1)
        results = [(str(self.index2itemId[idx]), float(dist)) for idx, dist in zip(indices[0][:], distances[0][:])]
        # print(userid, results)
        return results
    
    
if __name__ == '__main__':
    mf_recall = MFRecall(
        '/Users/zhanghaoyang/Desktop/Movie_Recsys/recall/MF/MF_emb/MF_user_emb.pkl',
        '/Users/zhanghaoyang/Desktop/Movie_Recsys/recall/MF/MF_emb/MF_item_emb.pkl',
    )

    mf_recall.eval(val_data_path='/Users/zhanghaoyang/Desktop/Movie_Recsys/cache/val_data.pkl', k=50)