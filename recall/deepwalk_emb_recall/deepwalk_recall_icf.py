import sys
sys.path.append('..')

import pickle
import os
import gensim
import yaml
import faiss
import numpy as np
import os.path as osp
from baseRecall import BaseRecall



class DeepWalkRecall(BaseRecall):
    def __init__(self):
        
        with open('./deepwalk_conf.yaml', 'r') as f:
            conf = yaml.safe_load(f)

        self.train_readlist = pickle.load(open(conf['train_readlist_path'], 'rb'))

        # 解析存储的embedding
        index_id_dict = {}
        id_index_dict = {}
        all_emb = []
        with open(conf['emb_path'], 'r') as f:
            line_num = 0
            for line in f:
                data = line.split(' ')
                # 第一行跳过
                if line_num == 0:
                    line_num += 1
                    continue
                id = data[0]
                index_id_dict[line_num - 1] = id
                id_index_dict[id] = line_num - 1
                id_emb = data[1:]
                id_emb = [float(d) for d in id_emb]
                all_emb.append(id_emb)
                line_num += 1
        
        item_emb_np = np.array(all_emb)

        # 存入faiss数据库
        dimension = item_emb_np.shape[1]  # 获取向量的维度
        # self.index = faiss.IndexFlatL2(dimension)  # 使用L2距离度量
        self.index = faiss.IndexFlatIP(dimension)  # 使用内积度量
        self.index.add(item_emb_np)  # 将向量添加到Faiss索引中

        self.index_id_dict = index_id_dict
        self.id_index_dict = id_index_dict
        self.item_emb_np = item_emb_np

        self.user_num = conf['user_num']

    def recall(self, user_id, k=5):

        # user_emb = np.array(self.user_emb[int(userid)]).astype('float32')
        # user_emb = np.expand_dims(user_emb, axis=0) 
        # distances, indices = self.index.search(user_emb, k + 1)
        # results = [(str(self.index2itemId[idx]), float(dist)) for idx, dist in zip(indices[0][:], distances[0][:])]
        # # print(userid, results)
        # return results


        recall_res = {}
        
        # sim_info = []
        for item_info in self.train_readlist[user_id]:
            item_id = item_info[0]
            item_id = str(int(item_id) + self.user_num)
            item_emb = np.array(self.item_emb_np[self.id_index_dict[item_id]]).astype('float32')
            item_emb = np.expand_dims(item_emb, axis=0)
            distances, indices = self.index.search(item_emb, k + 1)
            for idx, dist in zip(indices[0][1:], distances[0][1:]):
                id = self.index_id_dict[idx]
                if id not in recall_res:
                    recall_res[id] = 0
                recall_res[id] += dist
            

        # for l in sim_info:
        #     for w in l:
        #         if w[0] not in recall_res:
        #             recall_res[w[0]] = 0
        #         recall_res[w[0]] += w[1]
        recall_res = list(recall_res.items())
        recall_res = sorted(recall_res, key=lambda x: x[1], reverse=True)[:k]
        recall_res = [(str(int(id) - self.user_num), sim) for id, sim in recall_res]
        return recall_res
    


if __name__ == '__main__':
    model = DeepWalkRecall()
    model.eval('/Users/zhanghaoyang/Desktop/Movie_Recsys/cache/val_data.pkl', k=50)
    # print(model.recall('9999'))