import sys
sys.path.append('..')

import pickle
import os
import gensim
import yaml
import os.path as osp
from baseRecall import BaseRecall



class W2VRecall(BaseRecall):
    def __init__(self):
        
        with open('./w2v_conf.yaml', 'r') as f:
            conf = yaml.safe_load(f)

        self.model = gensim.models.word2vec.Word2Vec.load(osp.join(conf['save_path'], 'word2vec_model.pth'))
        self.train_readlist = pickle.load(open(conf['train_readlist_path'], 'rb'))

    def recall(self, user_id, k=5):
        recall_res = {}
        sim_info = []
        for item_info in self.train_readlist[user_id]:
            item_id = item_info[0]
            similar_words = self.model.wv.most_similar(item_id, topn=k)
            sim_info.append(similar_words)
        for l in sim_info:
            for w in l:
                if w[0] not in recall_res:
                    recall_res[w[0]] = 0
                recall_res[w[0]] += w[1]
        recall_res = list(recall_res.items())
        recall_res = sorted(recall_res, key=lambda x: x[1], reverse=True)[:k]
        return recall_res

if __name__ == '__main__':
    model = W2VRecall()
    model.eval('/Users/zhanghaoyang/Desktop/Movie_Recsys/cache/val_data.pkl', k=50)