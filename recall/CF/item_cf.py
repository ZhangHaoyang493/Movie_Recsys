import sys
sys.path.append('..')


import os
import pickle
import yaml
from tqdm import tqdm
from typing import List
from baseRecall import BaseRecall

# itemCF，从train_readlist中构建出物品和用户的共现矩阵，然后计算物品之间的相似度

class ItemCF(BaseRecall):
    def __init__(self, conf=''):
        conf = yaml.safe_load(open(conf, 'r'))
        self.conf = conf
        train_readlist = pickle.load(open(os.path.join(conf['basedir'], 'cache', 'train_readlist.pkl'), 'rb'))

        item_user_list = {}
        user_item_set = {}

        # 构建item user list： item:{user1: rating, user2: rating, ...}
        print('Building item_user list...')
        for userid in tqdm(train_readlist):
            user_readlist = train_readlist[userid]
            user_item_set[userid] = set([x[0] for x in user_readlist])
            for rate_info in user_readlist:
                item, rating, timestamp = rate_info
                if item not in item_user_list:
                    item_user_list[item] = {}
                item_user_list[item][userid] = rating
        
        self.train_readlist = train_readlist
        self.item_user_list = item_user_list
        self.user_item_set = user_item_set

        # 构建item-item相似矩阵
        item_item_sim_matrix = {}

        if not os.path.exists(os.path.join(conf['basedir'], 'cache', 'ii_sim_matrix.pkl')):
            print('Building item-item similar matrix...')
            for item1 in tqdm(item_user_list.keys()):
                for item2 in item_user_list.keys():
                    if item1 == item2:
                        continue
                    if item1 not in item_item_sim_matrix:
                        item_item_sim_matrix[item1] = {}
                    for user in item_user_list[item1].keys():
                        if user in item_user_list[item2]:
                            if item2 not in item_item_sim_matrix[item1]:
                                item_item_sim_matrix[item1][item2] = 0
                            item_item_sim_matrix[item1][item2] += item_user_list[item1][user] * item_user_list[item2][user]
            
            for item1 in tqdm(item_item_sim_matrix.keys()):
                for item2 in item_item_sim_matrix[item1].keys():
                    len_item1 = sum([v**2 for v in item_user_list[item1].values()]) ** 0.5
                    len_item2 = sum([v**2 for v in item_user_list[item2].values()]) ** 0.5
                    item_item_sim_matrix[item1][item2] /= (len_item1 * len_item2)

            pickle.dump(item_item_sim_matrix, open(os.path.join(conf['basedir'], 'cache', 'ii_sim_matrix.pkl'), 'wb'))
        else:
            item_item_sim_matrix = pickle.load(open(os.path.join(conf['basedir'], 'cache', 'ii_sim_matrix.pkl'), 'rb'))
        
        self.item_item_sim_matrix = item_item_sim_matrix

        # 对ii相似性矩阵的每个物品相似的物品进行排序
        item_item_sim_sorted_matrix = {}
        if not os.path.exists(os.path.join(conf['basedir'], 'cache', 'ii_sim_sorted_matrix.pkl')):
            for item in tqdm(item_item_sim_matrix.keys()):
                item_sim_list = []
                for k, v in item_item_sim_matrix[item].items():
                    item_sim_list.append((k, v))
                item_item_sim_sorted_matrix[item] = sorted(item_sim_list, key=lambda x: x[1], reverse=True)
            pickle.dump(item_item_sim_sorted_matrix, open(os.path.join(conf['basedir'], 'cache', 'ii_sim_sorted_matrix.pkl'), 'wb'))
        else:
            item_item_sim_sorted_matrix = pickle.load(open(os.path.join(conf['basedir'], 'cache', 'ii_sim_sorted_matrix.pkl'), 'rb'))

        self.item_item_sim_sorted_matrix = item_item_sim_sorted_matrix
    # def print_item_similar

    def recall(self, userid: str, topk: int=5) -> List[str]:
        read_list = self.train_readlist[userid]
        recall_list = {}
        simlar_item = []
        for ite in read_list:
            # 如果用户给了正向的评分
            if ite[1] >= 4.0:
                simlar_list = self.item_item_sim_sorted_matrix[ite[0]]
                for sim_ite in simlar_list[:topk]:
                    if sim_ite[0] not in recall_list:
                        recall_list[sim_ite[0]] = 0
                    recall_list[sim_ite[0]] += sim_ite[1] * ite[1]
        recall_list = [(k, v) for k, v in recall_list.items()]
        recall_list = sorted(recall_list, key=lambda x: x[1], reverse=True)
        return recall_list[:topk]
    
    # def eval(self, topk: int=5):
    #     val_data = pickle.load(open(os.path.join(self.conf['basedir'], 'cache', 'val_data.pkl'), 'rb'))
    #     recall_res = {}
    #     recall_num = 0
    #     for userid in val_data.keys():
    #         recall_for_user = self.item_cf_recall(userid, topk)
    #         recall_res[userid] = [x[0] for x in recall_for_user]
    #         if val_data[userid][0] in recall_res[userid]:
    #             recall_num += 1
    #     print('Recall Precision: %.2f%s' % (recall_num / len(val_data.keys()) * 100, '%'))




if __name__ == '__main__':
    itemcf = ItemCF('./cf_conf.yaml')
    itemcf.eval('/Users/zhanghaoyang/Desktop/Movie_Recsys/cache/val_data.pkl', 50)