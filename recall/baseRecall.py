import pickle

class BaseRecall():
    """
    所有的召回方法都会继承这个类，这个类主要包含一个评估召回层命中率的函数
    """

    def __init__(self):
        pass
    
    # 对召回效果进行评估，基于命中率（Hit Rate）
    def eval(self, val_data_path, k=5):
        """
        val_data_path：验证数据的路径
        k：对于每个用户召回多少条样本进行评估
        """
        val_data = pickle.load(open(val_data_path, 'rb'))
        recall_res = {}
        recall_num = 0
        all_num = 0
        for userid in val_data.keys():
            recall_for_user = self.recall(userid, k)
            recall_res[userid] = [x[0] for x in recall_for_user]
            for data in val_data[userid]:
                if data[1] >= 4.0:
                    all_num += 1
                    if data[0] in recall_res[userid]:
                        recall_num += 1
                

            if val_data[userid][0] in recall_res[userid]:
                recall_num += 1
        print('Recall Precision: %.2f%s' % (recall_num / all_num * 100, '%'))

    