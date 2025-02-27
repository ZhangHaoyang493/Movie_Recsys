import pickle

class BaseRecall():
    def __init__(self):
        pass

    def eval(self, val_data_path, k=5):
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

    