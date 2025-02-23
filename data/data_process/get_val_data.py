import os
import os.path as osp
import pickle
import yaml
import random

# 读取readlist，然后选取一部分的用户作为验证组，将他们打分为正向的最后一次记录拿出来作为验证数据
# 然后这一次后面的数据要丢弃
# conf = yaml.safe_load(open('/Users/zhanghaoyang/Desktop/Movie_Recsys/global_conf.yaml', 'r'))

readlist = pickle.load(open('/Users/zhanghaoyang/Desktop/Movie_Recsys/cache/readlist.pkl', 'rb'))
userid = readlist.keys()
val_userid = list(userid)# random.sample(userid, 1500)

val_data = {}

for v_userid in val_userid:
    v_user_readlist = readlist[v_userid]
    # 倒着选择最后一次正向打分
    index = len(v_user_readlist) - 1
    while index >= 0:
        if v_user_readlist[index][1] >= 4:
            break
        else:
            index -= 1
    #  确保把最起码用户最后两次点击的数据选到测试集中
    if len(v_user_readlist) - index < 2:
        while index >= 0 and len(v_user_readlist) - index < 2:
            index -= 1

    val_data[v_userid] = v_user_readlist[index:]
    readlist[v_userid] = v_user_readlist[:index]

print(len(val_data.keys()))

pickle.dump(val_data, open(osp.join('/Users/zhanghaoyang/Desktop/Movie_Recsys/cache', 'val_data.pkl'), 'wb'))
pickle.dump(readlist, open(osp.join('/Users/zhanghaoyang/Desktop/Movie_Recsys/cache', 'train_readlist.pkl'), 'wb'))
    