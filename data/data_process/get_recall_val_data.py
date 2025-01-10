import os
import os.path as osp
import pickle
import yaml
import random

# 读取readlist，然后选取一部分的用户作为验证组，将他们打分为正向的最后一次记录拿出来作为验证数据
# 然后这一次后面的数据要丢弃
conf = yaml.safe_load(open('../../global_conf.yaml', 'r'))

readlist = pickle.load(open(osp.join(conf['basedir'], 'cache', 'readlist.pkl'), 'rb'))
userid = readlist.keys()
val_userid = random.sample(userid, 500)

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
    # 如果用户只有第一次为正向打分或者根本就没有正向打分，就不考虑这个用户
    if index <= 0:
        continue

    val_data[v_userid] = v_user_readlist[index]
    readlist[v_userid] = v_user_readlist[:index]

pickle.dump(val_data, open(osp.join(conf['basedir'], 'cache', 'val_data.pkl'), 'wb'))
pickle.dump(readlist, open(osp.join(conf['basedir'], 'cache', 'train_readlist.pkl'), 'wb'))
    