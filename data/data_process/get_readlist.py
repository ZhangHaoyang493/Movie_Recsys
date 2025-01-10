import os
import pickle
import os.path as osp

basedir = '/Users/zhanghaoyang04/Desktop/Movie_Recsys/'

readlist = {}

with open(osp.join(basedir, 'data', 'ratings.dat'), 'r') as f:
    for line in f:
        line = line.strip()
        data = line.split('::')
        # UserID::MovieID::Rating::Timestamp
        userid = data[0]
        if userid not in readlist:
            readlist[userid] = []
        readlist[userid].append((data[1], float(data[2]), int(data[3])))

for userid in readlist.keys():
    k_list = readlist[userid]
    # 按照timestamp从小到大排序
    k_list = sorted(k_list, key=lambda x: x[-1])
    readlist[userid] = k_list

pickle.dump(readlist, open(os.path.join(basedir, 'cache', 'readlist.pkl'), 'wb'))