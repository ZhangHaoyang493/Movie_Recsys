import pickle


readlist = pickle.load(open('/Users/zhanghaoyang/Desktop/Movie_Recsys/cache/train_readlist.pkl', 'rb'))
item_info = pickle.load(open('/Users/zhanghaoyang/Desktop/Movie_Recsys/cache/movie_info.pkl', 'rb'))


def get_mean(arr):
    return sum(arr) / len(arr)

def get_std(arr):
    mean = get_mean(arr)
    arr = [i - mean for i in arr]
    arr = [i ** 2 for i in arr]
    return sum(arr) / len(arr)

# 获取用户偏爱的电影类型的前两名
def get_most_kinds(arr):
    kind_dic = {}
    for info in arr:
        item_kinds = item_info[info[0]][-1]
        for kind in item_kinds:
            if kind not in kind_dic:
                kind_dic[kind] = 0
            kind_dic[kind] += 1
    kind_list = []
    for kind in kind_dic:
        kind_list.append([kind, kind_dic[kind]])
    kind_list = sorted(kind_list, key=lambda x: x[-1], reverse=True)
    like_kind = kind_list[:3]
    while len(like_kind) < 3:
        like_kind.append('')
    return like_kind



# 用户活跃度字典（与用户打过分的电影数量成正相关）
user_act = {}
# 用户评分均值
user_mean_score = {}
# 用户评分方差
user_std_score = {}
# 用户偏爱的电影类型
user_like_kinds = {}

for userid in readlist.keys():
    user_act[userid] = len(readlist[userid])
    user_score = [i[1] for i in readlist[userid]]
    user_mean_score[userid] = get_mean(user_score)
    user_std_score[userid] = get_std(user_score)
    user_like_kinds[userid] = get_most_kinds(readlist[userid])

user_features = {}

for userid in readlist.keys():
    user_features[userid] = {'active': user_act[userid], 'mean_score': user_mean_score[userid],
                             'std_score': user_std_score[userid], 'like_kinds': user_like_kinds[userid]}
    
pickle.dump(user_features, open('/Users/zhanghaoyang/Desktop/Movie_Recsys/cache/user_fe.pkl', 'wb'))


