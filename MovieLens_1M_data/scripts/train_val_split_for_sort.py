
ratings = {}
with open('../ratings.dat', 'r') as f:
    for line in f:
        line = line.strip()
        user, item, score, timestamp = line.split('::')
        if user not in ratings:
            ratings[user] = []
        ratings[user].append([user, item, score, timestamp])

vals = []
for user in ratings:
    # 对于每一个user，按照timestamp从小到大排序
    ratings[user] = sorted(ratings[user], key=lambda x: int(x[-1]))
    # 选取用户最后的10个打分作为排序层的验证集
    val_this_user = ratings[user][-10:]
    ratings[user] = ratings[user][:-10]
    
    vals += val_this_user


with open('../sort_train_val_data/train_ratings_for_sort.dat', 'w') as f:
    for user in ratings:
        for data in ratings[user]:
            f.write('::'.join(data) + '\n')

with open('../sort_train_val_data/val_ratings_for_sort.dat', 'w') as f:
    for data in vals:
        f.write('::'.join(data) + '\n')