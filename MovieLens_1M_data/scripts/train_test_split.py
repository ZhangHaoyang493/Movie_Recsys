
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
    # 最后一个正反馈用作验证
    last_pos_idx = len(ratings[user]) - 1
    while last_pos_idx >= 0 and float(ratings[user][last_pos_idx][2]) < 4:
        last_pos_idx -= 1
    if last_pos_idx == -1:
        continue
    vals.append(ratings[user][last_pos_idx])
    ratings[user] = ratings[user][:last_pos_idx]


with open('../train_ratings.dat', 'w') as f:
    for user in ratings:
        for data in ratings[user]:
            f.write('::'.join(data) + '\n')

with open('../test_ratings.dat', 'w') as f:
    for data in vals:
        f.write('::'.join(data) + '\n')