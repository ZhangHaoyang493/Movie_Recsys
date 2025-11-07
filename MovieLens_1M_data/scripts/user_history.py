# 生成用户的评分历史，用作后续评估召回层和排序层消重
import os
import json
from tqdm import tqdm

with open('../train_ratings.dat', 'r') as f:
    user_ratings = {}
    for line in tqdm(f):
        line = line.strip()
        user, item, score, timestamp = line.split('::')
        if user not in user_ratings:
            user_ratings[user] = {}
        user_ratings[user][item] = [score, timestamp]  # 存储评分和时间戳
# 将用户评分历史写入文件(json格式)
output_file = '../user_history.json'
with open(output_file, 'w') as f:
    json.dump(user_ratings, f)