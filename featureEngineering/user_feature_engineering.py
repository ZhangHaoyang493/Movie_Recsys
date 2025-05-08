import sys



class UserFeatureEngineering:
    def __init__(self):
        self.rating_dict = {}
        self.user_dict = {}
        self.user_feature_dict = {}
        self.movie_dict = {}

        self.write_user_dict()
        self.write_movie_dict()
        self.write_rating_dict()

        # 写入用户特征
        self.write_base_user_feature()
        self.write_user_avg_score()
        self.write_user_std_score()
        self.write_user_his_favourite_movie_kind()
        self.write_user_read_history_5_itemid()
        self.write_user_read_history_like_3_itemid()
    
    def write_rating_dict(self):
        with open('../data/train_ratings.dat', 'r') as f:
            for line in f:
                line = line.strip()
                userid, movieid, score, timestamp = line.split('::')
                if userid not in self.rating_dict:
                    self.rating_dict[userid] = []
                self.rating_dict[userid].append([movieid, float(score), int(timestamp)])
                self.rating_dict[userid] = sorted(self.rating_dict[userid], key=lambda x: x[-1]) # 将用户的打分历史按照timestamp从小到大排序
    
    def write_user_dict(self):
        with open('../data/users.dat', 'r') as f:
            for line in f:
                line = line.strip()
                userid, gender, age, occupation, _ = line.split('::')
                self.user_dict[userid] = [gender, int(age), occupation]

    def write_movie_dict(self):
        with open('../data/movies.dat', 'r', encoding='ISO-8859-1') as f:
            for line in f:
                line = line.strip()
                movieid, name, kinds = line.split('::')
                self.movie_dict[movieid] = [movieid, name, kinds.split('|')]

    def write_base_user_feature(self):
        for userid in self.user_dict:
            self.user_feature_dict[userid] = self.user_dict[userid]
    
    # 将用户历史的平均打分加入用户的特征中
    def write_user_avg_score(self):
        for userid in self.user_feature_dict:
            # 如果userid从没有出现在打分历史中
            if userid in self.rating_dict:
                user_scores = [i[1] for i in self.rating_dict[userid]]
                avg_score = sum(user_scores) / len(user_scores)
                self.user_feature_dict[userid].append(avg_score)
            # 否则的话，认为这个用户的平均打分就是2.5
            else:
                self.user_feature_dict[userid].append(2.5)
    
    # 将用户历史的打分的方差加入用户的特征中
    def write_user_std_score(self):
        for userid in self.user_feature_dict:
            if userid in self.rating_dict:
                user_scores = [i[1] for i in self.rating_dict[userid]]
                avg_score = sum(user_scores) / len(user_scores)
                user_scores = [i - avg_score for i in user_scores]
                user_scores = [i**2 for i in user_scores]
                std_score = (sum(user_scores) / len(user_scores)) ** 0.5
                self.user_feature_dict[userid].append(std_score)
            else:
                self.user_feature_dict[userid].append(0.0)

    # 获取用户最喜欢的前5个电影的类型
    def write_user_his_favourite_movie_kind(self):
        for userid in self.user_feature_dict:
            if userid in self.rating_dict:
                kind_dict = {}
                for rating_info in self.rating_dict[userid]:
                    movieid = rating_info[0]
                    kinds = self.movie_dict[movieid][-1]
                    for k in kinds:
                        if k in kind_dict:
                            kind_dict[k] += 1
                        else:
                            kind_dict[k] = 1
                kind_arr = list(kind_dict.items())
                kind_arr = sorted(kind_arr, key=lambda x: x[1], reverse=True)
                kind_arr = kind_arr[:5]
                kind_arr = [i[0] for i in kind_arr]
                self.user_feature_dict[userid].append(kind_arr)
            else:
                # 如果userid不曾出现在打分历史中，就把热度最高的三个类型给这个用户
                self.user_feature_dict[userid].append(['Comedy', 'Drama', 'Action'])
    
    # 用户最近的5个打分过的item id
    def write_user_read_history_5_itemid(self):
        for userid in self.user_feature_dict:
            if userid in self.rating_dict:
                read_history = self.rating_dict[userid][-5:]
                read_history = [i[0] for i in read_history]
                self.user_feature_dict[userid].append(read_history)
            else:
                self.user_feature_dict[userid].append([])

    # 用户最近的3个打正分的item id
    def write_user_read_history_like_3_itemid(self):
        for userid in self.user_feature_dict:
            if userid in self.rating_dict:
                read_history = self.rating_dict[userid]
                like_read_history = []
                for i in range(len(read_history)):
                    if read_history[i][1] >= 4.0:
                        like_read_history.append(read_history[i][0])
                like_read_history = like_read_history[-3:]
                self.user_feature_dict[userid].append(like_read_history)
            else:
                self.user_feature_dict[userid].append([])


    def save_user_feature_to_csv(self, save_path):
        with open(save_path, 'w') as f:
            for userid in self.user_feature_dict:
                line = [userid]
                for fea in self.user_feature_dict[userid]:
                    line.append(str(fea))
                f.write('::'.join(line) + '\n')
    

if __name__ == '__main__':
    user_fea_engineer = UserFeatureEngineering()
    user_fea_engineer.save_user_feature_to_csv('../data/user_feature.dat')