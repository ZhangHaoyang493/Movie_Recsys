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

        self.write_base_user_feature()
        self.write_user_avg_score()
        self.write_user_std_score()
        self.write_user_his_favourite_movie_kind()
    
    def write_rating_dict(self):
        with open('../data/ratings.dat', 'r') as f:
            for line in f:
                line = line.strip()
                userid, movieid, score, timestamp = line.split('::')
                if userid not in self.rating_dict:
                    self.rating_dict[userid] = []
                self.rating_dict[userid].append([movieid, float(score), int(timestamp)])
    
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
            user_scores = [i[1] for i in self.rating_dict[userid]]
            avg_score = sum(user_scores) / len(user_scores)
            self.user_feature_dict[userid].append(avg_score)
    
    # 将用户历史的打分的方差加入用户的特征中
    def write_user_std_score(self):
        for userid in self.user_feature_dict:
            user_scores = [i[1] for i in self.rating_dict[userid]]
            avg_score = sum(user_scores) / len(user_scores)
            user_scores = [i - avg_score for i in user_scores]
            user_scores = [i**2 for i in user_scores]
            std_score = (sum(user_scores) / len(user_scores)) ** 0.5
            self.user_feature_dict[userid].append(std_score)

    # 获取用户最喜欢的前5个
    def write_user_his_favourite_movie_kind(self):
        for userid in self.user_feature_dict:
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