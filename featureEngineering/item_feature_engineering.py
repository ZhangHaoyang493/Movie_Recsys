import sys



class ItemFeatureEngineering:
    def __init__(self):
        self.rating_dict = {}
        self.user_dict = {}
        self.movie_feature_dict = {}
        self.movie_dict = {}

        self.write_user_dict()
        self.write_movie_dict()
        self.write_rating_dict()

        self.write_base_item_feature()
        self.write_movie_hot_degree()
        self.write_movie_avg_rating()

    
    def write_rating_dict(self):
        with open('../data/train_ratings.dat', 'r') as f:
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

    def write_base_item_feature(self):
        for movieid in self.movie_dict:
            self.movie_feature_dict[movieid] = self.movie_dict[movieid]
        
    
    # 计算电影热度，也就是被多少用户看过
    def write_movie_hot_degree(self):
        movie_hot_degree = {}
        for userid in self.rating_dict:
            for rating_info in self.rating_dict[userid]:
                movieid = rating_info[0]
                if movieid not in movie_hot_degree:
                    movie_hot_degree[movieid] = 1
                else:
                    movie_hot_degree[movieid] += 1
        for movieid in movie_hot_degree:
            self.movie_feature_dict[movieid].append(movie_hot_degree[movieid])

    def write_movie_avg_rating(self):
        movie_score_info = {}
        for userid in self.rating_dict:
            for rating_info in self.rating_dict[userid]:
                movieid, score = rating_info[0], rating_info[1]
                if movieid not in movie_score_info:
                    movie_score_info[movieid] = [score, 1]
                else:
                    movie_score_info[movieid][0] += score
                    movie_score_info[movieid][1] += 1
        for movieid in movie_score_info:
            avg_score = movie_score_info[movieid][0] / movie_score_info[movieid][1]
            self.movie_feature_dict[movieid].append(avg_score)
    
        
    def save_user_feature_to_csv(self, save_path):
        with open(save_path, 'w') as f:
            for movieid in self.movie_feature_dict:
                line = [movieid]
                for fea in self.movie_feature_dict[movieid]:
                    line.append(str(fea))
                f.write('::'.join(line) + '\n')

    

if __name__ == '__main__':
    item_fea_engineer = ItemFeatureEngineering()
    item_fea_engineer.save_user_feature_to_csv('../data/item_feature.dat')