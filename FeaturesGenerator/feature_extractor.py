
from tqdm import tqdm
import hashlib
import os
import pyjson5 as json
from feature_extractor_base import FeatureExtractorBase

class FeatureExtractor(FeatureExtractorBase):
    def __init__(self, config: dict):
        super().__init__(config)

    # 对于一些定制化的feature name的特征提取函数，可以在这里对需要的特殊的类变量进行初始化
    def initialization(self):
        self.user_recent_10_movies = {}  # user_id: [movie_id1, movie_id2, ..., movie_id10]
        self.positive_rating_history = {}  # user_id: [rating1, rating2, ..., ratingN] 只保存用户的正反馈评分记录
        self.user_genre_count = {}  # user_id: {genre: count}

    def feature_extractor_user_id(self, data_line):  # 提取用户的id
        user_id = data_line['rating'][0]
        # 获取对应的feature_name 'user_id'的embedding索引映射字典
        embedding_idx = self.get_feature_embedding_idx('user_id', user_id)
        # 将hash后的字符串存储到extracted_feature中
        self.extracted_feature['user_id'] = embedding_idx

    def feature_extractor_movie_id(self, data_line):  # 提取电影的id
        movie_id = data_line['movie_info'].get('movie_id', '')
        # 获取对应的feature_name 'movie_id'的embedding索引映射字典
        embedding_idx = self.get_feature_embedding_idx('movie_id', movie_id)
        # 将hash后的字符串存储到extracted_feature中
        self.extracted_feature['movie_id'] = embedding_idx

    def feature_extractor_user_gender(self, data_line):  # 提取用户性别
        user_gender = data_line['user_info'].get('gender', '')
        # 获取对应的feature_name 'user_gender'的embedding索引映射字典
        embedding_idx = self.get_feature_embedding_idx('user_gender', user_gender)
        # 将hash后的字符串存储到extracted_feature中
        self.extracted_feature['user_gender'] = embedding_idx

    def feature_extractor_user_age(self, data_line):  # 提取用户年龄
        user_age = data_line['user_info'].get('age', '')
        # 将hash后的字符串存储到extracted_feature中
        embedding_idx = self.get_feature_embedding_idx('user_age', user_age)  # 将年龄除以10，10年为一个区间
        self.extracted_feature['user_age'] = embedding_idx

    def feature_extractor_movie_year(self, data_line):  # 提取电影年份
        movie_year = data_line['movie_info'].get('year', '')
        # 将hash后的字符串存储到extracted_feature中
        embedding_idx = self.get_feature_embedding_idx('movie_year', int(movie_year) // 5)  # 将年份除以5，5年为一个区间
        self.extracted_feature['movie_year'] = embedding_idx

    def feature_extractor_user_recent_10_movies(self, data_line):  # 提取用户最近观看的10部电影
        user_id = data_line['rating'][0]
        movie_id = data_line['rating'][1]
        if user_id not in self.user_recent_10_movies:
            self.user_recent_10_movies[user_id] = []
        
        recent_movies = self.user_recent_10_movies[user_id]

        if len(recent_movies) > self.array_max_length.get('user_recent_10_movies', 10):
            recent_movies.pop(0)  # 保持只保存最近的10部电影
        
        recent_movies_str = ','.join(map(str, recent_movies))  # 转换为字符串存储
        self.extracted_feature['user_recent_10_movies'] = recent_movies_str

        recent_movies.append(
            self.get_feature_embedding_idx(
                self.share_emb_table_features.get('user_recent_10_movies', 'user_recent_10_movies'), # 如果share_emb_table_features中没有配置user_recent_10_movies的共享feature，则使用默认的feature name
                movie_id
            )
        )  # 使用电影的embedding索引来表示电影

    def feature_extractor_user_occupation(self, data_line):  # 提取用户的职业
        user_occupation = data_line['user_info'].get('occupation', '')
        # 获取对应的feature_name 'user_occupation'的embedding索引映射字典
        embedding_idx = self.get_feature_embedding_idx('user_occupation', user_occupation)
        # 将hash后的字符串存储到extracted_feature中
        self.extracted_feature['user_occupation'] = embedding_idx


    def feature_extractor_movie_genres(self, data_line):  # 提取电影的类别
        movie_genres = data_line['movie_info'].get('genres', [])
        genres = []
        # 获取对应的feature name的embedding索引映射字典
        for genre in movie_genres:
            embedding_idx = self.get_feature_embedding_idx(
                self.share_emb_table_features.get('movie_genres', 'movie_genres'),  # 如果share_emb_table_features中没有配置movie_genres的共享feature name，则使用默认的feature name
                genre
            )
            genres.append(embedding_idx)  # 将每个类别的embedding索引添加到列表中

        if len(genres) > self.array_max_length.get('movie_genres', 5):
            genres = genres[:self.array_max_length.get('movie_genres', 5)]  # 保持只保存最近的5个类别
        # 将hash后的字符串存储到extracted_feature中
        self.extracted_feature['movie_genres'] = ','.join(map(str, genres))


    def feature_extractor_user_recent_positive_score_10_movie(self, data_line):  # 提取用户最近打正分的10部电影
        user_id = data_line['rating'][0]
        rating_score = data_line['rating'][2]
        movie_id = data_line['rating'][1]
        if user_id not in self.positive_rating_history:
            self.positive_rating_history[user_id] = []
        
        user_positive_history = self.positive_rating_history[user_id]
        self.extracted_feature['user_recent_positive_score_10_movie'] = ','.join(map(str, user_positive_history))

        if float(rating_score) >= 4.0:
            if len(user_positive_history) == self.array_max_length.get('user_recent_positive_score_10_movie', 10):
                user_positive_history.pop(0)  # 保持只保存最近的5个电影

            user_positive_history.append(
                self.get_feature_embedding_idx(
                    self.share_emb_table_features.get('user_recent_positive_score_10_movie', 'user_recent_positive_score_10_movie'),
                    movie_id
                )
            )

    def feature_extractor_user_history_favourite_genre(self, data_line):  # 提取用户历史最喜欢的电影类别
        user_id = data_line['rating'][0]
        movie_genres = data_line['movie_info'].get('genres', [])
        if user_id not in self.user_genre_count:
            self.user_genre_count[user_id] = {}
        
        genre_count = self.user_genre_count[user_id]
        for genre in movie_genres:
            genre_count[genre] = genre_count.get(genre, 0) + 1
        
        genre_list = [[k, v] for k, v in genre_count.items()]
        genre_list = sorted(genre_list, key=lambda x: x[1], reverse=True)  # 按照count排序
        if len(genre_list) >= 1:
            embedding_idx = self.get_feature_embedding_idx(
                self.share_emb_table_features.get('user_history_favourite_genre', 'user_history_favourite_genre'),
                genre_list[0][0]
            )
        else:
            embedding_idx = self.get_feature_embedding_idx(
                self.share_emb_table_features.get('user_history_favourite_genre', 'user_history_favourite_genre'),
                'UNKNOW'
            )

        self.extracted_feature['user_history_favourite_genre'] = embedding_idx

    # 提取标签，返回一个列表形式
    def label_extractor(self, data_line):
        rating_score = data_line['rating'][2]
        is_like = 1 if rating_score >= 4.0 else 0
        return [rating_score, is_like]  # 评分作为标签，返回一个列表形式