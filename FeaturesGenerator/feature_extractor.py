
from tqdm import tqdm
import hashlib
import os
import pyjson5 as json
from feature_extractor_base import FeatureExtractorBase

class FeatureExtractor(FeatureExtractorBase):
    def __init__(self, config: dict):
        super().__init__(config)

    # 对于一些定制化的slot id的特征提取函数，可以在这里对需要的特殊的类变量进行初始化
    def initialization(self):
        self.user_recent_10_movies = {}  # user_id: [movie_id1, movie_id2, ..., movie_id10]
        self.positive_rating_history = {}  # user_id: [rating1, rating2, ..., ratingN] 只保存用户的正反馈评分记录
        self.user_genre_count = {}  # user_id: {genre: count}

    def feature_extractor_1(self, data_line):  # 提取用户的id
        user_id = data_line['rating'][0]
        # 获取对应的slot_id 1的embedding索引映射字典
        embedding_idx = self.get_feature_embedding_idx(1, user_id)
        # 将hash后的字符串存储到extracted_feature中
        self.extracted_feature['1'] = embedding_idx

    def feature_extractor_2(self, data_line):  # 提取电影的id
        movie_id = data_line['movie_info'].get('movie_id', '')
        # 获取对应的slot_id 2的embedding索引映射字典
        embedding_idx = self.get_feature_embedding_idx(2, movie_id)
        # 将hash后的字符串存储到extracted_feature中
        self.extracted_feature['2'] = embedding_idx

    def feature_extractor_3(self, data_line):  # 提取用户性别
        user_gender = data_line['user_info'].get('gender', '')
        # 获取对应的slot_id 3的embedding索引映射字典
        embedding_idx = self.get_feature_embedding_idx(3, user_gender)
        # 将hash后的字符串存储到extracted_feature中
        self.extracted_feature['3'] = embedding_idx

    def feature_extractor_4(self, data_line):  # 提取用户年龄
        user_age = data_line['user_info'].get('age', '')
        # 将hash后的字符串存储到extracted_feature中
        embedding_idx = self.get_feature_embedding_idx(4, user_age)  # 将年龄除以10，10年为一个区间
        self.extracted_feature['4'] = embedding_idx

    def feature_extractor_5(self, data_line):  # 提取电影年份
        movie_year = data_line['movie_info'].get('year', '')
        # 将hash后的字符串存储到extracted_feature中
        embedding_idx = self.get_feature_embedding_idx(5, int(movie_year) // 5)  # 将年份除以5，5年为一个区间
        self.extracted_feature['5'] = embedding_idx

    def feature_extractor_6(self, data_line):  # 提取用户最近观看的10部电影
        user_id = data_line['rating'][0]
        movie_id = data_line['rating'][1]
        if user_id not in self.user_recent_10_movies:
            self.user_recent_10_movies[user_id] = []
        
        recent_movies = self.user_recent_10_movies[user_id]
        
        if len(recent_movies) > self.array_max_length.get('6', 10):
            recent_movies.pop(0)  # 保持只保存最近的10部电影
        
        recent_movies_str = ','.join(map(str, recent_movies))  # 转换为字符串存储
        self.extracted_feature['6'] = recent_movies_str

        recent_movies.append(
            self.get_feature_embedding_idx(
                self.share_slot_ids.get('6', 6), # 如果share_slot_ids中没有配置slot id 6，则使用默认的slot id 6
                movie_id
            )
        )  # 使用电影的embedding索引来表示电影

    def feature_extractor_7(self, data_line):  # 提取用户的职业
        user_occupation = data_line['user_info'].get('occupation', '')
        # 获取对应的slot_id 7的embedding索引映射字典
        embedding_idx = self.get_feature_embedding_idx(7, user_occupation)
        # 将hash后的字符串存储到extracted_feature中
        self.extracted_feature['7'] = embedding_idx


    def feature_extractor_8(self, data_line):  # 提取电影的类别
        movie_genres = data_line['movie_info'].get('genres', [])
        genres = []
        # 获取对应的slot_id 8的embedding索引映射字典
        for genre in movie_genres:
            embedding_idx = self.get_feature_embedding_idx(
                self.share_slot_ids.get('8', 8),  # 如果share_slot_ids中没有配置slot id 8，则使用默认的slot id 8
                genre
            )
            genres.append(embedding_idx)  # 将每个类别的embedding索引添加到列表中
        
        if len(genres) > self.array_max_length.get('8', 5):
            genres = genres[:self.array_max_length.get('8', 5)]  # 保持只保存最近的5个类别
        # 将hash后的字符串存储到extracted_feature中
        self.extracted_feature['8'] = ','.join(map(str, genres))

    
    def feature_extractor_9(self, data_line):  # 提取用户最近打正分的5部电影
        user_id = data_line['rating'][0]
        rating_score = data_line['rating'][2]
        movie_id = data_line['rating'][1]
        if user_id not in self.positive_rating_history:
            self.positive_rating_history[user_id] = []
        
        user_positive_history = self.positive_rating_history[user_id]
        self.extracted_feature['9'] = ','.join(map(str, user_positive_history))

        if float(rating_score) >= 4.0:
            if len(user_positive_history) == self.array_max_length.get('9', 5):
                user_positive_history.pop(0)  # 保持只保存最近的5个电影

            user_positive_history.append(
                self.get_feature_embedding_idx(
                    self.share_slot_ids.get('9', 9),
                    movie_id
                )
            )
        
    def feature_extractor_10(self, data_line):  # 提取用户历史最喜欢的电影类别
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
                self.share_slot_ids.get('10', 10),
                genre_list[0][0]
            )
        else:
            embedding_idx = self.get_feature_embedding_idx(
                self.share_slot_ids.get('10', 10),
                'UNKNOW'
            )

        self.extracted_feature['10'] = embedding_idx

    # 提取标签，返回一个列表形式
    def label_extractor(self, data_line):
        rating_score = data_line['rating'][2]
        is_like = 1 if rating_score >= 4.0 else 0
        return [rating_score, is_like]  # 评分作为标签，返回一个列表形式