
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

    def feature_extractor_4(self, data_line):  # 提取用户年龄，数值特征不用哈希
        user_age = data_line['user_info'].get('age', '')
        # 将hash后的字符串存储到extracted_feature中
        self.extracted_feature['4'] = user_age

    def feature_extractor_5(self, data_line):  # 提取电影年份，数值特征不用哈希
        movie_year = data_line['movie_info'].get('year', '')
        # 将hash后的字符串存储到extracted_feature中
        self.extracted_feature['5'] = movie_year

    def feature_extractor_6(self, data_line):  # 提取用户最近观看的10部电影
        user_id = data_line['rating'][0]
        movie_id = data_line['rating'][1]
        if user_id not in self.user_recent_10_movies:
            self.user_recent_10_movies[user_id] = []
        
        recent_movies = self.user_recent_10_movies[user_id]
        
        if len(recent_movies) > 10:
            recent_movies.pop(0)  # 保持只保存最近的10部电影
        
        recent_movies_str = ','.join(map(str, recent_movies))  # 转换为字符串存储
        self.extracted_feature['6'] = recent_movies_str

        recent_movies.append(
            self.get_feature_embedding_idx(
                self.share_slot_ids.get('6', 6), # 如果share_slot_ids中没有配置slot id 6，则使用默认的slot id 6
                movie_id
            )
        )  # 使用电影的embedding索引来表示电影

    # 提取标签，返回一个列表形式
    def label_extractor(self, data_line):
        rating_score = data_line['rating'][2]
        is_like = 1 if rating_score >= 4.0 else 0
        return [rating_score, is_like]  # 评分作为标签，返回一个列表形式