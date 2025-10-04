
from tqdm import tqdm
import hashlib
import os
import pyjson5 as json

class FeatureExtractor():
    def __init__(self, ratings_path: str, movie_path: str, user_path: str, output_path: str, slot_ids: list, share_slot_ids: dict, embedding_idx_dict_path: str = None):
        self.ratings_path = ratings_path
        self.movie_path = movie_path
        self.user_path = user_path
        self.output_path = output_path
        self.share_slot_ids = share_slot_ids

        if os.path.exists(self.output_path):
            os.remove(self.output_path)

        self.movie_data_dict = {}
        self.user_data_dict = {}
        self.ratings_datas = []

        self.slot_ids = slot_ids

        # 读取电影的数据和用户的数据，评分数据
        self.movie_data_reader()
        self.user_data_reader()
        self.ratings_data_reader()

        # 将slot id对应的特征值映射为embedding表的索引
        if embedding_idx_dict_path and os.path.exists(embedding_idx_dict_path):
            with open(embedding_idx_dict_path, 'r') as f:
                self.slot_id_embedding_idx_dict = json.load(f)
        else:
            self.slot_id_embedding_idx_dict = {slot_id: [{}, 0] for slot_id in self.slot_ids}

        # 生成向output_path中写数据的句柄，需要不断的往文件最后面追加数据
        self.output_file_f = open(self.output_path, 'a')

        self.initialization()  # 对一些定制化的slot id的特征提取函数进行初始化

    # 对于一些定制化的slot id的特征提取函数，可以在这里对需要的特殊的类变量进行初始化
    def initialization(self):
        self.user_recent_10_movies = {}  # user_id: [movie_id1, movie_id2, ..., movie_id10]


    # 读取movie数据, 1::Toy Story (1995)::Animation|Children's|Comedy
    def movie_data_reader(self):
        with open(self.movie_path, 'r', encoding='ISO-8859-1') as file:
            for line in tqdm(file, desc="Reading movie data"):
                # 解析每一行数据
                movie_id, title, genres = line.strip().split('::')
                #  从title中获取year
                year = title.split('(')[-1].strip(')') if '(' in title else None
                # 从title中获取title
                title = title if '(' not in title else title.split('(')[0].strip()
                # 进一步处理数据
                self.movie_data_dict[movie_id] = {
                    'movie_id': movie_id,
                    'title': title,
                    'year': year,
                    'genres': genres.split('|')
                }

    # 读取user数据, 1::F::1::10::48067
    def user_data_reader(self):
        with open(self.user_path, 'r', encoding='ISO-8859-1') as file:
            for line in tqdm(file, desc="Reading user data"):
                user_id, gender, age, occupation, zip_code = line.strip().split('::')
                self.user_data_dict[user_id] = {
                    'user_id': user_id,
                    'gender': gender,
                    'age': age,
                    'occupation': occupation,
                    'zip_code': zip_code
                }
    
    # 读取ratings数据，1::1193::5::978300760
    def ratings_data_reader(self):
        with open(self.ratings_path, 'r', encoding='ISO-8859-1') as file:
            for line in tqdm(file, desc="Reading ratings data"):
                user_id, movie_id, rating, timestamp = line.strip().split('::')
                self.ratings_datas.append([user_id, movie_id, float(rating), int(timestamp)])
        # 将ratings_datas中的数据按照timestamp从小到大排序
        self.ratings_datas.sort(key=lambda x: x[3])


    # 调用各个特征提取函数来提取特征
    def feature_extractor(self):
        # 遍历ratings的每一行数据，获取对应的ratings数据、movie数据、user数据
        for rating in tqdm(self.ratings_datas, desc="Extracting features"):
            self.extracted_feature = {}  # slot_id: feature_hash_value
            self.label = rating[2]  # 评分作为标签

            # 获取对应的movie数据和user数据
            user_id, movie_id = rating[0], rating[1]
            movie_info = self.movie_data_dict.get(movie_id, {})
            user_info = self.user_data_dict.get(user_id, {})
            # 构造一个包含rating, movie_info, user_info的字典
            data_line = {
                'rating': rating,
                'movie_info': movie_info,
                'user_info': user_info
            }

            # 依次调用每个slot id对应的特征提取函数
            for slot_id in self.slot_ids:
                func_name = f"feature_extractor_{slot_id}"
                func = getattr(self, func_name)
                func(data_line)

            # 将提取到的特征和标签写入到output_path中，格式为 slot_id:feature_hash_value slot_id:feature_hash_value ... \t label
            feature_line = ' '.join([f"{slot_id}:{hash_value}" for slot_id, hash_value in self.extracted_feature.items()])
            self.output_file_f.write(f"{feature_line}\t{self.label}\n")
        
        # 存储slot_id_embedding_idx_dict到当前目录
        with open('embedding_idx_dict.json', 'w') as f:
            json.dump(self.slot_id_embedding_idx_dict, f)

        self.output_file_f.close()

    # 获取slot id对应的特征的真实值对应的embedding表的索引
    def get_feature_embedding_idx(self, slot_id, feature_value):
        feature_dict, current_idx = self.slot_id_embedding_idx_dict[slot_id]

        if feature_value not in feature_dict:
            feature_dict[feature_value] = current_idx
            self.slot_id_embedding_idx_dict[slot_id][1] += 1  # 更新当前的索引

        return feature_dict[feature_value]


    def feature_extractor_1(self, data_line):  # 提取用户的id
        user_id = data_line['rating'][0]
        # 获取对应的slot_id 1的embedding索引映射字典
        embedding_idx = self.get_feature_embedding_idx(1, user_id)
        # 将hash后的字符串存储到extracted_feature中
        self.extracted_feature['1'] = embedding_idx

    def feature_extractor_2(self, data_line):  # 提取电影的id
        movie_id = data_line['rating'][1]
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
        recent_movies.append(
            self.get_feature_embedding_idx(
                self.share_slot_ids.get('6', 6), # 如果share_slot_ids中没有配置slot id 6，则使用默认的slot id 6
                movie_id
            )
        )  # 使用电影的embedding索引来表示电影
        if len(recent_movies) > 10:
            recent_movies.pop(0)  # 保持只保存最近的10部电影
        
        recent_movies_str = ','.join(map(str, recent_movies))  # 转换为字符串存储
        self.extracted_feature['6'] = recent_movies_str