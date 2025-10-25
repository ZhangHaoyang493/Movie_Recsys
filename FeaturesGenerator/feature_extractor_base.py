
from tqdm import tqdm
import hashlib
import os
import pyjson5 as json

class FeatureExtractorBase():
    def __init__(self, config: dict):
        self.ratings_path = config.get('ratings_path')
        self.movie_path = config.get('movies_path')
        self.user_path = config.get('users_path')
        self.out_basedir = config.get('out_basedir')
        self.share_slot_ids = config.get('share_slot_ids')
        self.slot_ids = config.get('slot_ids')
        self.item_slots = config.get('item_slots', [])
        self.movie_features_file_name = 'movie_features.txt'        
        embedding_idx_dict_path = config.get('embedding_idx_dict_path', None)
        

        self.movie_data_dict = {}
        self.user_data_dict = {}


        # 读取电影的数据和用户的数据，评分数据
        self.movie_data_reader()
        self.user_data_reader()

        # 将slot id对应的特征值映射为embedding表的索引
        if embedding_idx_dict_path and os.path.exists(embedding_idx_dict_path):
            with open(embedding_idx_dict_path, 'r') as f:
                self.slot_id_embedding_idx_dict = json.load(f)
        else:
            self.slot_id_embedding_idx_dict = {slot_id: [{}, 0] for slot_id in self.slot_ids}

        self.initialization()  # 对一些定制化的slot id的特征提取函数进行初始化


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
    def ratings_data_reader(self, rating_path):
        self.ratings_datas = []  # [[user_id, movie_id, rating, timestamp], ...]
        with open(rating_path, 'r', encoding='ISO-8859-1') as file:
            for line in tqdm(file, desc="Reading ratings data"):
                user_id, movie_id, rating, timestamp = line.strip().split('::')
                self.ratings_datas.append([user_id, movie_id, float(rating), int(timestamp)])
        # 将ratings_datas中的数据按照timestamp从小到大排序
        self.ratings_datas.sort(key=lambda x: x[3])


    # 调用各个特征提取函数来提取特征
    def feature_extractor(self):
        for rating_path in self.ratings_path:
            self.ratings_data_reader(rating_path)

            output_file_name = os.path.basename(rating_path).split('.')[0] + '_features.txt'
            # 如果output_file_name已经存在，则删除
            if os.path.exists(os.path.join(self.out_basedir, output_file_name)):
                os.remove(os.path.join(self.out_basedir, output_file_name))

            # 以追加的方式打开output_path文件
            self.output_file_f = open(os.path.join(self.out_basedir, output_file_name), 'a')
            # 遍历ratings的每一行数据，获取对应的ratings数据、movie数据、user数据
            for rating in tqdm(self.ratings_datas, desc="Extracting features"):
                self.extracted_feature = {}  # slot_id: feature_hash_value

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

                self.label = self.label_extractor(data_line)  # 评分作为标签

                # 将提取到的特征和标签写入到output_path中，格式为 slot_id:feature_hash_value slot_id:feature_hash_value ... \t label1 label2
                feature_line = ' '.join([f"{slot_id}:{hash_value}" for slot_id, hash_value in self.extracted_feature.items()])
                label_line = ' '.join([str(l) for l in self.label])
                self.output_file_f.write(f"{feature_line}\t{label_line}\n")

            self.output_file_f.close()

        # 将所有物料进行特征抽取，只抽取物料侧。
        # 这是因为推荐的时候需要从物料池进行推荐，打分数据中可能没有包含所有的物料
        
        # 如果output_file_name已经存在，则删除
        if os.path.exists(os.path.join(self.out_basedir, self.movie_features_file_name)):
            os.remove(os.path.join(self.out_basedir, self.movie_features_file_name))
        # 以追加的方式打开output_path文件
        self.output_file_f = open(os.path.join(self.out_basedir, self.movie_features_file_name), 'a')
        
        for movie_id, movie_info in tqdm(self.movie_data_dict.items(), desc="Extracting item features"):
            self.extracted_feature = {}  # slot_id: feature_hash_value

            # 构造一个包含movie_info的字典
            data_line = {
                'movie_info': movie_info
            }

            
            # 依次调用每个item slot id对应的特征提取函数
            for slot_id in self.item_slots:
                func_name = f"feature_extractor_{slot_id}"
                func = getattr(self, func_name)
                func(data_line)

            # 将提取到的特征和标签写入到output_path中，格式为 slot_id:feature_hash_value slot_id:feature_hash_value ... \t label
            feature_line = ' '.join([f"{slot_id}:{hash_value}" for slot_id, hash_value in self.extracted_feature.items()])
            self.output_file_f.write(f"{feature_line}\t-1\n")  # 物料侧没有标签，统一写为-1
        self.output_file_f.close()

        # 存储slot_id_embedding_idx_dict到json文件中
        with open(os.path.join(self.out_basedir, 'embedding_idx_dict.json'), 'w') as f:
            json.dump(self.slot_id_embedding_idx_dict, f)

    # 获取slot id对应的特征的真实值对应的embedding表的索引
    def get_feature_embedding_idx(self, slot_id, feature_value):
        feature_dict, current_idx = self.slot_id_embedding_idx_dict[slot_id]

        if feature_value not in feature_dict:
            feature_dict[feature_value] = current_idx
            self.slot_id_embedding_idx_dict[slot_id][1] += 1  # 更新当前的索引

        return feature_dict[feature_value]