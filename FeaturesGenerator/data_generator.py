import yaml
import os
import pyjson5 as json


class DataGenerator():
    def __init__(self, config_path):
        # 读取json配置文件
        with open(config_path, 'r') as f:
            config = json.load(f)

        # 从配置文件中获取各个路径和slot_ids
        self.ratings_path = config.get('ratings_path')
        self.movie_path = config.get('movies_path')
        self.user_path = config.get('users_path')
        self.slot_ids = config.get('slot_ids')
        self.out_basedir = config.get('out_basedir')
        self.share_slot_ids = config.get('share_slot_ids', {})

        
        # 获取feature_extractor_path中FeatureExtractor类的所有函数名称
        function_names = self.get_feature_extractor_functions(config)
        # 验证slot_ids中的每个slot id都在feature_extractor_path中
        self.validate_slot_ids(self.slot_ids, function_names)

        # 初始化FeatureExtractor
        feature_extractor_path = config.get('feature_extractor_path')
        import importlib.util
        spec = importlib.util.spec_from_file_location("feature_extractor", feature_extractor_path)
        feature_extractor_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(feature_extractor_module)
        self.feature_extractor = feature_extractor_module.FeatureExtractor(
            ratings_path=self.ratings_path,
            movie_path=self.movie_path,
            user_path=self.user_path,
            out_basedir=self.out_basedir,
            slot_ids=self.slot_ids,
            share_slot_ids=self.share_slot_ids
        )

        # 写出数据
        self.feature_extractor.feature_extractor()


    # 读取并解析yaml文件
    def read_yaml(self, file_path):
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)

    # 获取yaml文件的feature_extractor_path属性，读取其中的FeatureExtractor类的所有函数名称
    def get_feature_extractor_functions(self, config):
        feature_extractor_path = config.get('feature_extractor_path')
        if not feature_extractor_path:
            raise ValueError("feature_extractor_path not found in config")
        
        # 动态导入模块
        import importlib.util
        spec = importlib.util.spec_from_file_location("feature_extractor", feature_extractor_path)
        feature_extractor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(feature_extractor)

        # 获取FeatureExtractor类的所有函数名称
        function_names = [func for func in dir(feature_extractor.FeatureExtractor) if callable(getattr(feature_extractor.FeatureExtractor, func)) and not func.startswith("__")]
        
        return function_names
    
    # 判断slot id中的每个slot id都在feature_generator_path中定义的函数名称中
    def validate_slot_ids(self, slot_ids, function_names):
        function_names = set(function_names)  # 转换为集合以提高查找效率
        for slot_id in slot_ids: # 遍历每个slot_id
            if f"feature_extractor_{slot_id}" not in function_names: # 检查对应的函数名称是否存在
                raise ValueError(f"slot_id {slot_id} does not correspond to any function in feature_generator")


# 示例用法
if __name__ == "__main__":
    config_path = "/Users/zhanghaoyang/Desktop/Movie_Recsys/feature.json"
    data_generator = DataGenerator(config_path)