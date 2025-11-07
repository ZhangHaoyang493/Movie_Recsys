
from tqdm import tqdm
import hashlib
import os
import pyjson5 as json
from feature_extractor_base import FeatureExtractorBase

class FeatureExtractor(FeatureExtractorBase):
    def __init__(self, config: dict):
        super().__init__(config)

    # 对于一些定制化的feature的特征提取函数，可以在这里对需要的特殊的类变量进行初始化
    def initialization(self):
        pass

    def feature_extractor_xxx(self, data_line):
        pass

    # 提取标签，返回一个列表形式
    def label_extractor(self, data_line):
        pass