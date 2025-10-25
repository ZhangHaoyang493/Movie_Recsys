# 🔥Movie_Recsys概述
这是项目我在学习推荐算法道路上对学习到的一些理论的实践，基于MovieLens 1M数据集，属于一个玩具项目。项目虽然很简单，
但是在这个项目中，我实现了一个包含召回层、排序层在内的一个完整的推荐系统，同时也涉及到了特征工程、模型评估等一系列为了提升推荐系统的性能所必备的操作。
本项目的深度学习模型全部使用Pytorch进行构建，方便像我一样在学习推荐系统之前只会Pytorch的同学。

### 🔥🔥🔥正在更新
### 👏欢迎大家参与本项目💪，多Star🌟，大家的Star是我最大的动力🌟



## 📈数据集介绍
MovieLens数据集1M（https://grouplens.org/datasets/movielens/）

该数据集包含大约6000个用户对4000部左右的电影的100万条评分数据。每条评分数据包含用户ID、电影ID、评分（1-5分）和时间戳。此外，数据集中还包含电影的元信息，如标题、类型等，以及用户的元信息，如年龄、性别、职业等。

数据集文件说明：
- ratings.dat：用户对电影的评分数据，格式为 user_id::movie_id::rating::timestamp
- movies.dat：电影的元信息，格式为 movie_id::title::genres
- users.dat：用户的元信息，格式为 user_id::age::gender::occupation::zip_code
- README.txt：数据集的说明文件

详细的字段说明可以参考数据集中的README.txt文件。

数据集保存在该项目的MovieLens_1M_data文件夹中。

### 关于MovieLens_1M_data文件夹内容介绍

#### 基本文件
|文件名|说明|
|-------|-------|
|ratings.dat|用户对电影的评分数据，格式为 user_id::movie_id::rating::timestamp|
|movies.dat|电影的元信息，格式为 movie_id::title::genres|
|users.dat|用户的元信息，格式为 user_id::age::gender::occupation::zip_code|
|README.txt|数据集的说明文件|
|train_test_split.ipynb|用于将原始数据集划分为训练集和测试集的Jupyter Notebook文件，划分思路为选定一个时间戳节点，然后将该时间戳之前的数据作为训练集，之后的数据作为测试集。|
|train_ratings.dat|划分好的训练集的评分数据，格式同ratings.dat，包含用户对电影的评分数据。|
|test_ratings.dat|划分好的测试集的评分数据，格式同ratings.dat，包含用户对电影的评分数据。|

#### 使用LLM对数据集做了一些增强的相关文件
##### movie_info_generation.py
使用LLM，根据movies.dat中的电影标题，生成每部电影的简介、导演、主演等信息，LLM以json的格式返回，json的格式如下：
```json
{
    "English_title": "电影英文名称",
    "Chinese_title": "电影中文名称",
    "Chinese_description": "电影中文简介",
    "tags": ["标签1", "标签2"],
    "director_chinese_name": "导演中文名称",
    "direction_English_name": "导演英文名称",
    "cast_English_name": ["主演英文名称1", "主演英文名称2"],
    "cast_Chinese_name": ["主演中文名称1", "主演中文名称2"],
    "country": "制片国家/地区",
    "language": "语言",
    "release_year": "上映时间",
    "duration": "片长"
}
```

##### movie_details.json
使用movie_info_generation.py生成的电影详情信息，包含每部电影的简介、导演、主演等信息，存储为JSON格式的文件，方便后续使用。

Example:
```json
{
    "1": {
        "1": {
        "English_title": "Toy Story (1995)",
        "Chinese_title": "玩具总动员",
        "Chinese_description": "《玩具总动员》是一部由皮克斯动画工作室制作、迪士尼发行的动画电影，讲述了玩具们在主人离开后拥有的神奇生命。影片围绕小主人安迪最喜欢的牛仔胡迪展开，他原本是安迪最喜爱的玩具，但随着新玩具——一位太空骑警巴斯光年加入后，胡迪开始担心自己会被取代。在一次意外中，胡迪和巴斯光年被安迪的妹妹带离了家，他们必须携手合作，才能找回回家的路。途中，玩具们经历了各种冒险，彼此之间也逐渐建立了深厚的友谊。影片通过幽默和感人的情节，探讨了友情、忠诚和成长的主题。",
        "tags": [
            "动画",
            "冒险",
            "喜剧",
            "家庭"
        ],
        "director_chinese_name": "约翰·拉塞特",
        "direction_English_name": "John Lasseter",
        "cast_English_name": [
            "Tom Hanks",
            "Tim Allen",
            "Don Rickles",
            "Jim Varney",
            "Wallace Shawn"
        ],
        "cast_Chinese_name": [
            "汤姆·汉克斯",
            "提姆·艾伦",
            "唐·里克斯",
            "吉姆·瓦尼",
            "沃尔特·肖恩"
        ],
        "country": "美国",
        "language": "英语",
        "release_year": "1995",
        "duration": 81
    },
    ...
}
```

## 特征框架

## 召回层
### Baseline：基于物品的协同过滤（ItemCF）

ItemCF的各个版本的HitRate@50表现如下：
|文件路径|改进点|HitRate@50|
|-------|-------|------|
|`recall/ItemCF/itemCF_base.py`|最简单的ItemCF|14.9418%|
|`recall/ItemCF/itemCF_version1.py`|引入评分权重，评分>=3.0的权重为1，评分<3.0的权重为0.5|15.8907%|
|`recall/ItemCF/itemCF_version2.py`|更精细的评分权重，评分=5.0的权重为1，评分=4.0的权重为0.8，...，评分=1.0的权重为0.2|16.9662%|


# 项目介绍
# 召回层
召回层的实现在`recall`文件夹中，`recall`文件夹的结构如下：
```python
recall
├── CF # 协同过滤
│   ├── cf_conf.yaml        # 协同过滤配置文件
│   └── item_cf.py          # 实现基于物品的协同过滤推荐逻辑，根据历史行为推荐相似电影
├── DSSM # 双塔召回模型
│   ├── DSSM_recall.py      # 基于已训练的 DSSM 模型进行向量相似度召回候选集
│   ├── DSSM_train.py       # 使用双塔网络训练用户与物品的向量表示（embedding）
│   └── dataloader.py       # 加载训练/测试数据、用户行为序列、物品特征等
├── baseRecall.py           # 定义所有召回模型的抽象基类，包含统一的接口和结构
├── deepwalk_emb_recall # 基于DeepWalk的图嵌入召回
│   ├── deepwalk_conf.yaml  # DeepWalk 配置文件，定义游走策略和嵌入维度等参数
│   ├── deepwalk_model.pth  # 训练得到的 DeepWalk 嵌入模型（保存的权重）
│   ├── deepwalk_recall_icf.py  # 使用物品向量进行召回（ItemCF模式）
│   ├── deepwalk_recall_ucf.py  # 使用用户向量进行召回（UserCF模式）
│   └── deepwalk_train.py   # 基于用户-物品图结构训练 DeepWalk 嵌入向量
├── word2vec_emb_recall # 基于Word2Vec序列嵌入召回
│   ├── w2v_conf.yaml       # Word2Vec 模型训练配置文件
│   ├── word2vec_model.pth  # 保存训练后的物品向量模型
│   ├── word2vec_recall.py  # 利用 Word2Vec 向量计算相似电影进行召回
│   └── word2vec_train.py   # 使用用户观看序列训练 Word2Vec 模型
└── youtubeDNN # YouTube DNN召回（开发中）
    ├── dataloader.py       # 为 YouTubeDNN 模型准备训练/测试数据
    └── youtubeDNN.py       # 实现 YouTube DNN 双塔模型的结构与召回逻辑
```

整体来说，召回层实现了以下几个常用的召回方法及召回模型：
+ ✅ 协同过滤（CF）：协同过滤基于物品进行
+ ✅ 矩阵分解（MF）：可以理解为协同过滤的改进
+ ✅ 双塔模型（DSSM）：推荐领域经典的双塔模型
+ ✅ 基于Embedding的召回：首先使用流行的训练Embedding的方法训练出物品侧和用户侧的Embedding，然后将物品侧的Embedding存到faiss向量数据库中，召回时查询用户Embedding然后去faiss中查找topK最近邻向量。这里训练Embedding的方法包括两种：
    + ✅ word2vec方式
    + ✅ deepwalk方式
+ youtubeDNN：youtube提出的召回模型，目前正在开发...

## 召回层模型性能分析
|方法|HitRate@10|HitRate@20|HitRate@30|HitRate@40|HitRate@50|
|--------|------|------|------------|----------|----------|
|协同过滤（CF）|1.58%|3.30%|4.99%|6.68%|8.21%|
# 精排层
精排的模型实现在`sort`中，文件夹结构如下：
```python
sort
├── DCNv1.py  #  DCN v1模型
├── DCNv2.py  # DCN v2模型
├── DIN.py # 阿里DIN模型
├── Deep.py  
├── FM.py  # FM模型
├── LR.py # 逻辑回归
├── MMoE.py # MMOE模型
├── PLE.py  # 腾讯PLE模型
├── Wide_deep.py  # Wide&Deep模型
├── deepFM.py # deepFM模型
├── sortDataLoader.py  # 训练数据读取
└── trainer.py  # 训练器
```

整体来说，精排层实现了以下几个模型：
+ ✅ LR：逻辑回归模型
+ ✅ FM模型
+ ✅ Deep模型：Embedding+MLP
+ ✅ Wide&Deep：LR+Deep模型
+ ✅ DeepFM：FM+Deep模型
+ ✅ DCNv1&DCNv2：深度交叉网络+Deep模型
+ ✅ DIN：深度兴趣模型
+ ✅ MMoE：混合专家模型
+ ✅ PLE：渐进式分层提取

# 展望
+ ⏰ 特征工程的完善，加入到模型的训练：
    + 用户历史评分均值
    + 用户历史评分方差
    + 用户的活跃度（历史评分的物品个数）
    + 用户偏爱的电影类型前三名
    + 物品的热度
    + 物品历史得分均值
    + ...
+ ⏰ youtubeDNN模型的开发
+ ⏰ 召回层代码的重构，召回层是刚开始写的代码，结构比较混乱
+ ⏰ 排序层的LR和召回层的矩阵分解MF实现的不太好
+ ⏰ 召回层的负采样方式写的比较混乱，负采样方式包括全局随机采样，batch内负采样，这里需要改进
+ ⏰ 在召回层和排序层引入一些修偏机制，比如双塔模型batch内负采样会打压热门物料，在Loss内引入修偏机制 
+ ⏰ 代码加入注释和讲解
+ ⏰ 考虑写一个特征读取框架，目前的特征读取依赖硬编码，灵活度不够，每加一个特征都需要大费周章的加很多代码



# 日志
- 2025.10.4 
更新了特征工程框架，使用工业界普遍使用的slot_id的方式进行特征管理，方便添加新的特征以及特征的复用，目前支持sparse特征、dense特征以及序列特征。后续会写一个文档来介绍这个特征工程框架的使用方法，便于大家使用。