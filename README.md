# 🔥Movie_Recsys概述
这是项目我在学习推荐算法道路上对学习到的一些理论的实践，基于MovieLens 1M数据集，属于一个玩具项目。项目虽然很简单，
但是在这个项目中，我实现了一个包含召回层、排序层在内的一个完整的推荐系统，同时也涉及到了特征工程、模型评估等一系列为了提升推荐系统的性能所必备的操作。
本项目的深度学习模型全部使用Pytorch进行构建，方便像我一样在学习推荐系统之前只会Pytorch的同学。

### 🔥🔥🔥正在更新
### 👏欢迎大家参与本项目💪，多Star🌟，大家的Star是我最大的动力🌟

## 环境安装
- python版本：3.9.24
- 安装依赖包
```bash
pip install -r requirements.txt
```
其中torch需要手动下载安装，torch下载地址：https://download.pytorch.org/whl/torch/

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

正在更新...

## 召回层
### Baseline：基于物品的协同过滤（ItemCF）

ItemCF的各个版本的HitRate@50表现如下：
|文件路径|改进点|HitRate@50|
|-------|-------|------|
|`recall/ItemCF/itemCF_base.py`|最简单的ItemCF|14.9418%|
|`recall/ItemCF/itemCF_version1.py`|引入评分权重，评分>=3.0的权重为1，评分<3s.0的权重为0.5|15.8907%|
|`recall/ItemCF/itemCF_version2.py`|更精细的评分权重，评分=5.0的权重为1，评分=4.0的权重为0.8，...，评分=1.0的权重为0.2|16.9662%|

正在更新...


# 日志
- 2025.10.4 
更新了特征工程框架，使用工业界普遍使用的slot_id的方式进行特征管理，方便添加新的特征以及特征的复用，目前支持sparse特征、dense特征以及序列特征。后续会写一个文档来介绍这个特征工程框架的使用方法，便于大家使用。