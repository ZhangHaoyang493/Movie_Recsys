# 关于配置文件的说明
本项目的配置文件在做特征工程、模型构建的时候都会使用到。配置文件是一个json文件，包含了下面内容：
- **特征工程配置**：定义了做特征工程所需要的参数：
  - 特征提取所需要的文件路径：例如元数据文件路径、评分数据文件路径、特征提取脚本路径等。
  - 需要提取的特征名称。
- **模型的Embedding Table配置**：定义了模型中使用的Embedding Table的相关参数，例如Embedding Size等。
- **模型的相关配置**：模型的相关配置主要有两类：
  - 召回模型在跑测试集的时候，对于召回的物料可能需要进行历史消重，需要传入一些预先准备好的用户历史文件等。
  - 有些模型可能需要额外指定一些相关参数，例如DeepFM模型的FM侧的Embedding的维度。

下面我们将详细介绍配置文件的各个部分。

## 特征工程配置

### 关于特征提取所需要的文件路径的配置字段
首先给出一个特征工程文件路径配置的示例：
```json
{
  "ratings_path": [
      "/data2/zhy/Movie_Recsys/MovieLens_1M_data/train_ratings.dat",
      "/data2/zhy/Movie_Recsys/MovieLens_1M_data/test_ratings.dat"
    ],
  "movies_path": "/data2/zhy/Movie_Recsys/MovieLens_1M_data/movies.dat",
  "users_path": "/data2/zhy/Movie_Recsys/MovieLens_1M_data/users.dat",
  "feature_extractor_path": "/data2/zhy/Movie_Recsys/FeaturesGenerator/feature_extractor.py",
  "out_basedir": "/data2/zhy/Movie_Recsys/FeatureFiles"
}
```
我们对于以上字段进行解释：
|字段名 | 值类型 | 说明 |
|-------|-------|-------|
|`ratings_path` | 列表 | 评分元数据的路径，每一行的数据是以下格式`user_id::movie_id::rating::timestamp`，和MovieLens数据集提供的ratings.dat文件的格式保持一致，它的值是列表类型，可以包含多个ratings文件的路径，因此可以同时对多个ratings文件进行特征处理。一般来说，`ratings_path`的值包含训练集和测试集的评分数据文件路径，**注意，训练集的路径一定要在测试集路径的前面。**|
|`movies_path` | 字符串 | 电影的元信息文件路径，是一个字符串，指向电影的元信息文件。数据行格式为`movie_id::name::movie_genre`，和MovieLens数据集提供的movies.dat文件的格式保持一致。|
|`users_path` | 字符串 | 用户的元信息文件路径，是一个字符串，指向用户的元信息文件。数据行格式为`user_id::gender::age::occupation::zip_code`，和MovieLens数据集提供的users.dat文件的格式保持一致。|
|`feature_extractor_path` | 字符串 | 特征工程脚本的路径，是一个字符串，指向用户自定义的特征提取脚本的文件路径。有关特征提取脚本的详细说明，请参考[特征工程框架介绍](./feature_engineering_introduction.md)|
|`out_basedir` | 字符串 | 特征文件的输出目录，通常是一个字符串，指向特征文件的输出目录。|

### 需要提取的特征名称配置字段

