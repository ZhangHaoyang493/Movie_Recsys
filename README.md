<center>
<h1>🔥Movie_Recsys：基于MovieLens的推荐系统实践</h1>
</center>

本项目是一个推荐系统的相关项目，基于MovieLens 1M数据集。在本项目中，作者实现了一个较为完整的推荐系统框架，包括数据切分，特征工程，召回层和排序层的模型实现、训练以及评估。

项目目前**正在更新**，后续会持续完善特征工程框架，添加更多的召回和排序模型，并且会撰写相关的技术文档，方便大家学习和使用。

当然，作者学识有限，难免会有疏漏之处，欢迎大家指出问题，一起交流学习，共同进步！

### 🔥🔥🔥本项目正在更新，👏欢迎大家参与本项目💪，多Star🌟，大家的Star是我最大的动力🌟

## 环境安装
- python版本：3.9.24
- 安装依赖包
```bash
pip install -r requirements.txt
```
其中torch需要手动下载安装，torch下载地址：https://download.pytorch.org/whl/torch/

## 📈数据集介绍
[MovieLens数据集1M](https://grouplens.org/datasets/movielens/)

该数据集包含大约6000个用户对4000部左右的电影的100万条评分数据。每条评分数据包含用户ID、电影ID、评分（1-5分）和时间戳。此外，数据集中还包含电影的元信息，如标题、类型等，以及用户的元信息，如年龄、性别、职业等。

数据集文件说明：
- ratings.dat：用户对电影的评分数据，格式为 user_id::movie_id::rating::timestamp
- movies.dat：电影的元信息，格式为 movie_id::title::genres
- users.dat：用户的元信息，格式为 user_id::age::gender::occupation::zip_code
- README.txt：数据集的说明文件

详细的字段说明可以参考数据集中的README.txt文件。

数据集保存在该项目的MovieLens_1M_data文件夹中。

<!-- ### 关于MovieLens_1M_data文件夹内容介绍 -->

<!-- #### 基本文件
|文件名|说明|
|-------|-------|
|ratings.dat|用户对电影的评分数据，格式为 user_id::movie_id::rating::timestamp|
|movies.dat|电影的元信息，格式为 movie_id::title::genres|
|users.dat|用户的元信息，格式为 user_id::age::gender::occupation::zip_code|
|README.txt|数据集的说明文件|
|train_test_split.ipynb|用于将原始数据集划分为训练集和测试集的Jupyter Notebook文件，划分思路为选定一个时间戳节点，然后将该时间戳之前的数据作为训练集，之后的数据作为测试集。|
|train_ratings.dat|划分好的训练集的评分数据，格式同ratings.dat，包含用户对电影的评分数据。|
|test_ratings.dat|划分好的测试集的评分数据，格式同ratings.dat，包含用户对电影的评分数据。| -->

<!-- ## 如何定义正负样本
正负样本如何定义，这是业务场景来决定的。对于MovieLens数据集，打分数据中给出了用户针对某一部电影的具体评分，有一点可以肯定，对于这些有评分的电影，用户肯定是看过电影再来打分的（假设没有胡乱打分的用户）。如果我们的业务场景是给用户推荐用户可能喜欢的电影，那么MovieLens数据集其实确实缺失户曝光未点击的样本。在本项目中，正负样本的定义如下：
##### 召回层
- 正样本：用户评分>=4.0的电影
- 负样本：batch内负采样
##### 排序层
- 正样本：用户评分>=4.0的电影
- 负样本：用户评分<4.0的电影 -->

<!-- ## 训练集、验证集的划分
对于每一个用户，我们找出其最后一个正反馈的电影作为验证集，我们的目标是根据用户的历史评分，推荐其可能正反馈的电影。 -->

## 特征框架

正在更新...

## 召回层
### 各种召回方式的测试结果
|文件路径|说明|HR@10|
|-------|-------|------|
|`recall/ItemCF/itemCF_base.py`|ItemCF：基于物料的协同过滤|8.3637%|
|`recall/DSSM`|DSSM：双塔模型|11.5601%|

正在更新...

## 排序层

### 各种排序方式的测试结果
|文件路径|说明|AUC|Params(K)|Paper Link|
|-------|-------|------|------|-------|
|`sort/LR`|逻辑回归(Baseline)|0.7604|10.0K|-|
|`sort/FM`|因子分解机(Factorization Machines)|0.7689|10.0K|[Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5694074)|
|`sort/deep`|多层MLP+ReLU|0.7824|39.8K + 2.0K|-|
|`sort/DCN`|加入了Deep Cross Network|0.7866|39.8K+2.224K|[Paper Link](https://arxiv.org/pdf/1708.05123)|
|`sort/Wide_Deep`|Wide&Deep模型|0.7847|49.8K+2.0K|[Paper Link](https://arxiv.org/pdf/1606.07792)|

# 日志
- 2025.10.4 
更新了特征工程框架，使用工业界普遍使用的slot_id的方式进行特征管理，方便添加新的特征以及特征的复用，目前支持sparse特征、dense特征以及序列特征。后续会写一个文档来介绍这个特征工程框架的使用方法，便于大家使用。