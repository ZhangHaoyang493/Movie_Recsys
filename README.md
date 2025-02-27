# Movie_Recsys
电影推荐系统，基于MovieLens数据集，项目正在开发，文档正在完善...

# Data
MovieLens数据集1M（https://grouplens.org/datasets/movielens/）

# 项目介绍
本项目实现了推荐链路中的召回层和精排层。
# 召回层
召回层实现了以下几个常用的召回方法及召回模型：
+ 协同过滤（CF）：协同过滤基于物品进行
+ 矩阵分解（MF）：可以理解为协同过滤的改进
+ 双塔模型（DSSM）：推荐领域经典的双塔模型
+ 基于Embedding的召回：首先使用流行的训练Embedding的方法训练出物品侧和用户侧的Embedding，然后将物品侧的Embedding存到faiss向量数据库中，召回时查询用户Embedding然后去faiss中查找topK最近邻向量。这里训练Embedding的方法包括两种：
    + word2vec方式
    + deepwalk方式
+ youtubeDNN：youtube提出的召回模型，目前正在开发...

# 精排层
精排层实现了以下几个模型：
+ LR：逻辑回归模型
+ FM模型
+ Deep模型：Embedding+MLP
+ Wide&Deep：LR+Deep模型
+ DeepFM：FM+Deep模型
+ DCNv1&DCNv2：深度交叉网络+Deep模型
+ DIN：深度兴趣模型
+ MMoE：混合专家模型
+ PLE：渐进式分层提取

# 展望
+ 特征工程的完善，加入到模型的训练：
    + 用户历史评分均值
    + 用户历史评分方差
    + 用户的活跃度（历史评分的物品个数）
    + 用户偏爱的电影类型前三名
    + 物品的热度
    + 物品历史得分均值
    + ...
+ youtubeDNN模型的开发
+ PLE