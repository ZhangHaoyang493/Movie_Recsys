# 🌟Movie_Recsys概述
这是项目我在学习推荐算法道路上对学习到的一些理论的实践，基于MovieLens 1M数据集，属于一个玩具项目（toy project）。项目虽然很简单，
但是在这个项目中，我实现了一个包含召回层、排序层在内的一个完整的推荐系统，同时也涉及到了特征工程、模型评估等一系列为了提升推荐系统的性能所必备的操作。
本项目的深度学习模型全部使用Pytorch进行构建，方便像我一样在学习推荐系统之前只会Pytorch的同学（笑）。

# 📈Data
MovieLens数据集1M（https://grouplens.org/datasets/movielens/）

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
├── MF # 矩阵分解召回模型
│   ├── MF_recall.py        # 使用训练好的 MF 模型进行用户-物品的召回推荐
│   └── MF_train.py         # 使用评分矩阵进行 MF 模型训练，学习潜在因子向量
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
+ ⏰ 考虑写一个特征读取框架，目前的特征读取依赖硬编码，灵活度不够，每加一个特征都需要大费周章的加很多代码
+ ⏰ 召回层代码的重构，召回层是刚开始写的代码，结构比较混乱
+ ⏰ 排序层的LR和召回层的矩阵分解MF实现的不太好
+ ⏰ 召回层的负采样方式写的比较混乱，负采样方式包括全局随机采样，batch内负采样，这里需要改进
+ ⏰ 在召回层和排序层引入一些修偏机制，比如双塔模型batch内负采样会打压热门物料，在Loss内引入修偏机制 
+ ⏰ 代码加入注释和讲解


### 👏欢迎大家参与本项目💪，多Star🌟，大家的Star是我最大的动力🌟

# 日志
2025.4.6

最近有点忙，更新速度会减慢，过来这一两个月会继续更新💪