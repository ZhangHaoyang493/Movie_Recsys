# Movie_Recsys
电影推荐系统，基于MovieLens数据集

# Data
MovieLens数据集1M（https://grouplens.org/datasets/movielens/）


# 一些问题
+ DSSM使用纯NEG Loss出现问题，猜想是数据规模太小，负采样容易采样到正样本，对最终结果产生影响