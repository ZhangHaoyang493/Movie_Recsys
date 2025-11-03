# 关于配置文件字段的说明
我们提供了一个示例的配置文件：
```json
{
  "ratings_path": [
    "/data2/zhy/Movie_Recsys/MovieLens_1M_data/train_ratings.dat",
    "/data2/zhy/Movie_Recsys/MovieLens_1M_data/test_ratings.dat"
  ],
  "test_ratings_path": "/data2/zhy/Movie_Recsys/MovieLens_1M_data/test_ratings.dat",
  "movies_path": "/data2/zhy/Movie_Recsys/MovieLens_1M_data/movies.dat",
  "users_path": "/data2/zhy/Movie_Recsys/MovieLens_1M_data/users.dat",
  "feature_extractor_path": "/data2/zhy/Movie_Recsys/FeaturesGenerator/feature_extractor.py",
  "out_basedir": "/data2/zhy/Movie_Recsys/FeatureFiles",
  "data_path": "/data2/zhy/Movie_Recsys/FeaturesGenerator/train_data.txt",
  "embedding_idx_dict_path": null,
  "slot_ids": [
    1,  // 用户ID
    2,  // 电影ID
    3,  // 用户性别
    4,  // 年龄
    5,  // 电影年份
    6   // 最近观看的电影
  ],
  "share_slot_ids": {
    "6": 2
  },
  "sparse_slots": [1, 2, 3],
  "dense_slots": [4, 5],
  "array_slots": [6],
  "embedding_size": {
    "1": 16,
    "2": 16,
    "3": 8
  },
  "embedding_table_size": {
    "1": 6040,
    "2": 3883,
    "3": 2
  },
  "array_max_length": {
    "6": 10
  },
  "item_slots": [2, 5],
  "user_slots": [1, 4, 6],
  "dense_feature_dim": 1,
  "stage": "recall"
}
```
接下来，我们对配置文件中的各个字段进行说明：
```json

```