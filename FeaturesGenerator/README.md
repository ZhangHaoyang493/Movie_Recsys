# 关于Feature Generator中的json配置文件的字段说明

## ratings_path
ratings_path: 评分数据文件路径，格式为 user_id::movie_id::rating::timestamp

## movies_path
movie_path: 电影数据文件路径，格式为 movie_id::title::genres

## users_path
users_path: 用户数据文件路径，格式为 user_id::gender::age::occupation::zip_code

## feature_extractor_path
feature_extractor_path: 特征提取器文件路径，格式为 python 文件

## out_feature_file_path
out_feature_file_path: 特征输出文件路径，格式为 slot_id:feature_hash_value slot_id:feature_hash_value ... \t label

## embedding_idx_dict_path
embedding_idx_dict_path: 特征embedding索引字典文件路径，格式为 json 文件

这个主要用于存储每个slot_id对应的特征值到embedding索引的映射关系，方便后续再次使用这次的映射关系构造测试数据，格式如下：
```json
{
  "slot_id1": {
    "feature_value1": embedding_idx1,
    "feature_value2": embedding_idx2,
    ...
  },
  "slot_id2": {
    "feature_value1": embedding_idx1,
    "feature_value2": embedding_idx2,
    ...
  },
  ...
}
```

## slot_ids
slot_ids: 特征提取的slot id列表，格式为 [1, 2, 3, ...]，每个slot id对应一个特征提取函数 feature_extractor_{slot_id}，例如 slot_id 1 对应的函数为 feature_extractor_1

## share_slot_ids
share_slot_ids: 共享特征的slot id列表，格式为 [1, 2, 3, ...]，这些slot id对应的特征提取函数会共享同一个embedding索引映射字典

例如对于用户最近观看的10部电影这个特征，假设slot id为6，那么对于所有用户来说，这个slot id 6的特征值都是电影id的组合，因此这些电影id的embedding索引映射字典可以和movie id的这个特征的embedding共享，当然，考虑到数据量，你可以选择不共享，单独为slot id 6创建一个映射字典