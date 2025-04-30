import gensim
import yaml
import pickle
import random
from tqdm import tqdm
import os.path as osp



# 构图
# 给item的id统统加上user的个数，防止id重复
def build_graph(train_readlist, user_num):
    graph: dict = {}
    for userid in train_readlist.keys():
        for item_info in train_readlist[userid]:
            itemid = item_info[0]
            itemid = str(int(itemid) + user_num)
            if userid not in graph:
                graph[userid] = []
            graph[userid].append(itemid)
            if itemid not in graph:
                graph[itemid] = []
            graph[itemid].append(userid)
    return graph

def random_walk(graph, node_walk_num, walk_len):
    """
    node_walk_num: 每个节点游走几次
    walk_len: 游走的路径长度
    """
    walk_res = []
    for id in graph.keys():
        for i in range(node_walk_num):
            path = []
            path.append(id)
            while len(path) < walk_len:
                path.append(random.choice(graph[path[-1]]))
            walk_res.append(path)
    return walk_res



# 构造word2vec的训练样本
# train_sentence = []
# for user_id in tqdm(train_readlist.keys()):
#     item_sequence = train_readlist[user_id]
#     item_sequence = [x[0] for x in item_sequence]
#     if len(item_sequence) >= conf['sequence_len']:
#         for i in range(len(item_sequence) - conf['sequence_len']):
#             train_sentence.append(item_sequence[i: i + conf['sequence_len']])
#     else:
#         ori_item_seq = item_sequence[:]
#         while len(item_sequence) < conf['sequence_len']:
#             item_sequence += ori_item_seq
#         train_sentence.append(item_sequence[:conf['sequence_len']])





if __name__ == '__main__':
    with open('./deepwalk_conf.yaml', 'r') as f:
        conf = yaml.safe_load(f)

    train_readlist: dict = pickle.load(open(conf['train_readlist_path'], 'rb'))
    graph = build_graph(train_readlist, conf['user_num'])
    walk_paths = random_walk(graph, conf['walk_num'], conf['sequence_len'])

    model = gensim.models.Word2Vec(
        walk_paths,
        vector_size=conf['emb_dim'],
        window=conf['window_size'],
        min_count=1,
        workers=4
    )

    model.save(osp.join(conf['save_path'], 'deepwalk_model.pth'))
    # 保存嵌入向量
    model.wv.save_word2vec_format(osp.join(conf['save_path'], 'deepwalk_embeddings.txt'), binary=False)