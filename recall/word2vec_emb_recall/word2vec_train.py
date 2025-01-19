import gensim
import yaml
import pickle
from tqdm import tqdm

with open('./w2v_conf.yaml', 'r') as f:
    conf = yaml.safe_load(f)

train_readlist: dict = pickle.load(open(conf['train_readlist_path'], 'rb'))

# 构造word2vec的训练样本
train_sentence = []
for user_id in tqdm(train_readlist.keys()):
    item_sequence = train_readlist[user_id]
    item_sequence = [x[0] for x in item_sequence]
    if len(item_sequence) >= conf['sequence_len']:
        for i in range(len(item_sequence) - conf['sequence_len']):
            train_sentence.append(item_sequence[i: i + conf['sequence_len']])
    else:
        ori_item_seq = item_sequence[:]
        while len(item_sequence) < conf['sequence_len']:
            item_sequence += ori_item_seq
        train_sentence.append(item_sequence[:conf['sequence_len']])

model = gensim.models.Word2Vec(
    train_sentence,
    vector_size=conf['emb_dim'],
    window=conf['window_size'],
    min_count=1,
    workers=4
)


wordvec = model.wv['word']
print(wordvec)