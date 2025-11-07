import sys
sys.path.append('/data2/zhy/Movie_Recsys/')

from BaseModel.base_model import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import faiss
import numpy as np
from math import log2
import random

from DataReader.data_reader import DataReader
from tqdm import tqdm
from model_utils.lr_schedule import CosinDecayLR
from sklearn.metrics import roc_auc_score

class Deep(BaseModel):
    def __init__(self, config_path, dataloaders={}, hparams={}):
        super(Deep, self).__init__(config_path)
        
        self.save_hyperparameters(hparams)
        self.hparams_ = hparams

        # 定义Deep模型的网络结构
        self.score_fc = nn.Sequential(
            nn.Linear(self.user_input_dim + self.item_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.movies_dataloader = dataloaders.get('movies_dataloader', None)
        self.val_dataloader_ = dataloaders.get('val_dataloader', None)


    def bceLoss(self, preds, labels):
        return F.binary_cross_entropy(preds.view(-1), labels.view(-1), reduction='mean')


    def forward(self, x):
        inp_feature = self.get_inp_embedding(x)  # 获取输入特征向量
        scores = F.sigmoid(self.score_fc(inp_feature))  # 通过全连接层计算得分
        return scores  # 返回预测分数


    # def get_inp_embedding(self, batch):
    #     embeddings = []
    #     all_feature_names = list(self.user_feature_names) + list(self.item_feature_names)
    #     for feature_name in all_feature_names:
    #         emb = self.get_features_embedding(feature_name, batch[feature_name])
    #         if feature_name in self.sparse_feature_names:
    #             embeddings.append(emb)
    #         elif feature_name in self.dense_feature_names:
    #             embeddings.append(emb)
    #         elif feature_name in self.array_feature_names:
    #             mask = batch.get(f"{feature_name}_mask", None)  # bxarr_lenxdim
    #             if mask is not None:
    #                 emb = emb * mask.unsqueeze(-1)  # 应用mask
    #                 emb = emb.sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)  # 避免除以零，mean pooling
    #             embeddings.append(emb)
    #     feature_vector = torch.cat(embeddings, dim=1)  # 在特征维度上拼接
    #     return feature_vector
    def get_user_embedding(self, batch):
        user_embeddings = []
        for feature_name in self.user_feature_names:
            emb = self.get_features_embedding(feature_name, batch[feature_name])
            if feature_name in self.sparse_feature_names:
                user_embeddings.append(emb)
            elif feature_name in self.dense_feature_names:
                user_embeddings.append(emb)
            elif feature_name in self.array_feature_names:
                mask = batch.get(f"{feature_name}_mask", None)  # bxarr_lenxdim
                if mask is not None:
                    emb = emb * mask.unsqueeze(-1)  # 应用mask
                    emb = emb.sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)  # 避免除以零，mean pooling
                user_embeddings.append(emb)
        user_feature_vector = torch.cat(user_embeddings, dim=1)  # 在特征维度上拼接
        return user_feature_vector
    
    def get_item_embedding(self, batch):
        item_embeddings = []
        for feature_name in self.item_feature_names:
            emb = self.get_features_embedding(feature_name, batch[feature_name])
            if feature_name in self.sparse_feature_names:
                item_embeddings.append(emb)
            elif feature_name in self.dense_feature_names:
                item_embeddings.append(emb)
            elif feature_name in self.array_feature_names:
                mask = batch.get(f"{feature_name}_mask", None)  # bxarr_lenxdim
                if mask is not None:
                    emb = emb * mask.unsqueeze(-1)  # 应用mask
                    emb = emb.sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)  # 避免除以零，mean pooling
                item_embeddings.append(emb)
        item_feature_vector = torch.cat(item_embeddings, dim=1)  # 在特征维度上拼接
        return item_feature_vector
    
    def get_inp_embedding(self, batch):
        user_feature_vector = self.get_user_embedding(batch)
        item_feature_vector = self.get_item_embedding(batch)
        feature_vector = torch.cat([user_feature_vector, item_feature_vector], dim=1)  # 在特征维度上拼接
        return feature_vector
    
    def training_step(self, batch, batch_idx):
        scores = self.forward(batch)
        labels = batch['label'][:, 1]  # 获取是否喜欢的标签
        loss = self.bceLoss(scores, labels)  # 计算二元交叉熵损失
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams_['lr'], betas=(0.9, 0.999))
        lr_scheduler = CosinDecayLR(optimizer, lrs=[self.hparams_['lr'], self.hparams_['min_lr']], milestones=self.hparams_['lr_milestones'])
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'interval': 'step',  # 每个训练步骤调用一次
                'frequency': 1
            }

        }

    @torch.no_grad()
    def eval_auc(self):
        auc_sum = 0.0
        idx = 0
        pred, labels = [], []
        for batch in tqdm(self.val_dataloader_, desc="Evaluating AUC", ncols=100):
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
            inp_feature = self.get_inp_embedding(batch)  # 获取输入特征向量
            scores = F.sigmoid(self.score_fc(inp_feature))  # 通过全连接层计算得分
            label = batch['label'][:, 1]  # 获取是否喜欢的标签
            pred.extend(scores.view(-1).cpu().detach().numpy().tolist())
            labels.extend(label.view(-1).cpu().numpy().tolist())
            # 计算AUC
            # auc = roc_auc_score(labels.cpu().numpy(), scores.cpu().detach().numpy())
            # auc_sum += auc
            # idx += 1
        avg_auc = roc_auc_score(labels, pred)
        self.log('Val_AUC', avg_auc)
        print(f"Validation AUC: {avg_auc}")

    # @torch.no_grad()
    # def eval_ndcg_and_hr(self, k=10):
    #     ndcg_sum = 0.0
    #     hr_sum = 0.0
    #     idx = 0

    #     # 将电影批次添加到列表中
    #     item_batchs = []
    #     for movie_batch in self.movies_dataloader:
    #         for key in movie_batch:
    #             if isinstance(movie_batch[key], torch.Tensor):
    #                 movie_batch[key] = movie_batch[key].to(self.device)
    #         item_batchs.append(movie_batch)
        
    #     for batch in tqdm(self.val_dataloader_, desc="Evaluating NDCG", ncols=100):
    #         # if random.random() > 0.15:
    #         #     continue  # 只评估15%的样本以加快速度
    #         batch_size = batch['user_id'].size(0)
    #         if batch_size != 1:
    #             raise ValueError("NDCG evaluation only supports batch_size=1 for accurate user history filtering.")
    #         for key in batch:
    #             if isinstance(batch[key], torch.Tensor):
    #                 batch[key] = batch[key].to(self.device)
            
    #         user_id = batch['user_id'][0].item()
    #         user_true_id = self.emb_idx_2_val_dict['user_id'][str(user_id)]
    #         user_history = set(self.user_history.get(user_true_id, []))
    #         user_feature = self.get_user_embedding(batch)
    #         all_score = []
    #         # for movie_batch in self.movies_dataloader:
    #         #     for key in movie_batch:
    #         #         if isinstance(movie_batch[key], torch.Tensor):
    #         #             movie_batch[key] = movie_batch[key].to(self.device)
    #         for movie_batch in item_batchs:
    #             item_feature = self.get_item_embedding(movie_batch)
    #             movie_batch_size = item_feature.size(0)
    #             user_feature_ = user_feature.repeat(movie_batch_size, 1)
    #             inp_feature = torch.cat([user_feature_, item_feature], dim=1)
    #             scores = F.sigmoid(self.score_fc(inp_feature))  # 通过全连接层计算得分
    #             scores = scores.view(-1).cpu().detach().numpy().tolist()
    #             movie_id = movie_batch['movie_id'].cpu().numpy().tolist()
    #             for m_id, score in zip(movie_id, scores):
    #                 if self.emb_idx_2_val_dict['movie_id'][str(m_id)] in user_history:
    #                     continue  # 过滤掉用户历史评分过的电影
    #                 all_score.append((m_id, score))
    #         all_score = sorted(all_score, key=lambda x: x[1], reverse=True)[:k]
    #         sort_items = [i[0] for i in all_score]
    #         target_item = batch['movie_id'][0].item()
    #         # 由于batch_size=1，并且验证集全部是正样本，我们只需要获取target item在top-k中的位置
    #         # 假设target的贡献度是1，其它都是0
    #         if target_item not in sort_items:
    #             dcg = 0.0
    #         else:
    #             hr_sum += 1.0
    #             index = sort_items.index(target_item)
    #             dcg = 1.0 / log2(index + 2)
    #         idcg = 1.0  # 理想情况下，只有一个正样本在第一位
    #         ndcg = dcg / idcg
    #         ndcg_sum += ndcg
    #         idx += 1
    #     avg_ndcg = ndcg_sum / idx if idx > 0 else 0
    #     self.log(f'Val_NDCG@{k}', avg_ndcg)
    #     self.log(f'Val_HR@{k}', hr_sum / idx if idx > 0 else 0)
    #     print(f"Validation NDCG@{k}: {avg_ndcg}")
    #     print(f"Validation HR@{k}: {hr_sum / idx if idx > 0 else 0}")

    @torch.no_grad()
    def on_train_epoch_end(self):
        if self.current_epoch % 1 == 0:
            self.eval_auc()