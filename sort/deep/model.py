import sys
sys.path.append('/data2/zhy/Movie_Recsys/')

from BaseModel.base_model import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import faiss
import numpy as np

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

        self._initialize_weights()

    # 初始化权重
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)  # 或 xavier_normal_
                if m.bias is not None:
                    init.zeros_(m.bias)

    def bceLoss(self, preds, labels):
        return F.binary_cross_entropy(preds.view(-1), labels.view(-1), reduction='mean')


    def forward(self, x):
        inp_feature = self.get_inp_embedding(x)  # 获取输入特征向量
        scores = F.sigmoid(self.score_fc(inp_feature))  # 通过全连接层计算得分

        gt_islike = x['label'][:, 1]  # 获取是否喜欢的标签

        loss = self.bceLoss(scores, gt_islike)
        self.log('train_loss', loss)
        return loss

    def log_last_layer_embeddings(self):
        # 获取self.score_fc最后一层的权重
        last_layer_weights = self.score_fc[-1].weight.data  # shape: 1
        # 计算平均值
        mean_weight = torch.mean(last_layer_weights).item()
        # 记录到日志中
        self.log('last_layer_mean_weight', mean_weight)


    def get_inp_embedding(self, batch):
        embeddings = []
        all_slots = list(self.user_slots) + list(self.item_slots)
        for slot_id in all_slots:
            emb = self.get_slots_embedding(slot_id, batch[slot_id])
            if slot_id in self.sparse_slots:
                embeddings.append(emb)
            elif slot_id in self.dense_slots:
                embeddings.append(emb)
            elif slot_id in self.array_slots:
                mask = batch.get(f"{slot_id}_mask", None)  # bxarr_lenxdim
                if mask is not None:
                    emb = emb * mask.unsqueeze(-1)  # 应用mask
                    emb = emb.sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)  # 避免除以零，mean pooling
                embeddings.append(emb)
        feature_vector = torch.cat(embeddings, dim=1)  # 在特征维度上拼接
        return feature_vector
    
    def training_step(self, batch, batch_idx):
        loss = self.forward(batch)
        # 记录最后一层的权重，用于监控训练过程
        self.log_last_layer_embeddings()
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
        for batch in tqdm(self.val_dataloader_, desc="Evaluating AUC", ncols=100):
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
            inp_feature = self.get_inp_embedding(batch)  # 获取输入特征向量
            scores = self.score_fc(inp_feature)  # 通过全连接层计算得分
            labels = batch['label'][:, 1]  # 获取是否喜欢的标签
            # 计算AUC
            auc = roc_auc_score(labels.cpu().numpy(), scores.cpu().detach().numpy())
            auc_sum += auc
            idx += 1
        avg_auc = auc_sum / idx if idx > 0 else 0
        self.log('Val_AUC', avg_auc)
        print(f"Validation AUC: {avg_auc}")

    @torch.no_grad()
    def on_train_epoch_end(self):
        if self.current_epoch % 1 == 0:
            self.eval_auc()