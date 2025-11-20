import sys
sys.path.append('/data2/zhy/Movie_Recsys/')

from BaseModel.base_model import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_utils.lr_schedule import CosinDecayLR
from model_utils.utils import MLP

class DeepFMModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=[32, 32, 1]):
        super(DeepFMModel, self).__init__()
        dims = [input_dim] + hidden_dims
        
        self.deep_network = MLP(dims=dims)
        self.bias = nn.Parameter(torch.zeros(1))

    
    def forward(self, fm_w, fm_v, deep_x):
        # FM一阶
        fm_first_order = torch.sum(fm_w, dim=1, keepdim=True) + self.bias # Bx1
        # FM二阶
        fm_second_order = 0.5 * torch.sum(
            torch.pow(torch.sum(fm_v, dim=1), 2) - torch.sum(torch.pow(fm_v, 2), dim=1),
            dim=1,
            keepdim=True
        )  # Bx1

        deep_out = self.deep_network(deep_x)
        return F.sigmoid(fm_first_order + fm_second_order + deep_out)
    

class DeepFM(BaseModel):
    def __init__(self, config_path, dataloaders={}, hparams={}):
        super(DeepFM, self).__init__(config_path)
        
        self.save_hyperparameters(hparams)
        self.hparams_ = hparams

        deepfm_config = self.config['deepfm_cfg']
        self.fm_feature_names = set(deepfm_config['fm_feature_names'])
        self.fm_dim = deepfm_config['fm_dim']

        fm_sum_dim = len(self.fm_feature_names) * (1 + self.fm_dim)  # fm一阶和二阶的总维度
        # 定义Deep模型的网络结构，包括输入维度和隐藏层维度，这里减去wide特征的维度
        self.score_fc = DeepFMModel(input_dim=self.user_input_dim + self.item_input_dim - fm_sum_dim, hidden_dims=[32, 32, 1])
        
        self.movies_dataloader = dataloaders.get('movies_dataloader', None)
        self.val_dataloader_ = dataloaders.get('val_dataloader', None)
        
        


    def bceLoss(self, preds, labels):
        return F.binary_cross_entropy(preds.view(-1), labels.view(-1), reduction='mean')


    def forward(self, x):
        fm_first_order_x, fm_second_order_x, deep_x = self.get_inp_embedding(x)  # 获取输入特征向量
        return self.score_fc(fm_first_order_x, fm_second_order_x, deep_x)  # 返回预测分数

    
    def get_inp_embedding(self, batch):
        features, dims, fnames = self.get_embedding_from_set(batch, self.user_feature_names | self.item_feature_names)
        fm_first_order_x = []
        fm_second_order_x = []
        deep_x = []
        start_idx = 0
        for dim, fname in zip(dims, fnames):
            end_idx = start_idx + dim
            if fname in self.fm_feature_names:
                fm_first_order_x.append(features[:, start_idx:start_idx+1])  # Bx1
                fm_second_order_x.append(features[:, start_idx+1:start_idx+1+self.fm_dim])  # Bxdim
                deep_x.append(features[:, start_idx+1+self.fm_dim:end_idx])  # Bx(剩余维度)
            else:
                deep_x.append(features[:, start_idx:end_idx])  # Bxdim
            start_idx = end_idx
        fm_first_order_x = torch.cat(fm_first_order_x, dim=1)
        fm_second_order_x = torch.stack(fm_second_order_x, dim=1)
        deep_x = torch.cat(deep_x, dim=1)

        return fm_first_order_x, fm_second_order_x, deep_x
    
    def training_step(self, batch, batch_idx):
        scores = self.forward(batch)
        labels = batch['label'][:, 1]  # 获取是否喜欢的标签
        loss = self.bceLoss(scores, labels)# + 1e-6 * torch.mean(torch.abs(wide_x))  # 计算二元交叉熵损失
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
    def inference(self, batch):
        fm_first_order_x, fm_second_order_x, deep_x = self.get_inp_embedding(batch)  # 获取输入特征向量
        return self.score_fc(fm_first_order_x, fm_second_order_x, deep_x)  # 返回预测分数


    @torch.no_grad()
    def on_train_epoch_end(self):
        if self.current_epoch % 1 == 0:
            self.eval()