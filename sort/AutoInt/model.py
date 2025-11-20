import sys
sys.path.append('/data2/zhy/Movie_Recsys/')

from BaseModel.base_model import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_utils.lr_schedule import CosinDecayLR
from model_utils.utils import MLP, TransformerBlock

class AutoIntModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=[32, 32, 1]):
        super(AutoIntModel, self).__init__()
        dims = [input_dim] + hidden_dims
        
        self.deep_network = MLP(dims=dims)
        self.transformer_block = nn.Sequential(*[TransformerBlock(embed_dim=input_dim, num_heads=2, ff_dim=32) for _ in range(1)])

    
    def forward(self, x):
        x = self.transformer_block(x.unsqueeze(1)).squeeze(1)  # B x input_dim
        deep_out = self.deep_network(x)
        return torch.sigmoid(deep_out)
    

class AutoInt(BaseModel):
    def __init__(self, config_path, dataloaders={}, hparams={}):
        super(AutoInt, self).__init__(config_path)
        
        self.save_hyperparameters(hparams)
        self.hparams_ = hparams

        # 定义Deep模型的网络结构，包括输入维度和隐藏层维度，这里减去wide特征的维度
        self.score_fc = AutoIntModel(input_dim=self.user_input_dim + self.item_input_dim, hidden_dims=[32, 32, 1])
        
        self.movies_dataloader = dataloaders.get('movies_dataloader', None)
        self.val_dataloader_ = dataloaders.get('val_dataloader', None)
        
        


    def bceLoss(self, preds, labels):
        return F.binary_cross_entropy(preds.view(-1), labels.view(-1), reduction='mean')


    def forward(self, x):
        features = self.get_inp_embedding(x)  # 获取输入特征向量
        return self.score_fc(features)  # 返回预测分数

    
    def get_inp_embedding(self, batch):
        features, _, _ = self.get_embedding_from_set(batch, self.user_feature_names | self.item_feature_names)
        return features
    
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
        feature = self.get_inp_embedding(batch)  # 获取输入特征向量
        return self.score_fc(feature)  # 返回预测分数


    @torch.no_grad()
    def on_train_epoch_end(self):
        if self.current_epoch % 1 == 0:
            self.eval()