import os
import torch
from torch import optim, nn, utils, Tensor
from torch.utils.data import Dataset, DataLoader
import pickle
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score


class Trainer:
    def __init__(self, model: nn.Module, model_name: str, log_dir: str=None):
        super().__init__()

        self.model = model
        self.model_name = model_name

        if log_dir is not None:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
        else:
            if not os.path.exists('./log/'):
                os.makedirs('./log/')
        self.log_dir = log_dir if log_dir is not None else './log/'

        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.train_step = 0

    def set_model(self, model: nn.Module):
        self.model = model

    def set_config(self, epoch, lr, optimizer: str, dataloader, eval_dataloader,
                   lr_schedule=None, lr_min=None, save_epoch: float=1.0, eval_epoch: float=1.0):
        self.epoch = epoch
        self.lr = lr
        
        assert optimizer in ['adam']
        if optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        

        self.dataloader = dataloader
        self.eval_dataloader = eval_dataloader

        assert lr_schedule in [None, 'cosin']
        if lr_schedule == 'cosin':
            assert lr_min is not None
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                                self.optimizer, 
                                T_max=(len(self.dataloader)) * self.epoch, 
                                eta_min=lr_min
                            )
        self.save_step = int(len(dataloader) * save_epoch)
        self.eval_step = int(len(dataloader) * eval_epoch)

    def add_log(self, ret):
        for k in ret.keys():
            self.writer.add_scalar(k, ret[k], self.train_step)


    def train(self):
        for epoch in range(self.epoch):
            self.model.train()
            # loss_epoch = 0.0
            tqdm_bar = tqdm(self.dataloader, ncols=100)
            # data_index = 1
            for data in tqdm_bar:
                self.optimizer.zero_grad()
                ret = self.model(data)
                self.add_log(ret)

                ret['loss'].backward()
                self.optimizer.step()
                # loss_epoch += ret['loss']
                self.scheduler.step()
                tqdm_bar.set_postfix_str(
                    'lr: %.6f' % (self.optimizer.state_dict()['param_groups'][0]['lr'])
                )
                self.train_step += 1
                # data_index += 1
            # print('Epoch: %d, Loss: %.3f' % (epoch, loss_epoch / len(self.dataloader)))

                if self.train_step != 0 and self.train_step % self.save_step == 0:
                    torch.save(self.model, './%s_epoch_%d.pth' % (self.model_name, self.train_step))
                if self.train_step != 0 and self.train_step % self.eval_step == 0:
                    self.eval()
        torch.save(self.model, './%s_final.pth' % self.model_name)
        self.writer.close()

    def eval(self):
        self.model.eval()
        tqdm_bar = tqdm(self.eval_dataloader, ncols=100)
        logits, labels = [], []
        # data_index = 1
        for data in tqdm_bar:
            with torch.no_grad():
                logit, label = self.model.eval_(data)
            logits.append(logit)
            labels.append(label)
        eval_auc = roc_auc_score(labels, logits)
        self.add_log({'eval_AUC': eval_auc})
        print('Eval AUC:', eval_auc)
