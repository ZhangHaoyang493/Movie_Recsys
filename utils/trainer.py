import os
import torch
from torch import optim, nn, utils, Tensor
from torch.utils.data import Dataset, DataLoader
import pickle
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
import yaml
import sys
sys.path.append('..')
from FeatureTools.BaseDataLoader import get_dataloader
import logging

class Logger:
    def get_logger(self, log_path):
        logger = logging.getLogger("my_logger")
        logger.setLevel(logging.DEBUG)

        # 创建一个控制台 handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 创建一个文件 handler
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)

        # 创建 formatter 并添加到 handler
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # 添加 handler 到 logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        return logger


    

class Trainer:
    def __init__(self, model_config_file, fea_config_file, model):
        super().__init__()

        with open(model_config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        self.fea_config_file = fea_config_file

        self.model = model
        self.model_name = self.config['model_name']

        # Log地址
        if 'log_dir' in self.config:
            if not os.path.exists(self.config['log_dir']):
                os.makedirs(self.config['log_dir'])
        else:
            if not os.path.exists('./log/'):
                os.makedirs('./log/')
        self.log_dir = self.config['log_dir'] if 'log_dir' in self.config else './log/'

        # 日志的writer
        # self.writer = SummaryWriter(log_dir=self.log_dir)
        self.train_step = 0

        self.set_dataloader()
        self.set_config()

        self.logger = Logger().get_logger('./train.log')


    def set_dataloader(self):
        self.dataloader = get_dataloader(self.fea_config_file, self.config['batch_size'], self.config['num_workers'], 'train')
        self.eval_dataloader = get_dataloader(self.fea_config_file, self.config['batch_size'], self.config['num_workers'], 'test')

    def set_config(self):
        self.epoch = int(self.config['epoch'])
        self.lr = float(self.config['lr'])
        
        assert self.config['optimizer'] in ['adam']
        if self.config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        
        # self.dataloader = dataLoader
        # self.eval_dataloader = eval_dataloader

        assert self.config['lr_schedule'] in [None, 'cosine']
        if self.config['lr_schedule'] == 'cosine':
            assert self.config['lr_min'] is not None
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                                self.optimizer, 
                                T_max=(len(self.dataloader)) * self.epoch, 
                                eta_min=float(self.config['lr_min'])
                            )
        self.save_step = int(len(self.dataloader) * self.config['save_epoch'])
        self.eval_step = int(len(self.dataloader) * self.config['save_epoch'])

        self.model_save_path = '.' if 'model_save_path' not in self.config else self.config['model_save_path']
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)



    # def add_log(self, ret):
    #     for k in ret.keys():
    #         self.writer.add_scalar(k, ret[k], self.train_step)


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
                    torch.save(self.model, os.path.join(self.model_save_path, '%s_epoch_%d.pth' % (self.model_name, self.train_step)))
                if self.train_step != 0 and self.train_step % self.eval_step == 0:
                    self.eval()
        torch.save(self.model, os.path.join(self.model_save_path, '%s_final.pth' % self.model_name))
        # self.writer.close()

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
        
        logits = torch.concat(logits, dim=0).view(-1) # Nx1
        labels = torch.concat(labels, dim=0).view(-1) # Nx1
        logits = logits.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        
        eval_auc = roc_auc_score(labels, logits)
        # self.add_log({'eval_AUC': eval_auc})
        self.logger.info('Iterations: %d, AUC: %.5f' % (self.train_step, eval_auc))
        # print('Eval AUC:', eval_auc)
