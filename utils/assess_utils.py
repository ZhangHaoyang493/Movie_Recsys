import torch
import numpy as np

class AccessTools:
    def __init__(self):
        pass

    # AUC的计算方式
    def AUC(self, logits: torch.Tensor, labels: torch.Tensor):
        logits = logits.view(-1)
        labels = labels.view(-1)
        assert logits.shape == labels.shape

        # 统计正负样本的数量
        pos_num = torch.sum(labels == 1)
        neg_num = torch.sum(labels == 0)

        if pos_num == 0 or neg_num == 0:
            return 0.0
        
        # 转换为python的list，方便后续处理
        logits = logits.detach().cpu().numpy().tolist()
        labels = labels.detach().cpu().numpy().tolist()

        # 按照分数进行排序
        logit_label = [(lo, la) for lo, la in zip(logits, labels)]
        logit_label = sorted(logit_label, key=lambda x: x[0])

        # 获取正样本排到负样本前的对数
        neg_num_now = 0
        pos_neg_pair_num = 0
        for i in range(len(logit_label)-1, -1, -1):
            logit, label = logit_label[i]
            if label == 0:
                neg_num_now += 1
            else:
                pos_neg_pair_num += neg_num_now

        return pos_neg_pair_num / (pos_num * neg_num)

    def RegAUC(self, predict: torch.Tensor, labels: torch.Tensor):
        pass

    def MSE(self, predict: torch.Tensor, labels: torch.Tensor):
        predict = predict.view(-1)
        labels = labels.view(-1)
        assert predict.shape == labels.shape

        return torch.mean((predict - labels) ** 2).item()
    
    def MAE(self, predict: torch.Tensor, labels: torch.Tensor):
        predict = predict.view(-1)
        labels = labels.view(-1)
        assert predict.shape == labels.shape

        return torch.mean(torch.abs(predict - labels)).item()
    
    