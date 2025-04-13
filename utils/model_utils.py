import torch
import torch.nn as nn


def fc_model(dims, use_relu=True):
    model = nn.ModuleList()
    for i in range(len(dims)-2):
        model.append(nn.Linear(dims[0], dims[1]))
        model.append(nn.ReLU())
    model.append(nn.Linear(dims[-2], dims[-1]))
    return model