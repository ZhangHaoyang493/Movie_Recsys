import torch
import torch.nn as nn

# [16, 128, 64, 16]
def fc_model(dims, use_relu=True):
    model = []
    for i in range(len(dims)-2):
        model.append(nn.Linear(dims[i], dims[i+1]))
        model.append(nn.ReLU())
    model.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*model)