import os
import torch
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
import pickle
from tqdm import tqdm
import numpy as np


class BaseSortModel(nn.Module):
    def __init__(self):
        super().__init__()


    def binary_cross_entropy_loss(self, logit, label):
        return -(label * torch.log(logit + 1e-6) + (1 - label) * torch.log(1 - logit + 1e-6))
    