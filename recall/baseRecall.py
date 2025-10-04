import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

import pyjson5 as json

class BaseRecall(L.LightningModule):
    def __init__(self, config_path: str):
        super(BaseRecall, self).__init__()
        with open(config_path, 'r') as f:
            self.config = json.load(f)
    
    def 