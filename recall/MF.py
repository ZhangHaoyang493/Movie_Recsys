# 矩阵分解算法

import os
import torch
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L
from torch.utils.data import Dataset, DataLoader
import pickle

class MFDataset(Dataset):
    def __init__(self, readlist_path: str):
        super().__init__()

        readlist = pickle.load(open(readlist_path, 'rb'))
        self.data = []
        for user in readlist.keys():
            for item_info in readlist[user]:
                self.data.append(torch.tensor([int(user), int(item_info[0]), item_info[1]], dtype=torch.int))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
class MFModel(L.LightningModule):
    def __init__(self, user_num, item_num, dim: int=32):
        super().__init__()

        self.user_embedding = nn.Embedding(user_num, dim)
        self.item_embedding = nn.Embedding(item_num, dim)

    # 输入： bx3
    def forward(self, user_id, item_id):
        return self.user_embedding(user_id), self.item_embedding(item_id)
    
    def training_step(self, batch, batch_idx):
        user_id, item_id, label = batch[:, 0], batch[:, 1], batch[:, 2]
        user_emb, item_emb = self(user_id, item_id)

        loss = nn.MSELoss()(torch.mul(user_emb, item_emb).sum(dim=1), label * 1.0)

        return loss
    
    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=0.001, weight_decay=1e-4)
        return optimizer
    
    def save_user_embedding_weights(self, save_path):
        weights_user = self.user_embedding.weight.detach().cpu().numpy()
        weights_dict_user = {i: weights_user[i].tolist() for i in range(weights_user.shape[0])}
        weights_item = self.item_embedding.weight.detach().cpu().numpy()
        weights_dict_item = {i: weights_item[i].tolist() for i in range(weights_item.shape[0])}
        pickle.dump(weights_dict_user, open(os.path.join(save_path, 'MF_user_emb.pkl'), 'wb'))
        pickle.dump(weights_dict_item, open(os.path.join(save_path, 'MF_item_emb.pkl'), 'wb'))

if __name__ == '__main__':
    dataset = MFDataset(
        '/data/zhy/recommendation_system/Movie_Recsys/cache/train_readlist.pkl'
    )
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

    item_num = 3952 + 1
    user_num = 6040 + 1

    model = MFModel(user_num, item_num)

    trainer = L.Trainer(max_epochs=5, accelerator='cpu')#, accelerator='gpu', devices='1')
    trainer.fit(model, dataloader)

    model.save_user_embedding_weights('/data/zhy/recommendation_system/Movie_Recsys/cache')





