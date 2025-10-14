from DSSM_model import DSSM
from torch.utils.data import DataLoader
import torch
import lightning as L
import sys
sys.path.append('/data2/zhy/Movie_Recsys')
from DataReader.data_reader import DataReader



if __name__ == "__main__":
    # 配置文件路径
    config_path = '/data2/zhy/Movie_Recsys/feature.json'
    # 训练数据路径
    train_feature_path = '/data2/zhy/Movie_Recsys/FeatureFiles/train_ratings_features.txt'
    # 验证数据路径
    val_feature_path = '/data2/zhy/Movie_Recsys/FeatureFiles/test_ratings_features.txt'
    
    # 初始化DataReader
    train_dataset = DataReader(config_path, train_feature_path)
    val_dataset = DataReader(config_path, val_feature_path)
    
    # 创建DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4)
    
    # 初始化DSSM模型
    model = DSSM(config_path)
    
    # 定义检查点回调，只保存模型，不做验证
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        save_top_k=-1,  # 保存所有epoch的模型
        every_n_epochs=1,  # 每个epoch都保存
        dirpath="./checkpoints",  # 保存路径
        filename="dssm-epoch{epoch}",  # 文件名格式
        save_weights_only=True
    )
    
    # 初始化Trainer
    trainer = L.Trainer(
        max_epochs=10,  # 最大训练轮数
        accelerator="gpu", # 使用GPU加速
        devices=[0], # 使用1块GPU
        callbacks=[checkpoint_callback]  # 添加检查点回调
    )

    # 训练模型
    trainer.fit(model, train_dataloader)