from DSSM_model import DSSM
from torch.utils.data import DataLoader
import torch
import lightning as L
import sys
sys.path.append('/Users/zhanghaoyang/Desktop/Movie_Recsys')
from DataReader.data_reader import DataReader



if __name__ == "__main__":
    # 配置文件路径
    config_path = '/Users/zhanghaoyang/Desktop/Movie_Recsys/feature.json'
    # 训练数据路径
    train_feature_path = '/Users/zhanghaoyang/Desktop/Movie_Recsys/FeatureFiles/train_ratings_features.txt'
    # 验证数据路径
    val_feature_path = '/Users/zhanghaoyang/Desktop/Movie_Recsys/FeatureFiles/test_ratings_features.txt'
    
    # 初始化DataReader
    train_dataset = DataReader(config_path, train_feature_path)
    val_dataset = DataReader(config_path, val_feature_path)
    
    # 创建DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)
    
    # 初始化DSSM模型
    model = DSSM(config_path)
    
    # 定义检查点回调，保存验证集上表现最好的模型
    checkpoint_callback = L.callbacks.ModelCheckpoint(
        monitor='val_loss',  # 监控验证集损失
        dirpath='./checkpoints',  # 模型保存路径
        filename='dssm-{epoch:02d}-{val_loss:.2f}',  # 模型文件名格式
        save_top_k=1,  # 只保存表现最好的模型
        mode='min'  # 监控指标越小越好
    )
    
    # 初始化Trainer
    trainer = L.Trainer(
        max_epochs=10,  # 最大训练轮数
        gpus=1 if torch.cuda.is_available() else 0,  # 使用GPU
        callbacks=[checkpoint_callback]  # 添加检查点回调
    )

    # 训练模型
    trainer.fit(model, train_dataloader, val_dataloader)