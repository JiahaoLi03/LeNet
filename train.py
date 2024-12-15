from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt

from model import LeNet

# 加载并预处理数据，同时返回训练集和验证集的数据加载器。
def train_val_data_process():
    train_data = FashionMNIST(root='./data',
                              train=True,
                              transform=transforms.Compose([
                                  transforms.Resize((28, 28)),
                                  transforms.ToTensor()
                              ]),
                              download=True)

    """
        划分训练集与验证集
        Data.random_split: 按指定比例随机划分数据集
        参数 [round(0.8 * len(train_data)), round(0.2 * len(train_data))]:
            将训练数据集按比例划分为 80% 的训练数据和 20% 的验证数据
            通过 round 确保划分后的数据集大小为整数
    """
    train_data, val_data = Data.random_split(train_data, [round(0.8 * len(train_data)), round(0.2 * len(train_data))])

    # 创建训练集的 DataLoader
    train_dataloader = Data.DataLoader(dataset=train_data,
                                       batch_size=128,  # 每次从训练集中提取 128 条样本（小批量处理）
                                       shuffle=True,
                                       num_workers=8)

    val_dataloader = Data.DataLoader(dataset=val_data,
                                     batch_size=128,
                                     shuffle=True,
                                     num_workers=8)

    return train_dataloader, val_dataloader


