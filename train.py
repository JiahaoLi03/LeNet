import torch
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import copy
import time

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

# 训练 num_epochs: 训练轮次
def train_model_process(model, train_dataloader, val_dataloader, num_epochs):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 定义优化器 使用 Adam 优化器来调整参数
    # model.parameters(): 传入模型的参数   lr: 学习率
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 定义损失函数（交叉熵损失），适用于分类任务
    criterion = nn.CrossEntropyLoss()

    # 将模型移动到指定设备
    model = model.to(device)

    # 保存模型权重 使用 copy.deepcopy 创建模型权重的深拷贝，用于保存最佳模型的权重
    beat_model_wts = copy.deepcopy(model.state_dict())

    # 初始化参数
    # 最高准确度
    best_ac = 0.0

    # 训练集 loss 值列表
    train_loss_all = []

    # 验证集 loss 值列表
    val_loss_all = []

    # 训练集准确度列表
    train_acc_all = []

    # 验证集准确度列表
    val_acc_all = []

    # 当前时间
    since = time.time()

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print('-' * 15)

        # 本轮训练集损失值
        train_loss = 0.0
        # 本轮训练集准确度
        train_corrects = 0.0

        # 本轮验证集损失值
        val_loss = 0.0
        # 本轮验证集准确度
        val_corrects = 0.0

        # 本轮的训练集样本数量
        train_num = 0
        # 本轮的验证集样本数量
        val_num = 0

        for step, (b_x, b_y) in enumerate(train_dataloader):
            # print(b_x) --> 128 个 Tensor
            # print(b_x.size()) -- >torch.Size([128, 1, 28, 28])

            # print(b_y) --> 128张图像对应的类别序号 标签
            # print(b_y.size()) --> torch.Size([128])

            # 将数据移动到指定设备
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            # 切换模型到 训练模式
            model.train()

            # 前向传播
            output = model(b_x)
            # print("output = ", output) 128 x 10 每一张图像 每个类别的概率
            # print("output.size() = ", output.size()) --> torch.Size([128, 10])

            # 预测类别：使用 torch.argmax 从 output 中找到预测类别（最大值对应的索引）
            pre_lab = torch.argmax(output, dim=1)
            # print("pre_lab = ", pre_lab) 预测每张图像对应的类别序号 即 output 中每一个图像对应的 10 个类别最大值对应的下标
            # print("pre_lab.size() = ", pre_lab.size()) --> torch.Size([128])

            # 计算模型预测结果 output 与真实标签 b_y 之间的损失值
            loss = criterion(output, b_y)

            # 梯度清零：防止之前计算的梯度影响当前训练
            optimizer.zero_grad()

            # 反向转播：计算损失函数相对于模型参数的梯度
            loss.backward()

            # 更新模型参数：根据计算出的梯度和学习率调整模型的权重
            optimizer.step()

            """
                累加损失值
                loss.item(): 当前批次的损失值（标量）
                b_x.size(0): 当前批次的样本数量
                将损失值乘以样本数后加到总损失 train_loss 中
            """
            # print("loss.item() = ", loss.item()) -->  0.7834932208061218
            # print("b_x.size(0) = ",b_x.size(0)) --> b_x.size(0) =  128
            train_loss += loss.item() * b_x.size(0)

            """
                累积准确样本数
                比较预测类别 pre_lab 和真实标签 b_y.data
                统计预测正确的样本数量，并累加到 train_corrects
            """
            train_corrects += torch.sum(pre_lab == b_y.data)

            train_num += b_x.size(0)

        for step, (b_x, b_y) in enumerate(val_dataloader):

            b_x = b_x.to(device)
            b_y = b_y.to(device)

            model.eval()

            output = model(b_x)

            pre_lab = torch.argmax(output, dim=1)

            loss = criterion(output, b_y)

            val_loss += loss.item() * b_x.size(0)

            val_corrects += torch.sum(pre_lab == b_y.data)

            val_num += b_x.size(0)

        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)

        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects.double().item() / val_num)

        print("{} Train Loss: {:.4f}  Train Acc: {:.4f}".format(epoch + 1, train_loss_all[-1], train_acc_all[-1]))
        print("{}   Val Loss: {:.4f}    Val Acc: {:.4f}".format(epoch + 1, val_loss_all[-1], val_acc_all[-1]))











if __name__ == "__main__":
    # 判断是否有 GPU (CUDA) 可用，如果有则使用 GPU，否则使用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device) --> cuda

    # 实例化 LeNet 模型，并将其移动到指定设备
    model = LeNet().to(device)

    # 显示模型的结构和参数信息
    train_dataloader, val_dataloader = train_val_data_process()
    train_model_process(model, train_dataloader, val_dataloader, 1)








