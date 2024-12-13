import torch
from torch import nn
from torchsummary import summary  # 神经网络模型的结构和参数信息

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()  # 调用父类的初始化方法
        # 定义 LeNet 的结构
        # stride 参数默认值为：1, padding 参数默认值为：0 可以省略
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)  # 卷积层
        self.sig = nn.Sigmoid()  # 激活函数 Sigmoid 是一种经典的非线性激活函数，将输出值映射到 [0,1]。
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)  # 平均池化层
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)  # 卷积层
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)  # 平均池化层

        self.flatten = nn.Flatten()  # 展平
        self.f5 = nn.Linear(in_features=400, out_features=120)  # 全连接层
        self.f6 = nn.Linear(in_features=120, out_features=84)  # 全连接层
        self.f7 = nn.Linear(in_features=84, out_features=10)  # 全连接层，输出特征数为 10（对应 10 个类别）

    # 前向传播
    def forward(self, x):  # 定义数据通过模型的计算流程
        x = self.sig(self.c1(x))
        x = self.s2(x)
        x = self.sig(self.c3(x))
        x = self.s4(x)
        x = self.flatten(x)
        x = self.f5(x)
        x = self.f6(x)
        x = self.f7(x)
        return x

# 测试
if __name__ == "__main__":
    # 判断是否有 GPU (CUDA) 可用，如果有则使用 GPU，否则使用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device) --> cuda

    # 实例化 LeNet 模型，并将其移动到指定设备
    model = LeNet().to(device)

    # 显示模型的结构和参数信息
    print(summary(model, input_size=(1, 28, 28)))

