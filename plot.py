from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt

train_data = FashionMNIST(root='./data',  # 数据集的保存路径
                          train=True,  # 加载训练集
                          transform=transforms.Compose([  # 数据预处理
                              transforms.Resize((224, 224)),  # 调整图像大小 28x28 --> 224x224
                              transforms.ToTensor()  # 将图像转换为张量
                          ]),
                          download=True) # 如果数据集不存在，则下载

# 创建数据加载器
train_loader = Data.DataLoader(dataset=train_data,  # 训练数据
                               batch_size=64,  # 每个 batch 的样本数为 64
                               shuffle=True,  # 打乱数据顺序
                               num_workers=0)  # 数据加载的子线程数

# 获得一个 batch 数据
"""
    train_loader 是通过 DataLoader 加载的训练数据
    每次迭代都会返回一个批次的数据，格式为 (batch_images, batch_labels)
    batch_images (b_x): 当前批次的图像张量 形状为 (batch_size, channels, height, width)
    batch_labels (b_y): 当前批次的标签张量 形状为 (batch_size,)
    enumerate() 为每次迭代的批次编号分配一个索引 step
"""
for step, (b_x, b_y) in enumerate(train_loader):
    if step > 0:  # step 表示是当前的第几次迭代（即第几个 batch）
        break

# (b_x, b_y) 是当前批次的数据和对应的标签
print(b_x.shape) # torch.Size([64, 1, 224, 224])
print(b_y.shape) # torch.Size([64])
print(b_x)
print(b_y)

# squeeze(): 去掉张量中为 1 的维度 [64, 1, 224, 224] --> [64, 224, 224]
# 因为 channels=1 对灰度图来说并不需要，去掉后便于处理
# numpy(): 将 PyTorch 张量 b_x 转换为 NumPy 数组，便于后续使用 NumPy 或 Matplotlib 进行处理和可视化
batch_x = b_x.squeeze().numpy()
batch_y = b_y.numpy()

print(batch_x.shape)  # (64, 224, 224)
print(batch_y.shape)  # (64,)

print(batch_x)
# 每个元素是一个整数，表示图像所属类别的索引（从 0 到 9）
print(batch_y) # [3 0 7 0 7 5 4 3 1 2 4 0 6 4 7 5 1 0 3 3 8 1 0 0 3 7 2 8 0 7 1 5 9 1 8 4 9 3 8 4 3 9 6 0 9 3 4 9 3 8 7 6 0 5 1 6 5 4 2 1 7 6 0 2]

"""
    classes 是 torchvision.datasets.FashionMNIST 类中定义的一个静态属性
            是一个列表，存储了 FashionMNIST 数据集中所有类别的名称
            保存类别名称，用于将数字标签转换为可读的文字形式
"""
class_label = train_data.classes
print(class_label)

# 可视化一个 batch 的图像
# plt.figure(): 创建一个新的图像窗口，开始绘制图形
# figsize(): 参数定义图像窗口的宽和高，单位为英寸
plt.figure(figsize=(16, 8))

# 生成一个从 0 到 len(batch_y)-1 的整数序列
print(len(batch_y)) # 64

# np.arange() 创建一个数组，其内容为每个图像在 batch 中的索引号
# 例如：如果 batch_y 的长度是 64，那么 np.arange(len(batch_y)) 生成 [0, 1, 2, ..., 63]
for ii in np.arange(len(batch_y)):

    """
        plt.subplot()
        在当前图像窗口中创建一个子图
        参数含义：
            4: 子图分为  4 行
           16: 子图分为 16 列
           ii + 1: 子图的位置索引，从 1 开始（不是从 0 开始）
           将整个 batch 的图像排列为一个 4×16 的网格，每个小格子显示一张图片
    """
    plt.subplot(4, 16, ii + 1)

    """
        plt.imshow()
        绘制图像数据
        batch_x 是当前批次的图像数据，形状为 (batch_size, height, width)（灰度图，通道已移除）
        batch_x[ii, :, :]: 
            从 batch_x 中提取第 ii 个样本的图像数据，形状为 (height, width)
            对于 FashionMNIST，图像大小为 224×224
        cmap=plt.cm.gray
            设置图像的颜色映射为灰度
            因为 FashionMNIST 是灰度图，所以这里使用灰度映射更符合直观感受
    """
    plt.imshow(batch_x[ii, :, :], cmap=plt.cm.gray)

    """
        plt.title()
        设置当前子图的标题
        class_label[batch_y[ii]]
            batch_y[ii] 是第 ii 个样本的标签，数值为 0 到 9 的整数
            class_label 是类别名称列表
            size=10: 设置标题文字的字体大小为 10
            loc='center': 参数调整标题的位置为居中
    """
    plt.title(class_label[batch_y[ii]], size=10, loc='center')

    # 控制是否显示坐标轴 "off": 关闭当前子图的坐标轴，避免干扰视觉效果
    plt.axis("off")

    # plt.subplots_adjust(): 调整子图之间的间距
    # wspace：调整水平方向的间隔
    # hspace：调整竖直方向的间隔
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

plt.show()