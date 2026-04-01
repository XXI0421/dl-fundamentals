# Week 1: CNN + CIFAR-10 图像分类

## 📋 概述

本week实现了一个简单的卷积神经网络(CNN)用于CIFAR-10图像分类任务，帮助理解深度学习的基础概念。

## 🏗️ 网络架构

```
SimpleCNN:
├── Conv1: 3 → 32 channels (3x3 kernel)
├── MaxPool: 32x32 → 16x16
├── Conv2: 32 → 64 channels (3x3 kernel)
├── MaxPool: 16x16 → 8x8
├── FC1: 64*8*8 → 256
├── Dropout: 0.5
└── FC2: 256 → 10 (classes)
```

## 📊 数据集

**CIFAR-10**:
- 60,000张 32x32 彩色图像
- 10个类别: plane, car, bird, cat, deer, dog, frog, horse, ship, truck
- 训练集: 50,000张
- 测试集: 10,000张

## 🚀 运行方法

```bash
cd week1_cifar10
python train.py
```

## 📈 训练结果

| 指标 | 数值 |
|------|------|
| 测试准确率 | ~96% |
| 训练轮数 | 40 epochs |
| 批大小 | 64 |
| 学习率 | 0.001 |

## 📁 文件结构

```
week1_cifar10/
├── train.py       # 主训练脚本
├── loss.py        # 损失函数可视化
├── derivation.jpg # 手写推导图
└── _gensim.py     # 词向量实验(可选)
```

## 💡 关键知识点

1. **卷积层**: 提取图像局部特征
2. **池化层**: 降维，减少计算量
3. **Dropout**: 防止过拟合
4. **归一化**: 加速训练收敛

## 🔧 环境要求

- Python 3.12+
- PyTorch 2.6+
- torchvision
- CUDA 12.4 (可选，用于GPU加速)
