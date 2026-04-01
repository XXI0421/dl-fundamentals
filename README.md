# 深度学习基础训练项目

从CNN到Transformer，从基础到前沿，系统学习深度学习核心技术。

## 📚 课程大纲

| Week | 主题 | 核心技术 | 难度 |
|:----:|------|----------|:----:|
| [Week 1](./week1_cifar10/README.md) | CNN图像分类 | 卷积、池化、Dropout | ⭐ |
| [Week 2](./week2_transformer/README.md) | Transformer注意力 | Self-Attention、Multi-Head | ⭐⭐ |
| [Week 3](./week3_improved/README.md) | Transformer改进 | KV-Cache、RoPE、量化 | ⭐⭐⭐ |
| [Week 4](./week4_mingpt/README.md) | 字符级语言模型 | GPT、LLaMA优化、生成 | ⭐⭐⭐ |

## 🗺️ 学习路径

```
Week 1: CNN基础
    ↓ 理解神经网络训练流程
Week 2: Transformer
    ↓ 掌握注意力机制
Week 3: 现代优化
    ↓ 学习KV-Cache、RoPE等
Week 4: 完整项目
    ↓ 实现字符级GPT
```

## 📁 项目结构

```
dl_fundamentals/
├── week1_cifar10/          # CNN图像分类
│   ├── train.py           # 训练脚本
│   └── README.md
│
├── week2_transformer/      # Transformer基础
│   ├── attention.py       # 注意力机制
│   ├── multihead_attention.py
│   ├── transformer_block.py
│   └── README.md
│
├── week3_improved/         # Transformer改进
│   ├── kv_cache.py        # KV-Cache
│   ├── rope.py            # RoPE位置编码
│   ├── final.py           # Qwen量化实战
│   └── README.md
│
├── week4_mingpt/           # 字符级GPT
│   ├── mingpt/            # 核心库
│   │   ├── model.py       # 原始GPT
│   │   ├── model_top_p.py # Top-p采样
│   │   └── im_model.py    # LLaMA风格
│   ├── projects/chargpt/  # 训练脚本
│   ├── forecast*.py       # 交互脚本
│   └── README.md
│
└── README.md              # 本文件
```

## 🚀 快速开始

### 环境配置

```bash
# 创建conda环境
conda create -n ai python=3.12
conda activate ai

# 安装PyTorch (CUDA 12.4)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 安装其他依赖
pip install transformers bitsandbytes
```

### 运行示例

```bash
# Week 1: 训练CNN
cd week1_cifar10 && python train.py

# Week 2: 测试注意力
cd week2_transformer && python attention.py

# Week 3: KV-Cache测试
cd week3_improved && python kv_cache.py

# Week 4: 训练GPT
cd week4_mingpt && python projects/chargpt/chargpt_im.py
```

## 📊 各Week核心成果

### Week 1: CNN + CIFAR-10
- ✅ 实现SimpleCNN，准确率~96%
- ✅ 理解卷积、池化、Dropout
- ✅ 掌握PyTorch训练流程

### Week 2: Transformer注意力
- ✅ 从零实现Scaled Dot-Product Attention
- ✅ 实现Multi-Head Attention
- ✅ 完成Transformer Block

### Week 3: Transformer改进
- ✅ KV-Cache加速10-100倍
- ✅ RoPE支持长度外推
- ✅ 4-bit量化部署7B模型

### Week 4: 字符级GPT
- ✅ 三种模型变体实现
- ✅ LLaMA风格优化(RMSNorm+RoPE+SwiGLU+KV-Cache)
- ✅ 交互式文本生成

## 🔧 环境要求

| 组件 | 版本 | 用途 |
|------|------|------|
| Python | 3.12+ | 运行环境 |
| PyTorch | 2.6+ | 深度学习框架 |
| CUDA | 12.4 | GPU加速(可选) |
| transformers | 4.40+ | Week 3量化实战 |
| bitsandbytes | 0.43+ | Week 3量化实战 |

## 💡 学习建议

1. **按顺序学习**: Week 1→2→3→4，循序渐进
2. **动手实践**: 每个模块都有可运行的代码
3. **阅读源码**: 理解实现细节比调API更重要
4. **修改实验**: 尝试修改参数、添加功能

## 📚 推荐资源

### 论文
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer原论文
- [RoFormer](https://arxiv.org/abs/2104.09864) - RoPE位置编码
- [LLaMA](https://arxiv.org/abs/2302.13971) - 开源大模型

### 博客
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [minGPT by Karpathy](https://github.com/karpathy/minGPT)
- [KV-Cache Arithmetic](https://kipp.ly/blog/transformer-inference-arithmetic/)

## 📝 更新日志

- **2026.04.02**: Week 4 完成，实现LLaMA风格优化
- **2026.03.29**: Week 3 完成，添加KV-Cache和RoPE
- **2026.03.26**: Week 2 完成，实现Transformer组件
- **2026.03.26**: Week 1 完成，CNN图像分类

## 📄 License

MIT License - 自由使用和学习
