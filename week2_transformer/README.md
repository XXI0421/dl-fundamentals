# Week 2: Transformer 注意力机制

## 📋 概述

本week从零实现Transformer的核心组件，深入理解注意力机制的工作原理。

## 🏗️ 模块结构

```
week2_transformer/
├── attention.py           # 缩放点积注意力
├── multihead_attention.py # 多头注意力
├── transformer_block.py   # Transformer块
└── char_lm.py            # 字符级语言模型应用
```

## 🔍 核心组件

### 1. Scaled Dot-Product Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d\_k}}\right)V$$

```python
# 核心计算流程
scores = Q @ K.T / sqrt(d_k)    # 缩放点积
attn = softmax(scores)           # 注意力权重
output = attn @ V                # 加权求和
```

### 2. Multi-Head Attention

$$\text{MultiHead}(Q,K,V) = \text{Concat}(head\_1, ..., head\_h)W^O$$

```python
# 多头并行计算
heads = [Attention(Q_i, K_i, V_i) for i in range(h)]
output = Concat(heads) @ W_O
```

### 3. Transformer Block

```
TransformerBlock:
├── Multi-Head Attention
├── Add & LayerNorm
├── Feed Forward (MLP)
└── Add & LayerNorm
```

## 🚀 运行示例

```bash
cd week2_transformer

# 测试注意力机制
python attention.py

# 测试多头注意力
python multihead_attention.py

# 测试Transformer块
python transformer_block.py

# 运行字符级语言模型
python char_lm.py
```

## 📊 关键概念

| 概念        | 说明             |
| --------- | -------------- |
| Query (Q) | 查询向量，表示"我要找什么" |
| Key (K)   | 键向量，表示"我是什么"   |
| Value (V) | 值向量，表示"我的内容"   |
| d\_k      | 向量维度，用于缩放      |
| Mask      | 因果掩码，防止看到未来    |

## 💡 为什么需要缩放？

当 $d\_k$ 很大时，点积结果会很大，导致softmax进入饱和区，梯度消失：

```python
# 不缩放的问题
scores = Q @ K.T  # 可能很大
softmax(scores)   # 输出接近one-hot，梯度≈0

# 缩放后
scores = Q @ K.T / sqrt(d_k)  # 数值稳定
softmax(scores)               # 梯度正常
```

## 🔧 环境要求

- Python 3.12+
- PyTorch 2.6+
- math (标准库)

## 📚 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

