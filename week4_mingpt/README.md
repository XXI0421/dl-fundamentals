# Week 4: minGPT 字符级语言模型

## 📋 概述

本week基于minGPT实现完整的字符级语言模型，包含三种模型变体和LLaMA风格优化。

## 🏗️ 项目结构

```
week4_mingpt/
├── mingpt/                    # 核心库
│   ├── model.py              # 原始GPT模型
│   ├── model_top_p.py        # 支持Top-p采样的GPT
│   ├── im_model.py           # LLaMA风格优化模型
│   ├── trainer.py            # 基础训练器
│   ├── im_trainer.py         # 增强训练器
│   ├── bpe.py                # BPE分词器
│   └── utils.py              # 工具函数
│
├── projects/                  # 应用项目
│   └── chargpt/              # 字符级语言模型
│       ├── chargpt_model.py      # 训练原始model
│       └── chargpt_im.py         # 训练im_model
│
├── forecast.py               # model交互脚本
├── forecast_top_p.py         # model_top_p交互脚本
├── forecast_im.py            # im_model交互脚本
└── input.txt                 # 训练数据(莎士比亚)
```

## 🤖 三种模型对比

| 模型 | 特性 | 文件 | 保存路径 |
|------|------|------|----------|
| **model** | 原始GPT | model.py | model.pt |
| **model_top_p** | +Top-p采样 | model_top_p.py | model_top_p.pt |
| **im_model** | +RMSNorm+RoPE+SwiGLU+KV-Cache | im_model.py | im_model.pt |

## 🔍 im_model 架构 (LLaMA风格)

```
im_model 改进:
├── LayerNorm → RMSNorm      # 去均值，无bias，速度↑15%
├── wpe → RoPE               # 旋转位置编码，支持长度外推
├── GELU-MLP → SwiGLU        # 门控机制，表达能力↑
└── 全量注意力 → KV-Cache     # 生成速度↑10-100倍
```

### 核心改进代码

```python
# 1. RMSNorm (vs LayerNorm)
class RMSNorm(nn.Module):
    def forward(self, x):
        rms = sqrt(mean(x^2) + eps)
        return weight * x / rms  # 无bias，无均值

# 2. RoPE (vs 绝对位置编码)
def apply_rope(q, k, offset=0):
    pos = arange(offset, offset + seq_len)
    angles = pos * 10000^(-2i/d)
    return rotate(q, angles), rotate(k, angles)

# 3. SwiGLU (vs GELU-MLP)
class SwiGLU(nn.Module):
    def forward(self, x):
        return W3(silu(W1(x)) * W2(x))  # 门控

# 4. KV-Cache (vs 全量计算)
def forward(x, cache=None):
    if cache:
        k = cat(cache.k, new_k)  # 拼接历史
        v = cat(cache.v, new_v)
    return output, (k, v)
```

## 🚀 快速开始

### 训练模型

```bash
cd week4_mingpt

# 训练原始model
python projects/chargpt/chargpt_model.py

# 训练im_model (LLaMA风格)
python projects/chargpt/chargpt_im.py
```

### 交互生成

```bash
# 使用原始model
python forecast.py

# 使用model_top_p (支持Top-p采样)
python forecast_top_p.py

# 使用im_model (LLaMA风格，支持KV-Cache加速)
python forecast_im.py
```

## 📊 训练配置

| 参数 | 值 |
|------|-----|
| 模型类型 | gpt-mini |
| 层数 | 6 |
| 注意力头数 | 6 |
| 嵌入维度 | 192 |
| 上下文长度 | 128 |
| 学习率 | 5e-4 |
| 批大小 | 512 |

## 🎯 生成示例

```
Prompt: O God, O God!

生成结果:
O God, O God! I cannot tell
The reason of this strange unrest
That fills my soul with sudden dread
And makes my heart beat in my breast...
```

## 💡 关键技术详解

### 1. 因果掩码 (Causal Mask)

```python
# Q[i] 只能看到 K[j] where j <= i
mask = torch.tril(torch.ones(T, T))
att = att.masked_fill(mask == 0, -inf)
```

### 2. Top-p (Nucleus) 采样

```python
# 从累积概率p的最小集合中采样
sorted_probs, sorted_idx = sort(probs, descending=True)
cumsum = cumsum(sorted_probs)
mask = cumsum > p
logits[mask] = -inf
```

### 3. 温度衰减

```python
# 逐渐降低温度，从创意走向确定
temp = initial_temp * (decay ** step)
logits = logits / temp
```

## 🔧 环境要求

- Python 3.12+
- PyTorch 2.6+
- CUDA 12.4 (可选)

## 📚 参考资料

- [minGPT by Andrej Karpathy](https://github.com/karpathy/minGPT)
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)

## 🐛 已修复的问题

### KV-Cache因果掩码Bug

**问题**: 原实现中因果掩码未考虑offset，导致KV-Cache模式下Q只能看到K[0]

**修复**: 正确计算Q和K的位置关系

```python
# 修复前 (错误)
mask = torch.tril(torch.ones(T, total_len))  # 未考虑offset

# 修复后 (正确)
q_pos = torch.arange(offset, offset + T).view(T, 1)
k_pos = torch.arange(total_len).view(1, total_len)
mask = (q_pos >= k_pos).float()  # Q[i]能看到K[0:i+offset]
```
