# Week 3: Transformer 改进技术

## 📋 概述

本week探索Transformer的现代改进技术，包括KV-Cache、RoPE位置编码等，并实战大模型量化部署。

## 🏗️ 模块结构

```
week3_improved/
├── kv_cache.py           # KV-Cache实现
├── rope.py               # RoPE旋转位置编码
├── attention.py          # 改进的注意力
├── multihead_attention.py
├── transformer_kv.py     # 带KV-Cache的Transformer
├── char_lm_kv.py         # KV-Cache语言模型
├── char_lm_rope.py       # RoPE语言模型
├── final.py              # Qwen2.5-7B量化实战
├── chat.py               # 对话接口
└── input.txt             # 训练数据
```

## 🔍 核心技术

### 1. KV-Cache (键值缓存)

**问题**: 自回归生成时，每次都要重新计算所有历史token的K、V

**解决**: 缓存历史的K、V，新token只需计算自己的Q、K、V

```python
# 无缓存: O(T²)
for step in range(max_len):
    output = model(all_tokens)  # 每次处理全部

# 有缓存: O(T)
cache = None
for step in range(max_len):
    output, cache = model(new_token, cache)  # 只处理新token
```

**加速效果**: 10-100倍

### 2. RoPE (旋转位置编码)

**问题**: 绝对位置编码无法外推到更长序列

**解决**: 通过旋转矩阵编码相对位置

```python
def apply_rope(x, pos):
    # 旋转角度
    theta = 10000^(-2i/d)
    angle = pos * theta
    
    # 复数旋转
    x_rot = x * cos(angle) + rotate(x) * sin(angle)
    return x_rot
```

**优势**:
- 支持长度外推
- 相对位置感知
- 无需学习参数

### 3. 4-bit 量化

**问题**: 7B模型FP16需要14GB显存

**解决**: 4-bit量化后仅需~4GB

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",      # Normal Float 4
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True  # 嵌套量化
)
```

## 🚀 运行示例

```bash
cd week3_improved

# 测试KV-Cache
python kv_cache.py

# 测试RoPE
python rope.py

# 训练字符级模型
python char_lm_kv.py

# Qwen2.5-7B量化实战 (需要GPU)
python final.py
```

## 📊 性能对比

| 技术 | 无优化 | 有优化 | 提升 |
|------|--------|--------|------|
| KV-Cache | 2.5s | 0.08s | 31x |
| RoPE外推 | 512 tokens | 4096+ tokens | 8x+ |
| 4-bit量化 | 14GB | 4GB | 3.5x |

## 💡 KV-Cache 原理图

```
Step 1: 输入 "Hello"
  Q1, K1, V1 = model("Hello")
  Cache: [(K1, V1)]
  
Step 2: 输入 "World"
  Q2, K2, V2 = model("World")
  K = concat(K1, K2)  # 使用缓存
  V = concat(V1, V2)
  Output = Attention(Q2, K, V)
  Cache: [(K1, V1), (K2, V2)]
```

## 🔧 环境要求

- Python 3.12+
- PyTorch 2.6+
- transformers (用于final.py)
- bitsandbytes (用于量化)
- CUDA 12.4 (量化需要GPU)

## 📚 参考资料

- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [The KV-Cache: Understanding the Memory Bottleneck](https://kipp.ly/blog/transformer-inference-arithmetic/)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
