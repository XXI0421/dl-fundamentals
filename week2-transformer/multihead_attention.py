import torch
import torch.nn as nn
import math
from attention import ScaledDotProductAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 64
        
        # 定义线性投影（不要定义8个独立Linear，用4个大Linear）
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)  # 输出投影
        
        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        """
        输入: (batch, seq, d_model)
        输出: (batch, seq, d_model), (batch, num_heads, seq, seq)
        """
        batch_size = query.size(0)
        
        # Step 1: 线性投影并分头
        # Q: (batch, seq, d_model) -> (batch, seq, num_heads, d_k) -> (batch, num_heads, seq, d_k)
        Q = self.W_Q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Step 2: 处理 mask（如果提供）
        # mask: (batch, 1, seq) -> (batch, 1, 1, seq) 广播到所有头
        if mask is not None:
            mask = mask.unsqueeze(1)
        
        # Step 3: 调用 ScaledDotProductAttention
        # 输出 x: (batch, num_heads, seq, d_k), attn: (batch, num_heads, seq, seq)
        x, attn = self.attention(Q, K, V, mask)
        
        # Step 4: 合并头（transpose + contiguous + view）
        # (batch, num_heads, seq, d_k) -> (batch, seq, num_heads, d_k) -> (batch, seq, d_model)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Step 5: 最终线性投影
        output = self.W_O(x)
        
        return output, attn

# 测试参数
batch, seq_len, d_model = 2, 10, 512
num_heads = 8

# 创建输入
x = torch.randn(batch, seq_len, d_model)

# 初始化
mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads).eval()

# 自注意力（Q=K=V=x）
with torch.no_grad():
    output, attn_weights = mha(x, x, x)

# 检查点 1: 输出形状
assert output.shape == (batch, seq_len, d_model), f"输出形状错误: {output.shape}"
print(f"✅ 输出形状正确: {output.shape}")

# 检查点 2: 注意力权重形状
assert attn_weights.shape == (batch, num_heads, seq_len, seq_len)
print(f"✅ 注意力权重形状正确: {attn_weights.shape}")

# 检查点 3: 权重归一化（每行和为1）
row_sum = attn_weights[0, 0, 0].sum()
assert abs(row_sum.item() - 1.0) < 1e-6
print(f"✅ 权重归一化检查通过: {row_sum:.6f}")

print("\n全部测试通过！准备进入 TransformerBlock")