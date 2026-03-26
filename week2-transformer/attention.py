import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # 如果需要，可以在这里注册softmax，但也可以用函数式
    
    def forward(self, query, key, value, mask=None):
        """
        输入:
            query: (batch, seq_len, d_k)
            key: (batch, seq_len, d_k)  
            value: (batch, seq_len, d_v)
            mask: (batch, 1, seq_len) 或 None，用于屏蔽某些位置
        输出:
            output: (batch, seq_len, d_v)
            attn_weights: (batch, seq_len, seq_len) 用于可视化
        """
        d_k = query.size(-1)
        
        # Step 1: 计算 Q @ K^T
        # 提示: key.transpose(-2, -1) 把最后两维转置
        scores = torch.matmul(query, key.transpose(-2, -1))
        
        # Step 2: 缩放（关键！）
        # 提示: math.sqrt(d_k)
        scores = scores / math.sqrt(d_k)
        
        # Step 3: Mask 屏蔽无效位置（如果有）
        # 提示: masked_fill(mask == 0, -1e9)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Step 4: Softmax + Dropout
        # 提示: dim=-1 表示在最后一个维度做softmax
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Step 5: 加权求和（注意力权重 @ V）
        output = torch.matmul(attn_weights, value)
        
        return output, attn_weights


batch, seq_len, d_k = 2, 10, 64
Q = torch.randn(batch, seq_len, d_k)
K = torch.randn(batch, seq_len, d_k)
V = torch.randn(batch, seq_len, d_k)

# 初始化
attn = ScaledDotProductAttention()
attn.eval()

# 前向传播
out, weights = attn(Q, K, V)

# 检查点 1: 形状
assert out.shape == (batch, seq_len, d_k), f"输出形状错误: {out.shape}"
assert weights.shape == (batch, seq_len, seq_len), f"权重形状错误: {weights.shape}"
print("✅ 形状检查通过")

# 检查点 2: 注意力权重每行和为1
row_sum = weights[0, 0].sum()
assert abs(row_sum.item() - 1.0) < 1e-6, f"权重和不等于1: {row_sum}"
print(f"✅ 权重归一化检查通过: {row_sum:.6f}")

print("全部测试通过！")