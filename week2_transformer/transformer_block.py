import torch
import torch.nn as nn
from multihead_attention import MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        
        # 1. Multi-Head Attention（已验证）
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # 2. LayerNorm（刚学的：纵向归一化）
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # 3. Dropout（正则化）
        self.dropout = nn.Dropout(dropout)
        
        # 4. FeedForward（升维-激活-降维）
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), # 升维: 512 -- 2048
            nn.ReLU(),                # 非线性
            nn.Dropout(dropout),        
            nn.Linear(d_ff, d_model),  # 降维
            nn.Dropout(dropout)
        )
        
        
    def forward(self, x, mask=None):
        """
        x: (batch, seq, d_model)
        """
        # 子层1: Self-Attention + 残差 + LayerNorm
        # Self-Attention
        attn_out, attn_weights = self.attention(x, x, x, mask)
        
        # 残差连接：x + Dropout(attn_out)
        # LayerNorm
        x = self.norm1(x + self.dropout(attn_out))
        
        # 子层2: FeedForward + 残差 + LayerNorm
        # FeedForward
        ff_out = self.ff(x)
        
        # 残差连接：x + FFN(x)
        # LayerNorm
        x = self.norm2(x + ff_out)
       
        
        return x, attn_weights

# 测试参数
batch, seq_len, d_model = 2, 10, 512

# 创建输入
x = torch.randn(batch, seq_len, d_model)

# 初始化
block = TransformerBlock(d_model=d_model, num_heads=8, d_ff=2048).eval()

# 前向传播
with torch.no_grad():
    output, attn_weights = block(x)

# 检查点 1: 输出形状（必须和输入相同，这是残差连接保证的）
assert output.shape == (batch, seq_len, d_model)
print(f"✅ 输出形状正确: {output.shape}")

# 检查点 2: 残差连接验证（输出不应与输入相差太远）
diff = (output - x).abs().mean()
print(f"✅ 残差连接正常，平均变化: {diff:.4f}（不应为0，也不应太大）")

# 检查点 3: 注意力权重形状
assert attn_weights.shape == (batch, 8, seq_len, seq_len)
print(f"✅ 注意力权重可访问: {attn_weights.shape}")

print("\n全部测试通过！TransformerBlock 组装完成！")
print("准备 Day 7：训练字符级语言模型")