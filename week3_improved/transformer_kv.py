import torch
import torch.nn as nn
import math
from multihead_attention import MultiHeadAttention, attn_weights

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
        
        
    def forward(self, x, kv_cache=None, layer_idx=0, mask=None):
        """
        x: (batch, seq, d_model)
        """
        # 子层1: Self-Attention + 残差 + LayerNorm
        # Self-Attention
        if kv_cache is not None:
            # TODO: 关键修改2：使用 cache 的增量 Attention
            # 需要把 x 拆分为：历史（来自cache）和当前（新计算）
            # 提示：当 cache 不为空时，x 只有 1 个 token（当前生成的）
            attn_out, attn_weights = self.attention_with_cache(x, kv_cache, layer_idx)
        else:
            # 训练时原逻辑
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

    def attention_with_cache(self, current_x, kv_cache, layer_idx):
        """
        增量 Attention：只计算当前 token 的 Q，复用历史的 K/V
        """
        B, T, C = current_x.size()  # T 应该为 1（当前token）

         # 计算当前 token 的 Q, K, V
        q = self.attention.W_Q(current_x).view(B, T, self.attention.num_heads, -1).transpose(1, 2)
        k_new = self.attention.W_K(current_x).view(B, T, self.attention.num_heads, -1).transpose(1, 2)
        v_new = self.attention.W_V(current_x).view(B, T, self.attention.num_heads, -1).transpose(1, 2)

        # TODO: 关键修改3：更新 cache 并获取完整的 K/V
        k_cache, v_cache = kv_cache.update(layer_idx, k_new, v_new)
        # 注意：k_cache 现在包含历史+新的，形状 (B, H, current_len, d_k)
        # 但 q 只有 (B, H, 1, d_k)，这是对的，我们只计算当前token与历史的Attention

        # 计算 Attention（q 与所有 k）
        scores = torch.matmul(q, k_cache.transpose(-2, -1)) / math.sqrt(q.size(-1))
        attn_weights = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v_cache)
        
        # 重塑回 (B, T, C)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
        attn_out = self.attention.W_O(attn_out)


        return attn_out, attn_weights
