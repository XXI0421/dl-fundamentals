import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

# ========== 0. KVCache 类（保持不变）==========
class KVCache:
    def __init__(self, num_layers, batch_size, num_heads, max_seq_len, d_k, dtype=torch.float32):
        self.num_layers = num_layers
        self.current_len = 0
        self.cache_k = torch.zeros(num_layers, batch_size, num_heads, max_seq_len, d_k, dtype=dtype)
        self.cache_v = torch.zeros(num_layers, batch_size, num_heads, max_seq_len, d_k, dtype=dtype)
        if torch.cuda.is_available():
            self.cache_k = self.cache_k.cuda()
            self.cache_v = self.cache_v.cuda()
    
    def update(self, layer_idx, new_k, new_v):
        self.cache_k[layer_idx, :, :, self.current_len, :] = new_k.squeeze(2)
        self.cache_v[layer_idx, :, :, self.current_len, :] = new_v.squeeze(2)
        return (self.cache_k[layer_idx, :, :, :self.current_len+1, :],
                self.cache_v[layer_idx, :, :, :self.current_len+1, :])
    
    def increment(self):
        self.current_len += 1

# ========== 1. 超参数（可调）==========
d_model = 256      # 可增大到 256 提升效果
num_heads = 4
num_layers = 4     # 可增大到 4
d_ff = 1028
seq_len = 64
batch_size = 32
epochs = 50       
lr = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 2. 数据加载（本地文件）==========
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
char2idx = {ch: i for i, ch in enumerate(chars)}
idx2char = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [char2idx[c] for c in s]
decode = lambda l: ''.join([idx2char[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

def get_batch():
    """获取随机批次的训练数据"""
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix])
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
    return x.to(device), y.to(device)

# ========== 3. Attention 层（支持 Cache）==========
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        
    def forward(self, x, kv_cache=None, layer_idx=0):
        B, T, C = x.size()
        
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        
        q = q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        
        if kv_cache is not None:
            k, v = kv_cache.update(layer_idx, k, v)
        
        # Attention 计算
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 训练时需要因果掩码，推理时用 cache 天然因果
        if kv_cache is None:
            mask = torch.tril(torch.ones(T, T)).view(1, 1, T, T).to(device)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return out, attn_weights

# ========== 4. Transformer Block & Model（同上）==========
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, kv_cache=None, layer_idx=0):
        attn_out, _ = self.attn(x, kv_cache=kv_cache, layer_idx=layer_idx)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.ff(x))
        return x

class CharTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, targets=None, kv_cache=None):
        B, T = x.size()
        tok_emb = self.token_embed(x)
        pos_emb = self.pos_embed(torch.arange(T, device=x.device))
        x = tok_emb + pos_emb
        
        for layer_idx, block in enumerate(self.blocks):
            x = block(x, kv_cache=kv_cache, layer_idx=layer_idx)
            
        x = self.norm(x)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss

# ========== 5. 生成函数（使用 KV-Cache）==========
@torch.no_grad()
def generate(model, prompt, max_new_tokens=200):
    model.eval()
    input_ids = torch.tensor(encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    
    # 初始化 Cache
    d_k = d_model // num_heads
    kv_cache = KVCache(num_layers, 1, num_heads, max_seq_len=512, d_k=d_k)
    
    generated = []
    
    # 预热 prompt
    for i in range(input_ids.size(1)):
        logits, _ = model(input_ids[:, i:i+1], kv_cache=kv_cache)
        kv_cache.increment()
    
    # 取最后一个预测
    probs = torch.softmax(logits[:, -1, :], dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    generated.append(next_token.item())
    
    # 继续生成
    for _ in range(max_new_tokens - 1):
        logits, _ = model(next_token, kv_cache=kv_cache)
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated.append(next_token.item())
        kv_cache.increment()
        
        if next_token.item() == char2idx.get('\n', 0):
            break
    
    return prompt + decode(generated)

# ========== 6. 主程序：训练 + 生成 ==========
if __name__ == "__main__":
    model = CharTransformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"开始训练 {epochs} 个 epoch...\n")
    
    # 训练循环（关键：训练时不用 KV-Cache，生成时用）
    for epoch in range(epochs):
        model.train()
        losses = []
        
        for _ in range(100):  # 每 epoch 100 个 batch
            xb, yb = get_batch()
            logits, loss = model(xb, yb)  # 训练时不传 kv_cache
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        avg_loss = sum(losses) / len(losses)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")
        
        # 每 5 个 epoch 生成一次看效果
        if (epoch + 1) % 5 == 0:
            print("\n生成测试:")
            result = generate(model, "ROMEO: ", max_new_tokens=100)
            print(result[:200] + "...\n")
    
    # 最终生成
    print("\n训练完成！最终生成:")
    final = generate(model, "The king: ", max_new_tokens=300)
    print(final)
