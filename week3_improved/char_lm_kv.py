import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_kv import TransformerBlock
from kv_cache import KVCache
import math, requests

# ========== 1. 超参数（小模型配置）==========
d_model = 128      
num_heads = 4
num_layers = 2    
d_ff = 512        
dropout = 0.2     
batch_size = 32
seq_len = 64
epochs = 20      
lr = 3e-4          
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 2. 数据加载 ==========
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 字符级 Tokenizer
chars = sorted(list(set(text)))
vocab_size = len(chars)
char2idx = {ch: i for i, ch in enumerate(chars)}
idx2char = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [char2idx[c] for c in s]
decode = lambda l: ''.join([idx2char[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

def get_batch():
    """获取训练批次"""
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix])
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
    return x.to(device), y.to(device)

# ========== 3. 完整 Transformer 模型 ==========
class CharTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(seq_len, d_model)
        
        # 堆叠 TransformerBlock
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff) 
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, targets=None, kv_cache=None, use_cache=False):
        """
        x: (batch, seq_len) 的字符索引
        targets: (batch, seq_len) 用于计算loss
        kv_cache: KVCache 实例
        use_cache: 是否使用缓存（训练时为False，生成时为True）
        """
        b, t = x.size()
        
        # Token + 位置嵌入
        tok_emb = self.token_embed(x)  # (b, t, d_model)
        pos_emb = self.pos_embed(torch.arange(t, device=device))  # (t, d_model)
        x = tok_emb + pos_emb
        
        # 通过 Transformer Blocks
        attn_weights = None
        for layer_idx, block in enumerate(self.blocks):
            if use_cache and kv_cache is not None:
                # 传入 cache 和 layer_idx
                x, attn_weights = block(x, kv_cache=kv_cache, layer_idx=layer_idx)
            else:
                # 训练时不用 cache
                x, attn_weights = block(x, mask=None)
        
        x = self.norm(x)
        logits = self.head(x)  # (b, t, vocab_size)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        

        return logits, loss, attn_weights

def generate(model, prompt, max_new_tokens=100, device='cuda'):
    """使用 KV-Cache 生成文本"""
    model.eval()
    
    # 编码 prompt
    input_ids = torch.tensor(encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    
    # 初始化 KV Cache
    num_layers = len(model.blocks)
    d_k = d_model // num_heads 
    kv_cache = KVCache(num_layers, batch_size=1, num_heads=num_heads, 
                       max_seq_len=512, d_k=d_k)
    
    generated = []
    
    # 预热：逐 token 处理 prompt
    with torch.no_grad():
        for i in range(input_ids.size(1)):
            logits, _, _ = model(input_ids[:, i:i+1], kv_cache=kv_cache, use_cache=True)
            kv_cache.increment()
        
        # 用最后一个 logits 生成第一个新 token
        temperature = 0.8  # 0.5=保守重复, 1.0=平衡, 1.5=混乱但有变化
        probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated.append(next_token.item())
        
        # 继续生成
        for _ in range(max_new_tokens - 1):
            logits, _, _ = model(next_token, kv_cache=kv_cache, use_cache=True)
            temperature = 0.8  # 0.5=保守重复, 1.0=平衡, 1.5=混乱但有变化
            probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated.append(next_token.item())
            kv_cache.increment()
            
    
    return prompt + decode(generated)

# ========== 4. 主程序 ==========
if __name__ == "__main__":
    model = CharTransformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"开始训练 {epochs} 个 epoch...\n")
    
    # 训练循环
    for epoch in range(epochs):
        model.train()
        losses = []
        
        for _ in range(100):  # 每 epoch 100 个 batch
            xb, yb = get_batch()
            logits, loss, _ = model(xb, yb, kv_cache=None, use_cache=False)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        avg_loss = sum(losses) / len(losses)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")
        
        print("\n生成测试:")
        result = generate(model, "ROMEO: ", max_new_tokens=100)
        print(result[:200] + "...\n")
    
    # 最终生成
    print("\n训练完成！最终生成:")
    final = generate(model, "The king: ", max_new_tokens=300)
    print(final)
