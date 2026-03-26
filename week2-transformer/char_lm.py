import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_block import TransformerBlock
import requests

# ========== 1. 超参数（小模型配置）==========
d_model = 512
num_heads = 4
num_layers = 4
d_ff = 2048      # 4*d_model
seq_len = 64
batch_size = 32
epochs = 30
lr = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 2. 数据加载（tiny-shakespeare）==========
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
text = requests.get(url).text

# 字符级 Tokenizer
chars = sorted(list(set(text)))
vocab_size = len(chars)
char2idx = {ch: i for i, ch in enumerate(chars)}
idx2char = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [char2idx[c] for c in s]
decode = lambda l: ''.join([idx2char[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

# ========== 3. 完整 Transformer 模型 ==========
class CharTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(seq_len, d_model)  # 位置编码
        
        # TODO: 堆叠 num_layers 个 TransformerBlock
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff) 
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, targets=None):
        """
        x: (batch, seq_len) 的字符索引
        targets: (batch, seq_len) 用于计算loss
        """
        b, t = x.size()
        
        # Token + 位置嵌入
        tok_emb = self.token_embed(x)  # (b, t, d_model)
        pos_emb = self.pos_embed(torch.arange(t, device=device))  # (t, d_model)
        x = tok_emb + pos_emb
        
        # TODO: 创建因果掩码（Causal Mask）
        # 关键：上三角矩阵，防止看到未来信息
        mask = torch.tril(torch.ones(t, t)).view(1, 1, t, t).to(device)
        
        # 通过 Transformer Blocks
        attn_weights = None
        for block in self.blocks:
            x, attn_weights = block(x, mask)
        
        x = self.norm(x)
        logits = self.head(x)  # (b, t, vocab_size)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        
        return logits, loss, attn_weights

# ========== 4. 训练准备 ==========
model = CharTransformer().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# 数据加载器（简单滑动窗口）
def get_batch():
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix])
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix])  # 预测下一个字符
    return x.to(device), y.to(device)

# ========== 5. 训练循环 ==========
print(f"参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
print(f"词汇表大小: {vocab_size}, 文本长度: {len(text)}")

for epoch in range(epochs):
    model.train()
    losses = []
    for _ in range(100):  # 每epoch迭代100次
        xb, yb = get_batch()
        logits, loss, _ = model(xb, yb)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
    
    avg_loss = sum(losses)/len(losses)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # 生成测试
    if epoch % 2 == 0:
        model.eval()
        context = torch.zeros((1, 1), dtype=torch.long).to(device)
        # TODO: 实现生成函数（提示：循环预测下一个字符）
        with torch.no_grad():
            for _ in range(200):  # 生成200个字符
                logits, _, _ = model(context[:, -seq_len:])  # 截断到seq_len
                probs = F.softmax(logits[:, -1, :], dim=-1)  # 取最后一个时间步
                next_char = torch.multinomial(probs, num_samples=1)
                context = torch.cat([context, next_char], dim=1)
        
        print(f"生成: {decode(context[0].tolist())}\n")

print("训练完成！")
