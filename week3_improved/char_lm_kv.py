import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import transformer
from ..week2_transformer.transformer_block import TransformerBlock
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