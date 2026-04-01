"""
Full definition of a GPT Language Model with LLaMA-style optimizations.
LLaMA 风格优化版 GPT - Week 4 Day 4 架构魔改完成版

与原版 minGPT 的关键差异：
1. LayerNorm → RMSNorm (去均值，无 bias，速度↑15%)
2. 绝对位置编码(wpe) → RoPE旋转位置编码 (支持更长外推)
3. GELU-MLP → SwiGLU (门控机制，表达能力↑)
4. 全量注意力 → KV-Cache (生成速度↑10-100倍)

References:
1) LLaMA: Open and Efficient Foundation Language Models
2) RoFormer: Enhanced Transformer with Rotary Position Embedding
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from mingpt.utils import CfgNode as CN


# ====================【Day 4改造1: RMSNorm】====================
# 原版: nn.LayerNorm (减均值+除方差，有bias)
# 改造: RMSNorm (只除方差，无bias，省内存，适合深层)
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        # 注意：只有weight，没有bias（LayerNorm有weight+bias）
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # RMSNorm公式: x * weight / sqrt(mean(x^2) + eps)
        # 比LayerNorm少一次mean计算，且去掉了bias
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * x / rms


# ====================【Day 4改造2: RoPE旋转位置编码】====================
# 原版: 可学习绝对位置编码 nn.Embedding(block_size, n_embd)
# 改造: RoPE (通过旋转矩阵编码相对位置，支持长度外推)
def apply_rope(q, k, seq_len, head_dim, offset=0):
    """
    Rotary Position Embedding (RoPE)
    q, k: (B, n_head, T, head_dim)
    offset: 位置偏移量，用于KV-Cache时的历史长度
    返回: 旋转后的 q, k (保持shape不变)
    """
    # 生成位置索引 (考虑偏移量)
    pos = torch.arange(offset, offset + seq_len, device=q.device)
    
    # 生成旋转频率 (head_dim//2,)
    dim_idx = torch.arange(0, head_dim, 2, device=q.device).float()
    inv_freq = 1.0 / (10000 ** (dim_idx / head_dim))
    
    # 计算旋转角度 (seq_len, head_dim//2)
    angles = torch.outer(pos, inv_freq)
    
    # 扩展维度匹配q/k: (1, 1, seq_len, head_dim//2)
    cos = torch.cos(angles).unsqueeze(0).unsqueeze(0)
    sin = torch.sin(angles).unsqueeze(0).unsqueeze(0)
    
    # 旋转q (复数乘法实现)
    q1, q2 = q[..., ::2], q[..., 1::2]  # 分离奇偶维
    q_rot = torch.stack([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1).flatten(-2)
    
    # 旋转k
    k1, k2 = k[..., ::2], k[..., 1::2]
    k_rot = torch.stack([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1).flatten(-2)
    
    return q_rot, k_rot


# ====================【Day 4改造3: SwiGLU激活】====================
# 原版: Linear->GELU->Linear (2个矩阵)
# 改造: 并行(W1,W2)->SiLU(W1*x)*W2*x->W3 (3个矩阵，门控机制，LLaMA标配)
class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = 4 * config.n_embd
        # 三个投影层，均不带bias (LLaMA风格)
        self.w1 = nn.Linear(config.n_embd, hidden, bias=False)  # 门控分支
        self.w2 = nn.Linear(config.n_embd, hidden, bias=False)  # 值分支
        self.w3 = nn.Linear(hidden, config.n_embd, bias=False)  # 输出投影
        
    def forward(self, x):
        # SwiGLU: SiLU(W1·x) ⊙ (W2·x) → W3
        # SiLU(x) = x * sigmoid(x)，比GELU更平滑的门控
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


# 保留原GELU用于兼容性 (实际已不用)
class NewGELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


# ====================【Day 4改造4: 支持KV-Cache的注意力】====================
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # QKV投影 (合并计算，效率更高)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        
        # 因果掩码 (仍需要，但支持动态长度)
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x, layer_past=None, use_cache=False):
        B, T, C = x.size()
        
        # 计算QKV
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # 计算位置偏移量: 如果有历史缓存，偏移量为历史长度
        offset = 0
        if layer_past is not None:
            past_k, past_v = layer_past  # (B, nh, T_past, hs)
            offset = past_k.size(2)
        
        # ====================【Day 4: RoPE应用】====================
        # 原版: 加wpe绝对位置编码 (在GPT.forward中)
        # 改造: RoPE相对旋转 (在Attention内部，只对QK)
        q, k = apply_rope(q, k, T, C // self.n_head, offset=offset)
        
        # ====================【Day 4: KV-Cache拼接】====================
        # 原版: 每次重新计算全序列 (O(T^2))
        # 改造: 缓存历史K/V，只拼新token (O(T))
        if layer_past is not None:
            k = torch.cat((past_k, k), dim=2)  # 拼接历史
            v = torch.cat((past_v, v), dim=2)
        
        # 注意力计算 (支持变长因果掩码)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        total_len = k.size(2)  # 缓存后总长度
        
        # ====================【FIX: 正确的因果掩码】====================
        # Q的位置是 [offset, offset+1, ..., offset+T-1]
        # K的位置是 [0, 1, ..., total_len-1]
        # Q[i]只能看到K[j] where j <= offset + i
        q_pos = torch.arange(offset, offset + T, device=x.device).view(T, 1)
        k_pos = torch.arange(total_len, device=x.device).view(1, total_len)
        mask = (q_pos >= k_pos).float().view(1, 1, T, total_len)
        att = att.masked_fill(mask == 0, float('-inf'))
        # ====================【FIX END】====================
        
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, hs)
        
        # 重组多头
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        
        # 返回当前层K/V缓存供下次使用
        if use_cache:
            return y, (k, v)
        else:
            return y, None


# ====================【Day 4: 完整Block】====================
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 改造: RMSNorm替换LayerNorm (Pre-Norm位置不变)
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        # 改造: SwiGLU替换GELU-MLP
        self.mlp = SwiGLU(config)

    def forward(self, x, layer_past=None, use_cache=False):
        # Pre-Norm结构 (与原版一致，但Norm类型已换)
        attn_out, kv_cache = self.attn(self.ln_1(x), layer_past, use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, kv_cache


# ====================【Day 4: 完整GPT模型】====================
class GPT(nn.Module):
    @staticmethod
    def get_default_config():
        C = CN()
        C.model_type = 'gpt'
        C.n_layer = None
        C.n_head = None
        C.n_embd = None
        C.vocab_size = None
        C.block_size = None
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        return C

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.block_size = config.block_size

        # 模型类型解析
        type_given = config.model_type is not None
        params_given = all([config.n_layer is not None, config.n_head is not None, config.n_embd is not None])
        assert type_given ^ params_given
        if type_given:
            config.merge_from_dict({
                'openai-gpt':   dict(n_layer=12, n_head=12, n_embd=768),
                'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
                'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
                'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
                'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
                'gopher-44m':   dict(n_layer=8, n_head=16, n_embd=512),
                'gpt-mini':     dict(n_layer=6, n_head=6, n_embd=192),
                'gpt-micro':    dict(n_layer=4, n_head=4, n_embd=128),
                'gpt-nano':     dict(n_layer=3, n_head=3, n_embd=48),
            }[config.model_type])

        # ====================【Day 4: 主干改造】====================
        # 原版: 有wpe绝对位置编码 + LayerNorm
        # 改造: 无wpe (RoPE在Attention内处理) + RMSNorm
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # 注意: 移除了wpe (原RoPE不需要)
            drop = nn.Dropout(config.embd_pdrop),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd),  # 最终RMSNorm
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 初始化
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        
        # 参数统计 (不含lm_head，因通常与wte共享)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # ====================【Day 4: RMSNorm初始化】====================
        # 原版: LayerNorm (weight=1, bias=0)
        # 改造: RMSNorm (只有weight=1，无bias)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)
            # 注意: 没有bias初始化，因为RMSNorm没有bias！

    # ====================【Day 4: 优化器配置适配】====================
    # 原版: 黑名单中有LayerNorm
    # 改造: 黑名单改为RMSNorm (同样不应用weight decay)
    def configure_optimizers(self, train_config):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        # 关键修改: RMSNorm加入黑名单 (与Embedding一样不衰减)
        blacklist_weight_modules = (RMSNorm, torch.nn.Embedding)
        
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
        
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    # ====================【Day 4: 前向传播适配KV-Cache】====================
    # 原版: 返回 (logits, loss)
    # 改造: 始终返回3元组 (logits, loss, new_kvs)，兼容训练和生成
    def forward(self, idx, targets=None, past_key_values=None, use_cache=False):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size
        
        # 词嵌入 (无wpe，RoPE在Attention内处理)
        tok_emb = self.transformer.wte(idx)
        x = self.transformer.drop(tok_emb)
        
        # 逐层传递 (支持KV-Cache)
        new_kvs = [] if use_cache else None
        for i, block in enumerate(self.transformer.h):
            layer_past = past_key_values[i] if past_key_values is not None else None
            x, kv_cache = block(x, layer_past=layer_past, use_cache=use_cache)
            if use_cache:
                new_kvs.append(kv_cache)
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        
        # 始终返回3个值：训练时忽略第3个，生成时使用
        return logits, loss, (new_kvs if use_cache else None)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, 
                 top_k=None, top_p=None, temperature_decay=1.0, use_cache=True):
        """
        增强版生成: 支持Top-p、温度衰减、KV-Cache
        
        与原版差异:
        - 使用KV-Cache加速 (use_cache=True时)
        - 支持RoPE位置编码 (无长度限制)
        - 支持Top-p采样和温度衰减
        """
        # 初始化缓存
        past_kvs = None
        
        for step in range(max_new_tokens):
            # 输入裁剪 (使用KV-Cache时只输入最后1个token，否则输入全序列)
            if use_cache and past_kvs is not None:
                idx_cond = idx[:, -1:]  # 只取最新token，历史通过cache传递
            else:
                idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            
            # 前向 (返回3值，取第1个logits和第3个cache)
            logits, _, past_kvs = self(idx_cond, past_key_values=past_kvs, use_cache=use_cache)
            logits = logits[:, -1, :]  # 取最后位置
            
            # 温度衰减 (安全下限0.1防除0)
            current_temp = temperature * (temperature_decay ** step)
            current_temp = max(current_temp, 0.1)
            logits = logits / current_temp
            
            # Top-p (Nucleus)采样
            if top_p is not None and 0 < top_p < 1:
                probs = F.softmax(logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumsum_probs > top_p
                sorted_indices_to_remove[..., 0] = False  # 至少保留第1个
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[0, indices_to_remove] = float('-inf')
            
            # Top-k裁剪
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # 采样
            probs = F.softmax(logits, dim=-1)
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
