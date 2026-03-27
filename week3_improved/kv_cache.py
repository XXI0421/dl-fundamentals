import torch
import torch.nn as nn

class KVCache:
    def __init__(self, num_layers, batch_size, num_heads, max_seq_len, d_k, dtype=torch.float32):
        """
        预分配 KV Cache 内存（避免动态扩容开销）
        
        Args:
            num_layers: 模型层数（如 2，对应 Week 2 的 TransformerBlock 堆叠）
            batch_size: 批次大小（通常是 1，生成时一个序列）
            num_heads: 多头注意力的头数
            max_seq_len: 最大序列长度（如 512）
            d_k: 每个头的维度
            dtype: 数据类型，fp16 节省显存
        """
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.d_k = d_k
        self.current_len = 0  # 当前已缓存的序列长度
        
        # TODO: 预分配内存
        # cache_k 和 cache_v 的形状：(num_layers, batch, num_heads, max_seq_len, d_k)
        # 使用 torch.zeros 初始化
        self.cache_k = torch.zeros(num_layers, batch_size, num_heads, max_seq_len, d_k, dtype=dtype)
        self.cache_v = torch.zeros(num_layers, batch_size, num_heads, max_seq_len, d_k, dtype=dtype)
        
        # 移动到 GPU（如果可用）
        if torch.cuda.is_available():
            self.cache_k = self.cache_k.cuda()
            self.cache_v = self.cache_v.cuda()
    
    def update(self, layer_idx, new_k, new_v):
        """
        将新的 K/V 写入 Cache
        
        Args:
            layer_idx: 当前层索引（0 到 num_layers-1）
            new_k: (batch, num_heads, 1, d_k) - 当前 token 的 Key
            new_v: (batch, num_heads, 1, d_k) - 当前 token 的 Value
            
        Returns:
            更新后的该层完整 K/V：(batch, num_heads, current_len, d_k)
        """
        # TODO: 将 new_k, new_v 写入 cache_k/cache_v 的 current_len 位置
        # 注意：new_k/new_v 可能有维度 (batch, num_heads, 1, d_k)，需要 squeeze 或调整
        
        # 写入位置：第 layer_idx 层，第 current_len 个 token
        self.cache_k[layer_idx, :, :, self.current_len, :] = new_k.squeeze(2)  # 去掉第3维的1
        self.cache_v[layer_idx, :, :, self.current_len, :] = new_v.squeeze(2)
        
        # 返回该层从开始到 current_len 的所有 K/V（用于 Attention 计算）
        return (self.cache_k[layer_idx, :, :, :self.current_len+1, :],
                self.cache_v[layer_idx, :, :, :self.current_len+1, :])
    
    def get(self, layer_idx):
        """
        获取当前累积的所有 K/V（用于 Attention 计算）
        
        Returns:
            (k, v): 都是 (batch, num_heads, current_len, d_k)
        """
        return (self.cache_k[layer_idx, :, :, :self.current_len, :],
                self.cache_v[layer_idx, :, :, :self.current_len, :])
    
    def increment(self):
        """生成完一个 token 后，增加当前长度计数"""
        self.current_len += 1
    
    def reset(self):
        """开始新序列时重置"""
        self.current_len = 0


# 测试配置（对应 Week 2 的小模型）
num_layers = 2
batch_size = 1
num_heads = 4
max_seq_len = 64
d_k = 32  # d_model=128, num_heads=4, 所以 d_k=32

# 初始化 Cache
cache = KVCache(num_layers, batch_size, num_heads, max_seq_len, d_k)

print(f"Cache K 形状: {cache.cache_k.shape}")  # 应为 (2, 1, 4, 64, 32)
print(f"初始 current_len: {cache.current_len}")  # 应为 0

# 模拟生成 3 个 token
for step in range(3):
    # 模拟当前 token 的 K/V（来自 TransformerBlock 的计算）
    new_k = torch.randn(batch_size, num_heads, 1, d_k)
    new_v = torch.randn(batch_size, num_heads, 1, d_k)
    
    # 更新第 0 层
    k_cache, v_cache = cache.update(0, new_k, new_v)
    
    print(f"\nStep {step+1}:")
    print(f"  New K/V shape: {new_k.shape}")
    print(f"  Cached K shape: {k_cache.shape}")  # 应为 (1, 4, step+1, 32)
    print(f"  Cache length: {cache.current_len}")
    
    cache.increment()

# 验证显存占用
cache_mb = (cache.cache_k.numel() + cache.cache_v.numel()) * 2 / 1024 / 1024  # fp16=2bytes
print(f"\n总 Cache 显存: {cache_mb:.2f} MB")
print("✅ KVCache 基础功能测试通过")
