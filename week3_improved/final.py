import torch
import time
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ========== 步骤 1: 环境配置与量化设置 ==========
print("=" * 50)
print("Day 7 实战：Qwen2.5-7B 4-bit 量化显存压力测试")
print("=" * 50)

# 4-bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",      # Normal Float 4，信息熵最优
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True  # 嵌套量化，进一步省显存
)

model_id = "Qwen/Qwen2.5-7B-Instruct"

# ========== 步骤 2: 模型加载与显存监控 ==========
def get_gpu_memory():
    """获取当前GPU显存占用（MB）"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2, torch.cuda.max_memory_allocated() / 1024**2
    return 0, 0

print("\n[阶段 1] 加载模型前显存：", f"{get_gpu_memory()[0]:.2f} MB")

tokenizer = AutoTokenizer.from_pretrained(model_id)

# 加载 4-bit 量化模型（约需 4-5GB 显存）
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",           # 自动分配层到GPU/CPU
    trust_remote_code=True,
    torch_dtype=torch.float16
)

model_memory, _ = get_gpu_memory()
print(f"[阶段 2] 模型加载后显存：{model_memory:.2f} MB (约 {model_memory/1024:.2f} GB)")
print(f"        纯模型权重约：4GB (4-bit量化后)")

# ========== 步骤 3: 理论 KV-Cache 计算 ==========
print("\n" + "=" * 50)
print("步骤 3: 理论 KV-Cache 占用计算")
print("=" * 50)

# Qwen2.5-7B 真实架构参数
num_layers = 28          # 层数
num_heads = 28           # Query 头数 (GQA 分组查询)
num_kv_heads = 4         # KV 头数（GQA 共享，这是关键！）
head_dim = 128           # 每个头的维度
seq_len = 4096           # 序列长度
batch_size = 1           # 批次大小
dtype_bytes = 2          # fp16 = 2字节

# 公式（GQA 版本）：2 * layers * batch * seq_len * kv_heads * head_dim * bytes
kv_cache_size_gqa = (2 * num_layers * batch_size * seq_len * num_kv_heads * head_dim * dtype_bytes)
kv_cache_mb = kv_cache_size_gqa / 1024**2

print(f"模型架构参数:")
print(f"  - 层数: {num_layers}")
print(f"  - Query Heads: {num_heads}")
print(f"  - KV Heads (GQA): {num_kv_heads} ← 关键优化点")
print(f"  - Head Dim: {head_dim}")
print(f"  - 序列长度: {seq_len}")

print(f"\n理论 KV-Cache 计算公式（GQA）:")
print(f"  2 * {num_layers}层 * {batch_size}batch * {seq_len}seq * {num_kv_heads}kv_heads * {head_dim}dim * {dtype_bytes}bytes")
print(f"  = {kv_cache_size_gqa:,} bytes = {kv_cache_mb:.2f} MB = {kv_cache_mb/1024:.2f} GB")

# 对比：如果是标准 MHA（无 GQA）
kv_cache_size_mha = (2 * num_layers * batch_size * seq_len * num_heads * head_dim * dtype_bytes)
print(f"\n对比：如果是标准 MHA（28 heads）:")
print(f"  理论占用: {kv_cache_size_mha/1024**2:.2f} MB = {kv_cache_size_mha/1024**3:.2f} GB")
print(f"  GQA 节省: {(1 - kv_cache_size_gqa/kv_cache_size_mha)*100:.1f}% 显存")

# ========== 步骤 4: 开启/关闭 KV-Cache 速度对比 ==========
print("\n" + "=" * 50)
print("步骤 4: 生成速度对比（使用 time.perf_counter）")
print("=" * 50)

test_prompt = "请详细介绍深度学习中的注意力机制原理，并举例说明其应用场景。"
inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

def benchmark_generate(use_cache, num_runs=3):
    """测量生成速度"""
    times = []
    tokens_per_sec_list = []
    
    for _ in range(num_runs):
        # 清理缓存并重置内存统计
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                use_cache=use_cache,           # 开关 KV-Cache
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end = time.perf_counter()
        elapsed = end - start
        num_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
        tps = num_tokens / elapsed
        
        times.append(elapsed)
        tokens_per_sec_list.append(tps)
        
        del outputs
        gc.collect()
        torch.cuda.empty_cache()
    
    return sum(times)/len(times), sum(tokens_per_sec_list)/len(tokens_per_sec_list)

print("\n[测试 A] 开启 use_cache=True（使用 KV-Cache）...")
avg_time_cache, avg_tps_cache = benchmark_generate(use_cache=True)
print(f"  平均耗时: {avg_time_cache:.2f}s")
print(f"  生成速度: {avg_tps_cache:.2f} tokens/sec")

print("\n[测试 B] 关闭 use_cache=False（实时计算注意力）...")
avg_time_no_cache, avg_tps_no_cache = benchmark_generate(use_cache=False)
print(f"  平均耗时: {avg_time_no_cache:.2f}s")
print(f"  生成速度: {avg_tps_no_cache:.2f} tokens/sec")

# 对比结果
speedup = avg_tps_cache / avg_tps_no_cache
print(f"\n[对比结果]")
print(f"  KV-Cache 加速比: {speedup:.1f}x")
print(f"  长文本生成时，KV-Cache 避免重复计算历史 Key/Value，速度提升显著")
print(f"  但代价是显存占用增加 {kv_cache_mb:.2f} MB（本例 seq_len=4096）")

# 最终显存占用
final_memory, peak_memory = get_gpu_memory()
print(f"\n[最终显存统计]")
print(f"  峰值显存: {peak_memory/1024:.2f} GB")
print(f"  组成: 模型权重(4GB) + KV-Cache({kv_cache_mb/1024:.2f}GB) + 激活值(~0.5GB) + 开销")
