"""
Day 4 架构诊断测试脚本
用于检测 SwiGLU、RoPE、KV-Cache 的正确性
"""

import torch
import torch.nn as nn
import sys
import os

# 添加 mingpt 到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mingpt.im_model import GPT, CausalSelfAttention, Block, RMSNorm, SwiGLU, apply_rope
from mingpt.utils import CfgNode as CN

def test_swiglu():
    """测试 SwiGLU 门控是否正常"""
    print("\n" + "="*60)
    print("🧪 测试 1: SwiGLU 门控状态")
    print("="*60)
    
    config = CN()
    config.n_embd = 192
    
    swiglu = SwiGLU(config).cuda()
    
    # 测试输入
    x = torch.randn(2, 10, 192).cuda()
    
    with torch.no_grad():
        out = swiglu(x)
        
        # 检查输出范数（应该与输入相近，不应爆炸或消失）
        in_norm = x.norm().item()
        out_norm = out.norm().item()
        ratio = out_norm / (in_norm + 1e-8)
        
        print(f"输入范数: {in_norm:.4f}")
        print(f"输出范数: {out_norm:.4f}")
        print(f"比值: {ratio:.4f} (应接近 1.0-5.0)")
        
        # 检查门控状态
        gate = torch.sigmoid(swiglu.w1(x))
        gate_mean = gate.mean().item()
        gate_std = gate.std().item()
        print(f"门控(SiLU)均值: {gate_mean:.4f} (应在 0.3-0.7 之间)")
        print(f"门控标准差: {gate_std:.4f} (应有明显分布，不应全为0或1)")
        
        # 检查死亡/饱和神经元
        dead_ratio = (gate < 0.01).float().mean().item()
        sat_ratio = (gate > 0.99).float().mean().item()
        print(f"死亡门控(<0.01): {dead_ratio:.2%} (应 <10%)")
        print(f"饱和门控(>0.99): {sat_ratio:.2%} (应 <10%)")
        
        if dead_ratio > 0.5 or sat_ratio > 0.5:
            print("❌ SwiGLU 严重饱和/死亡，需要调整初始化！")
            return False
        else:
            print("✅ SwiGLU 状态正常")
            return True


def test_rope_consistency():
    """测试 RoPE 旋转是否正确（关键：位置区分性）"""
    print("\n" + "="*60)
    print("🧪 测试 2: RoPE 位置区分性")
    print("="*60)
    
    B, n_head, T, head_dim = 1, 6, 128, 32
    
    # 创建相同内容的 query，但位于不同位置
    q_base = torch.randn(B, n_head, 1, head_dim).cuda()
    
    # 复制到多个位置
    q_all = q_base.repeat(1, 1, T, 1)  # (1, 6, 128, 32)，内容相同，位置不同
    
    with torch.no_grad():
        # 应用 RoPE
        q_rot, _ = apply_rope(q_all, q_all.clone(), T, head_dim)
        
        # 检查位置 0 和位置 50 的向量是否不同
        vec_0 = q_rot[0, 0, 0, :]
        vec_50 = q_rot[0, 0, 50, :]
        
        sim = torch.cosine_similarity(vec_0, vec_50, dim=0).item()
        diff_norm = (vec_0 - vec_50).norm().item()
        
        print(f"位置0 vs 位置50 的余弦相似度: {sim:.4f}")
        print(f"位置0 vs 位置50 的 L2 距离: {diff_norm:.4f}")
        
        if sim > 0.99 and diff_norm < 0.1:
            print("❌ RoPE 失效！不同位置的编码几乎相同")
            print("   这会导致生成时所有 token 看到相同位置信息，输出乱码！")
            return False
        else:
            print("✅ RoPE 能区分不同位置")
            
        # 检查旋转是否保持范数
        norm_before = q_all.norm().item()
        norm_after = q_rot.norm().item()
        print(f"RoPE 前后范数比: {norm_after/norm_before:.4f} (应 ≈ 1.0)")
        
        return True


def test_rope_with_cache():
    """测试 RoPE + KV-Cache 的致命 Bug（位置偏移）"""
    print("\n" + "="*60)
    print("🧪 测试 3: RoPE + KV-Cache 位置偏移（最关键！）")
    print("="*60)
    
    config = CN()
    config.n_embd = 192
    config.n_head = 6
    config.block_size = 128
    config.attn_pdrop = 0.0
    config.resid_pdrop = 0.0
    
    attn = CausalSelfAttention(config).cuda()
    
    # 模拟生成第 100 个 token 的场景
    step = 100
    
    # 创建缓存（假装前面已经生成了 100 个 token）
    B, head_dim = 1, 192 // 6
    cache_len = step  # 100
    fake_cache_k = torch.randn(B, config.n_head, cache_len, head_dim).cuda()
    fake_cache_v = torch.randn(B, config.n_head, cache_len, head_dim).cuda()
    
    # 当前输入只有 1 个 token（最后一个）
    current_x = torch.randn(1, 1, 192).cuda()
    
    with torch.no_grad():
        # 测试 1: 不使用 cache（应该正确）
        full_x = torch.randn(1, step+1, 192).cuda()  # 完整序列
        out_full, _ = attn(full_x, layer_past=None, use_cache=False)
        q_full = attn.c_attn(full_x)[:, :192].view(1, step+1, config.n_head, head_dim).transpose(1,2)
        q_full_rot, _ = apply_rope(q_full, q_full.clone(), step+1, head_dim)
        
        # 取第 100 个位置的 query 特征
        feature_full = q_full_rot[0, 0, step, :]
        
        # 测试 2: 使用 cache（关键测试）
        out_cache, (new_k, new_v) = attn(current_x, layer_past=(fake_cache_k, fake_cache_v), use_cache=True)
        q_cache = attn.c_attn(current_x)[:, :192].view(1, 1, config.n_head, head_dim).transpose(1,2)
        
        # 正确的做法：传递offset=100
        q_cache_rot, _ = apply_rope(q_cache, q_cache.clone(), 1, head_dim, offset=step)
        feature_cache = q_cache_rot[0, 0, 0, :]
        
        # 比较
        diff = (feature_full - feature_cache).norm().item()
        print(f"完整序列第100位 vs Cache模式第100位 特征差异: {diff:.4f}")
        
        if diff < 0.1:
            print("⚠️ 警告：特征几乎相同，可能 RoPE 没有正确应用位置偏移")
            print("   或者差异过大（>10），说明实现不一致")
        
        # 正确的做法应该是：
        # q_cache 应该被编码为位置 100，而不是位置 0
        # 这需要 apply_rope 知道 offset=100
        
        print("\n💡 诊断：如果生成乱码，且 diff 很大，说明 RoPE 需要 position_offset 参数")
        print("   当前 apply_rope 实现只使用 torch.arange(seq_len)，无法处理 Cache 场景！")
        
        return True


def test_rmsnorm_gradient():
    """测试 RMSNorm 梯度是否正常"""
    print("\n" + "="*60)
    print("🧪 测试 4: RMSNorm 梯度健康度")
    print("="*60)
    
    config = CN()
    config.n_embd = 192
    config.block_size = 128
    config.n_layer = 6
    config.n_head = 6
    config.vocab_size = 65
    config.embd_pdrop = 0.1
    config.resid_pdrop = 0.1
    config.attn_pdrop = 0.1
    
    model = GPT(config).cuda()
    model.train()
    
    # 模拟一次前向和反向
    x = torch.randint(0, 65, (2, 64)).cuda()
    y = torch.randint(0, 65, (2, 64)).cuda()
    
    logits, loss, _ = model(x, y)
    loss.backward()
    
    # 检查各层梯度
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None and 'weight' in name:
            norm = param.grad.norm().item()
            grad_norms.append((name, norm))
    
    # 排序看最大最小
    grad_norms.sort(key=lambda x: x[1])
    
    print("梯度范数最小 3 层:")
    for name, norm in grad_norms[:3]:
        print(f"  {name}: {norm:.6f}")
    
    print("\n梯度范数最大 3 层:")
    for name, norm in grad_norms[-3:]:
        print(f"  {name}: {norm:.6f}")
    
    max_grad = grad_norms[-1][1]
    min_grad = grad_norms[0][1]
    
    if max_grad > 1000:
        print("❌ 梯度爆炸！RMSNorm 需要降低学习率或梯度裁剪阈值")
        return False
    elif min_grad < 1e-7:
        print("❌ 梯度消失！可能是 RMSNorm 初始化或深度问题")
        return False
    else:
        print(f"✅ 梯度范围正常 (1e-6 ~ 100)")
        return True


def test_generation_step_by_step():
    """逐步测试生成过程，看哪一步开始乱码"""
    print("\n" + "="*60)
    print("🧪 测试 5: 生成过程逐步检查")
    print("="*60)
    
    config = CN()
    config.model_type = 'gpt-mini'
    config.vocab_size = 65
    config.block_size = 128
    
    model = GPT(config).cuda()
    model.eval()
    
    idx = torch.randint(0, 65, (1, 5)).cuda()
    
    print("逐步生成检查（观察缓存长度和位置）:")
    past_kvs = None
    
    for step in range(10):
        if past_kvs is not None:
            idx_cond = idx[:, -1:]  # 只用最后一个
            print(f"Step {step}: 输入长度=1, 缓存长度={past_kvs[0][0].size(2)}")
        else:
            idx_cond = idx
            print(f"Step {step}: 输入长度={idx.size(1)}, 无缓存")
        
        with torch.no_grad():
            logits, _, past_kvs = model(idx_cond, past_key_values=past_kvs, use_cache=True)
            
            # 检查 logits 分布
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
            max_prob = probs.max().item()
            
            print(f"       预测熵={entropy:.2f}, 最大概率={max_prob:.3f}")
            
            if entropy < 0.5:
                print("       ⚠️ 熵极低，模型非常确定（可能是重复模式）")
            if max_prob > 0.99:
                print("       ⚠️ 某个 token 概率接近 1，可能陷入循环")
        
        # 贪婪解码下一个
        idx_next = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)


if __name__ == "__main__":
    print("🚀 Day 4 架构诊断套件")
    print("用于检测 SwiGLU/RoPE/KV-Cache 的实现错误")
    
    # 检查 CUDA
    if not torch.cuda.is_available():
        print("❌ 需要 CUDA 运行测试")
        sys.exit(1)
    
    print(f"使用设备: {torch.cuda.get_device_name(0)}")
    
    # 运行所有测试
    results = []
    
    try:
        results.append(("SwiGLU", test_swiglu()))
    except Exception as e:
        print(f"❌ SwiGLU 测试崩溃: {e}")
        results.append(("SwiGLU", False))
    
    try:
        results.append(("RoPE Consistency", test_rope_consistency()))
    except Exception as e:
        print(f"❌ RoPE 测试崩溃: {e}")
        results.append(("RoPE Consistency", False))
    
    try:
        results.append(("RoPE+Cache", test_rope_with_cache()))
    except Exception as e:
        print(f"❌ RoPE+Cache 测试崩溃: {e}")
        results.append(("RoPE+Cache", False))
    
    try:
        results.append(("RMSNorm Grad", test_rmsnorm_gradient()))
    except Exception as e:
        print(f"❌ RMSNorm 测试崩溃: {e}")
        results.append(("RMSNorm Grad", False))
    
    try:
        test_generation_step_by_step()
    except Exception as e:
        print(f"❌ 生成测试崩溃: {e}")
    
    # 总结
    print("\n" + "="*60)
    print("📊 测试结果总结")
    print("="*60)
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{name}: {status}")
    
    print("\n💡 修复建议:")
    print("1. 如果 SwiGLU 失败: 在 _init_weights 中添加 Xavier 初始化，gain=0.5")
    print("2. 如果 RoPE 失败: apply_rope 需要添加 position_offset 参数")
    print("3. 如果 Cache 失败: CausalSelfAttention 需要知道真实序列长度而非 just T")
    print("4. 如果 RMSNorm 梯度失败: 降低 grad_norm_clip 到 0.5 或 0.1")
