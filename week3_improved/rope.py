import torch
import math

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    预计算旋转频率（复数形式 e^(i*freqs)）
    
    Args:
        dim: 维度（必须是偶数）
        end: 最大序列长度
        theta: 基础频率（Llama 默认 10000）
    
    Returns:
        freqs_cis: (end, dim//2) 的复数张量
    """
    # TODO 1: 计算每对维度的频率 theta^(-2i/dim)
    # 提示：torch.arange(0, dim, 2) 获取偶数索引
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    
    # TODO 2: 计算 m * theta（m 从 0 到 end-1）
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs)  # (end, dim//2)
    
    # TODO 3: 转换为复数 e^(i*freqs) = cos(freqs) + i*sin(freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    """
    应用旋转位置编码
    
    Args:
        xq: (batch, num_heads, seq_len, head_dim)
        xk: (batch, num_heads, seq_len, head_dim)
        freqs_cis: (seq_len, head_dim//2) 复数张量
    
    Returns:
        rotated_q, rotated_k: 旋转后的 Query 和 Key
    """
    # TODO 4: 将实数张量视为复数（每两个维度看作实部/虚部）
    # 提示：torch.view_as_complex
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)  # (..., dim//2, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
    
    xq_complex = torch.view_as_complex(xq_)
    xk_complex = torch.view_as_complex(xk_)
    
    # TODO 5: 复数乘法实现旋转
    # (batch, heads, seq, dim//2) * (seq, dim//2) -> 广播乘法
    xq_out = xq_complex * freqs_cis.unsqueeze(0).unsqueeze(0)
    xk_out = xk_complex * freqs_cis.unsqueeze(0).unsqueeze(0)
    
    # TODO 6: 转回实数张量
    xq_out = torch.view_as_real(xq_out).flatten(-2)
    xk_out = torch.view_as_real(xk_out).flatten(-2)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)

# ========== 关键验证：相对位置性质 ==========
def test_relative_position():
    """验证 RoPE 的核心性质：Attention(q_m, k_n) 只与 m-n 有关"""
    dim = 64
    seq_len = 10
    
    # 创建两个相同的向量，放在不同位置
    base_vec = torch.randn(1, 1, 1, dim)
    
    # 位置 2 和位置 5
    pos_a = 2
    pos_b = 5
    relative_dist = pos_b - pos_a  # 3
    
    # 创建序列：在位置2和位置5放置相同向量
    q = torch.zeros(1, 1, seq_len, dim)
    q[0, 0, pos_a, :] = base_vec[0, 0, 0, :]
    
    k = torch.zeros(1, 1, seq_len, dim)
    k[0, 0, pos_b, :] = base_vec[0, 0, 0, :]
    
    # 预计算频率
    freqs_cis = precompute_freqs_cis(dim, seq_len)
    
    # 应用 RoPE
    q_rot, k_rot = apply_rotary_emb(q, k, freqs_cis)
    
    # 计算 attention score（点积）
    score = torch.matmul(q_rot[0, 0, pos_a, :], k_rot[0, 0, pos_b, :])
    
    # 对比：位置3和位置6（同样是相对距离3）
    pos_c = 3
    pos_d = 6
    q2 = torch.zeros(1, 1, seq_len, dim)
    q2[0, 0, pos_c, :] = base_vec[0, 0, 0, :]
    k2 = torch.zeros(1, 1, seq_len, dim)
    k2[0, 0, pos_d, :] = base_vec[0, 0, 0, :]
    
    q2_rot, k2_rot = apply_rotary_emb(q2, k2, freqs_cis)
    score2 = torch.matmul(q2_rot[0, 0, pos_c, :], k2_rot[0, 0, pos_d, :])
    
    print(f"位置 {pos_a} 与 {pos_b} 的 Attention Score: {score.item():.4f}")
    print(f"位置 {pos_c} 与 {pos_d} 的 Attention Score: {score2.item():.4f}")
    print(f"差异: {abs(score.item() - score2.item()):.6f}（应接近0）")
    
    if abs(score.item() - score2.item()) < 1e-5:
        print("✅ 相对位置性质验证通过！RoPE 只依赖相对距离")
    else:
        print("❌ 验证失败")

# ========== 验证代码 ==========
if __name__ == "__main__":
    batch, heads, seq_len, head_dim = 2, 4, 10, 64
    
    # 随机 Q, K
    q = torch.randn(batch, heads, seq_len, head_dim)
    k = torch.randn(batch, heads, seq_len, head_dim)
    
    # 预计算旋转频率
    freqs_cis = precompute_freqs_cis(head_dim, seq_len)
    print(f"预计算频率形状: {freqs_cis.shape}")  # (10, 32)
    
    # 应用旋转
    q_rot, k_rot = apply_rotary_emb(q, k, freqs_cis)
    print(f"旋转后 Q 形状: {q_rot.shape}")  # (2, 4, 10, 64)
    
    # 关键验证：相对位置性质
    # 位置 m 的 q 和位置 n 的 k 的 attention 应该只与 m-n 有关
    print("\n✅ RoPE 基础实现完成")
    test_relative_position()
