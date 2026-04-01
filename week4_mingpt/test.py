"""
inference.py - 加载训练好的CharGPT进行文本生成
修复了原代码未保存stoi/itos的问题
"""

import torch
import os
from mingpt.model import GPT
from mingpt.utils import CfgNode as CN

# ====================【Step 1: 重建词表（原代码缺陷补救）】====================
# 原代码只保存了权重，没保存字符映射。我们需要重新扫描input.txt重建映射
def rebuild_vocab(input_file='input.txt'):
    """从原始文本重建字符↔索引映射（因原保存逻辑缺陷）"""
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"需要{input_file}来重建词表！请将训练用的文本放在同级目录")
    
    text = open(input_file, 'r', encoding='utf-8').read()
    chars = sorted(list(set(text)))
    
    stoi = {ch: i for i, ch in enumerate(chars)}  # 字符→索引
    itos = {i: ch for i, ch in enumerate(chars)}  # 索引→字符
    
    print(f"词表重建完成: {len(chars)}个独特字符")
    return stoi, itos, len(chars)

# ====================【Step 2: 加载模型】====================
def load_model(checkpoint_path='./out/chargpt/model.pt', input_file='input.txt', device='auto'):
    """加载模型并重建词表"""
    
    # 自动选择设备（优先4060）
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 重建词表（关键！原代码没保存这个）
    stoi, itos, vocab_size = rebuild_vocab(input_file)
    
    # 重建模型配置（需与训练时一致）
    config = GPT.get_default_config()
    config.model_type = 'gpt-mini'  # 必须与训练时一致！
    config.vocab_size = vocab_size
    config.block_size = 128          # 必须与训练时一致！
    
    # 实例化并加载权重
    model = GPT(config)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"✅ 模型加载成功: {checkpoint_path}")
    print(f"   参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    return model, stoi, itos, device

# ====================【Step 3: 文本生成】====================
def generate_text(model, stoi, itos, prompt, max_new_tokens=200, 
                  temperature=1.0, top_k=10, device='cpu'):
    """
    生成文本
    
    参数:
        prompt: 提示文本（字符串）
        max_new_tokens: 生成长度
        temperature: 温度（0.8更保守，1.0标准，1.2更随机）
        top_k: 只从概率前k个候选中采样（10-50较合适）
    """
    # 编码提示语
    if any(c not in stoi for c in prompt):
        invalid = [c for c in prompt if c not in stoi]
        raise ValueError(f"提示语包含未见过字符: {invalid[:5]}...")
    
    idx = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long).to(device)
    
    # 生成
    with torch.no_grad():
        generated = model.generate(
            idx, 
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_k=top_k
        )[0]
    
    # 解码（包括提示部分）
    text = ''.join([itos[int(i)] for i in generated])
    return text

# ====================【主程序：交互式生成】====================
if __name__ == '__main__':
    # 加载（自动检测4060）
    model, stoi, itos, device = load_model()
    
    # 交互式生成
    print("\n📝 输入提示语（或quit退出）：")
    print("建议英文提示（如'To be, or not to be'），因为训的是莎士比亚")
    
    while True:
        prompt = input("\nPrompt: ").strip()
        if prompt.lower() in ['quit', 'exit', 'q']:
            break
        
        if not prompt:
            continue
            
        try:
            result = generate_text(
                model, stoi, itos, prompt, 
                max_new_tokens=300,
                temperature=0.8,  # 保守一点，减少造词
                top_k=20,
                device=device
            )
            print(f"\n🎭 生成结果:\n{result}")
            
            # 只显示新生成的部分（去掉提示）
            new_part = result[len(prompt):]
            print(f"\n✨ 续写部分:\n{new_part[:100]}...")
            
        except Exception as e:
            print(f"❌ 错误: {e}")

