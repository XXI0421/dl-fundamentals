"""
inference.py - 加载训练好的CharGPT进行文本生成
修复了原代码未保存stoi/itos的问题
Day 3升级: 支持Top-p(Nucleus采样)与温度衰减(Temperature Decay)
"""

import torch
import os
from mingpt.model_top_p import GPT
from mingpt.utils import CfgNode as CN

# ====================【Step 1: 重建词表（原代码缺陷补救）】====================
def rebuild_vocab(input_file='input.txt'):
    """从原始文本重建字符↔索引映射（因原保存逻辑缺陷）"""
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"需要{input_file}来重建词表！请将训练用的文本放在同级目录")
    
    text = open(input_file, 'r', encoding='utf-8').read()
    chars = sorted(list(set(text)))
    
    stoi = {ch: i for i, ch in enumerate(chars)}  
    itos = {i: ch for i, ch in enumerate(chars)}  
    
    print(f"词表重建完成: {len(chars)}个独特字符")
    return stoi, itos, len(chars)

# ====================【Step 2: 加载模型】====================
def load_model(checkpoint_path='./out/chargpt/model_top_p.pt', input_file='input.txt', device='auto'):
    """加载模型并重建词表
    默认加载 model_top_p.pt (支持Top-p采样的模型)
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    stoi, itos, vocab_size = rebuild_vocab(input_file)
    
    config = GPT.get_default_config()
    config.model_type = 'gpt-mini'  
    config.vocab_size = vocab_size
    config.block_size = 128          
    
    model = GPT(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
        
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # 增强版格式：{model_state_dict, stoi, itos, ...}
            print("检测到增强版检查点格式（含完整配置）")
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # 如果检查点里有词表，优先使用（更可靠）
            if 'stoi' in checkpoint and 'itos' in checkpoint:
                stoi = checkpoint['stoi']
                itos = checkpoint['itos']
                print("使用检查点内嵌词表")
    else:
            # 原版格式：裸state_dict
            print("检测到原版检查点格式（仅权重）")
            model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    
    print(f"✅ 模型加载成功")
    print(f"   参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    return model, stoi, itos, device

# ====================【Step 3: 文本生成 - Day 3增强版】====================
def generate_text(model, stoi, itos, prompt, max_new_tokens=200, 
                  temperature=1.0, do_sample=True, top_k=10, top_p=None, temperature_decay=1.0, device='cpu'):
    """
    生成文本（Day 3增强：支持Nucleus采样与动态温度衰减）
    
    参数:
        prompt: 提示文本（字符串）
        max_new_tokens: 生成长度
        temperature: 初始温度（>1更随机，<1更保守）
        do_sample: True=采样(有随机性), False=贪婪解码(确定性)
        top_k: 只从概率前k个候选中采样（None=全部）
        top_p: Nucleus采样阈值（0.9=从累积概率90%的核内采样，None=不启用）
               与top_k互斥，建议只选其一或先用top_k裁剪再用top_p
        temperature_decay: 温度衰减因子（每步乘以该值，<1.0逐渐降温）
                          例如0.98：每步降温2%，使生成从创意走向确定
        device: 计算设备
    """
    # 编码提示语
    if any(c not in stoi for c in prompt):
        invalid = [c for c in prompt if c not in stoi]
        raise ValueError(f"提示语包含未见过字符: {invalid[:5]}...")
    
    idx = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long).to(device)
    
    # 生成（调用Day 3增强的model.generate）
    with torch.no_grad():
        generated = model.generate(
            idx, 
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,  # Day 3必须采样才能体现top_p和temp_decay效果
            top_k=top_k,
            top_p=top_p,                    # Day 3新增：Nucleus采样
            temperature_decay=temperature_decay  # Day 3新增：温度衰减
        )[0]
    
    # 解码
    text = ''.join([itos[int(i)] for i in generated])
    return text

# ====================【Day 3新增：批量对比测试】====================
def compare_strategies(model, stoi, itos, prompt, device='cpu'):
    """
    对比不同生成策略的效果（Day 3实验工具）
    """
    strategies = [
        ("贪婪解码", {"temperature": 1.0, "do_sample": False, "top_k": None, "top_p": None, "temperature_decay": 1.0}),
        ("Top-k(k=10)", {"temperature": 1.0, "do_sample": True, "top_k": 10, "top_p": None, "temperature_decay": 1.0}),
        ("Top-p(p=0.9)", {"temperature": 1.0, "do_sample": True, "top_k": None, "top_p": 0.9, "temperature_decay": 1.0}),
        ("高温创意(1.3)", {"temperature": 1.3, "do_sample": True, "top_k": None, "top_p": 0.95, "temperature_decay": 1.0}),
        ("温度衰减(1.2→0.3)", {"temperature": 1.2, "do_sample": True, "top_k": None, "top_p": 0.9, "temperature_decay": 0.97}),
    ]
    
    print(f"\n{'='*60}")
    print(f"提示语: '{prompt}'")
    print(f"{'='*60}")
    
    for name, kwargs in strategies:
        # 固定种子保证可比性
        torch.manual_seed(42)
        
        result = generate_text(model, stoi, itos, prompt, max_new_tokens=500, device=device, **kwargs)
        new_part = result[len(prompt):]
        
        print(f"\n🎭【{name}】:")
        print(f"{new_part[:120]}{'...' if len(new_part) > 120 else ''}")

# ====================【主程序：交互式生成 - Day 3增强】====================
if __name__ == '__main__':
    model, stoi, itos, device = load_model()
    
    print("\n📝 Day 3增强版生成器")
    print("支持: Top-k | Top-p(Nucleus) | 温度衰减")
    print("输入 'compare' 运行对比实验, 'quit' 退出")
    
    while True:
        prompt = input("\nPrompt (或命令): ").strip()
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            break
        
        if prompt.lower() == 'compare':
            # 运行对比实验
            test_prompt = input("输入对比测试的提示语: ").strip() or "To be, or not to be"
            compare_strategies(model, stoi, itos, test_prompt, device)
            continue
        
        if not prompt:
            continue
            
        try:
            # 解析高级参数（简化版交互）
            use_top_p = input("使用Top-p采样? (y/n, 默认n): ").lower() == 'y'
            top_p = 0.9 if use_top_p else None
            
            use_decay = input("使用温度衰减? (y/n, 默认n): ").lower() == 'y'
            temp_decay = 0.97 if use_decay else 1.0
            
            temp = float(input("温度 (默认1.0): ") or "1.0")
            
            result = generate_text(
                model, stoi, itos, prompt, 
                max_new_tokens=200,
                temperature=temp,
                top_k=10 if not use_top_p else None,  # Top-p和Top-k通常二选一
                top_p=top_p,
                temperature_decay=temp_decay,
                device=device
            )
            
            new_part = result[len(prompt):]
            print(f"\n✨ 续写 ({len(new_part)}字符):")
            print(new_part)
            
            if use_decay:
                print(f"\n💡 温度从{temp}衰减到{temp * (temp_decay ** len(new_part)):.2f}")
            
        except Exception as e:
            print(f"❌ 错误: {e}")
