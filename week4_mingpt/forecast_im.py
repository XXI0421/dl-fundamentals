"""
forecast_im.py - 加载训练好的im_model (LLaMA风格) 进行文本生成
支持: RMSNorm + RoPE + SwiGLU + KV-Cache
"""

import torch
import os
from mingpt.im_model import GPT
from mingpt.utils import CfgNode as CN

def rebuild_vocab(input_file='input.txt'):
    """从原始文本重建字符↔索引映射"""
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"需要{input_file}来重建词表！请将训练用的文本放在同级目录")
    
    text = open(input_file, 'r', encoding='utf-8').read()
    chars = sorted(list(set(text)))
    
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    print(f"词表重建完成: {len(chars)}个独特字符")
    return stoi, itos, len(chars)

def load_model(checkpoint_path='./out/chargpt/im_model.pt', input_file='input.txt', device='auto'):
    """加载im_model模型并重建词表"""
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    stoi, itos, vocab_size = rebuild_vocab(input_file)
    
    config = GPT.get_default_config()
    config.model_type = 'gpt-mini'
    config.vocab_size = vocab_size
    config.block_size = 128
    
    model = GPT(config)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        print("检测到增强版检查点格式（含完整配置）")
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'stoi' in checkpoint and 'itos' in checkpoint:
            stoi = checkpoint['stoi']
            itos = checkpoint['itos']
            print("使用检查点内嵌词表")
    else:
        print("检测到原版检查点格式（仅权重）")
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print(f"模型加载成功 (LLaMA风格: RMSNorm + RoPE + SwiGLU + KV-Cache)")
    print(f"   参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    return model, stoi, itos, device

def generate_text(model, stoi, itos, prompt, max_new_tokens=200, 
                  temperature=1.0, do_sample=True, top_k=10, top_p=None, 
                  temperature_decay=1.0, use_cache=True, device='cpu'):
    """
    生成文本 (im_model版本，支持KV-Cache加速)
    
    参数:
        prompt: 提示文本
        max_new_tokens: 生成长度
        temperature: 温度
        do_sample: 是否采样
        top_k: Top-k采样
        top_p: Top-p (Nucleus) 采样
        temperature_decay: 温度衰减
        use_cache: 是否使用KV-Cache加速 (默认True)
        device: 计算设备
    """
    if any(c not in stoi for c in prompt):
        invalid = [c for c in prompt if c not in stoi]
        raise ValueError(f"提示语包含未见过字符: {invalid[:5]}...")
    
    idx = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long).to(device)
    
    with torch.no_grad():
        generated = model.generate(
            idx, 
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature_decay=temperature_decay,
            use_cache=use_cache
        )[0]
    
    text = ''.join([itos[int(i)] for i in generated])
    return text

def compare_strategies(model, stoi, itos, prompt, device='cpu'):
    """对比不同生成策略的效果"""
    strategies = [
        ("贪婪解码", {"temperature": 1.0, "do_sample": False, "top_k": None, "top_p": None, "temperature_decay": 1.0}),
        ("Top-k(k=10)", {"temperature": 1.0, "do_sample": True, "top_k": 10, "top_p": None, "temperature_decay": 1.0}),
        ("Top-p(p=0.9)", {"temperature": 1.0, "do_sample": True, "top_k": None, "top_p": 0.9, "temperature_decay": 1.0}),
        ("高温创意(1.3)", {"temperature": 1.3, "do_sample": True, "top_k": None, "top_p": 0.95, "temperature_decay": 1.0}),
        ("温度衰减(1.2->0.3)", {"temperature": 1.2, "do_sample": True, "top_k": None, "top_p": 0.9, "temperature_decay": 0.97}),
    ]
    
    print(f"\n{'='*60}")
    print(f"提示语: '{prompt}'")
    print(f"{'='*60}")
    
    for name, kwargs in strategies:
        torch.manual_seed(42)
        
        result = generate_text(model, stoi, itos, prompt, max_new_tokens=500, device=device, **kwargs)
        new_part = result[len(prompt):]
        
        print(f"\n【{name}】:")
        print(f"{new_part[:120]}{'...' if len(new_part) > 120 else ''}")

def benchmark_cache(model, stoi, itos, prompt, device='cpu'):
    """对比KV-Cache开启/关闭的速度"""
    import time
    
    idx = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long).to(device)
    
    # 无缓存
    model.eval()
    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.time()
    with torch.no_grad():
        for _ in range(3):
            _ = model.generate(idx, max_new_tokens=100, use_cache=False, do_sample=False)
    torch.cuda.synchronize() if device == 'cuda' else None
    time_no_cache = (time.time() - start) / 3
    
    # 有缓存
    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.time()
    with torch.no_grad():
        for _ in range(3):
            _ = model.generate(idx, max_new_tokens=100, use_cache=True, do_sample=False)
    torch.cuda.synchronize() if device == 'cuda' else None
    time_with_cache = (time.time() - start) / 3
    
    print(f"\n{'='*40}")
    print(f"KV-Cache 性能测试 (生成100 tokens)")
    print(f"{'='*40}")
    print(f"无缓存: {time_no_cache:.3f}s")
    print(f"有缓存: {time_with_cache:.3f}s")
    print(f"加速比: {time_no_cache/time_with_cache:.2f}x")
    print(f"{'='*40}")

if __name__ == '__main__':
    model, stoi, itos, device = load_model()
    
    print("\nim_model 交互式生成器 (LLaMA风格)")
    print("支持: Top-k | Top-p | 温度衰减 | KV-Cache加速")
    print("命令: 'compare'=对比实验, 'bench'=速度测试, 'quit'=退出")
    
    while True:
        prompt = input("\nPrompt (或命令): ").strip()
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            break
        
        if prompt.lower() == 'compare':
            test_prompt = input("输入对比测试的提示语: ").strip() or "To be, or not to be"
            compare_strategies(model, stoi, itos, test_prompt, device)
            continue
        
        if prompt.lower() == 'bench':
            test_prompt = input("输入测试提示语: ").strip() or "O God, O God!"
            benchmark_cache(model, stoi, itos, test_prompt, device)
            continue
        
        if not prompt:
            continue
            
        try:
            use_top_p = input("使用Top-p采样? (y/n, 默认n): ").lower() == 'y'
            top_p = 0.9 if use_top_p else None
            
            use_decay = input("使用温度衰减? (y/n, 默认n): ").lower() == 'y'
            temp_decay = 0.97 if use_decay else 1.0
            
            temp = float(input("温度 (默认1.0): ") or "1.0")
            
            result = generate_text(
                model, stoi, itos, prompt, 
                max_new_tokens=200,
                temperature=temp,
                top_k=10 if not use_top_p else None,
                top_p=top_p,
                temperature_decay=temp_decay,
                device=device
            )
            
            new_part = result[len(prompt):]
            print(f"\n续写 ({len(new_part)}字符):")
            print(new_part)
            
        except Exception as e:
            print(f"错误: {e}")
