"""
forcast_cn.py - 中文文本生成交互工具
支持: RMSNorm + RoPE + SwiGLU + KV-Cache
适配chargpt_cn.py训练的中文模型
"""

import torch
import os
import sys
from mingpt.im_model import GPT
from mingpt.utils import CfgNode as CN


def rebuild_vocab(input_file='input_cn.txt'):
    """从原始文本重建字符↔索引映射"""
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"需要{input_file}来重建词表！请将训练用的文本放在同级目录")
    
    text = open(input_file, 'r', encoding='utf-8').read()
    chars = sorted(list(set(text)))
    
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    print(f"词表重建完成: {len(chars)}个独特字符")
    return stoi, itos, len(chars)


def load_model(checkpoint_path='./out/chargpt/im_model.pt', input_file='input_cn.txt', device='auto'):
    """加载中文模型并重建词表"""
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    stoi, itos, vocab_size = rebuild_vocab(input_file)
    
    config = GPT.get_default_config()
    config.model_type = 'gopher-44m'  # 与chargpt_cn.py一致
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
    生成中文文本
    
    参数:
        prompt: 提示文本（中文）
        max_new_tokens: 生成长度
        temperature: 温度
        do_sample: 是否采样
        top_k: Top-k采样
        top_p: Top-p (Nucleus) 采样
        temperature_decay: 温度衰减
        use_cache: 是否使用KV-Cache加速
        device: 计算设备
    """
    # 检查字符是否在词表中
    invalid_chars = [c for c in prompt if c not in stoi]
    if invalid_chars:
        raise ValueError(f"提示语包含未见过字符: {invalid_chars[:5]}...")
    
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


def continue_writing(model, stoi, itos, prompt, num_continuations=3, device='cpu'):
    """
    多路径续写：为同一提示生成多个不同续写
    
    参数:
        prompt: 起始文本
        num_continuations: 生成续写数量
        device: 计算设备
    """
    print(f"\n{'='*60}")
    print(f"多路径续写: '{prompt}'")
    print(f"{'='*60}")
    
    for i in range(num_continuations):
        torch.manual_seed(42 + i)  # 不同种子产生不同结果
        
        result = generate_text(
            model, stoi, itos, prompt,
            max_new_tokens=300,
            temperature=0.8,
            top_k=10,
            top_p=0.9,
            device=device
        )
        
        continuation = result[len(prompt):]
        print(f"\n【续写 {i+1}】:")
        print(continuation[:150] + ('...' if len(continuation) > 150 else ''))


def style_transfer(model, stoi, itos, prompt, styles, device='cpu'):
    """
    风格迁移：用不同风格重写/续写文本
    
    参数:
        prompt: 原始文本
        styles: 风格列表，如 ['古典', '现代', '诗意']
        device: 计算设备
    """
    print(f"\n{'='*60}")
    print(f"风格迁移: '{prompt}'")
    print(f"{'='*60}")
    
    # 为每种风格设置不同参数
    style_configs = {
        '古典': {'temperature': 0.6, 'top_p': 0.85},
        '现代': {'temperature': 1.0, 'top_p': 0.95},
        '诗意': {'temperature': 1.2, 'top_p': 0.9},
        '幽默': {'temperature': 1.3, 'top_p': 0.92},
        '严肃': {'temperature': 0.5, 'top_p': 0.8},
    }
    
    for style in styles:
        config = style_configs.get(style, {'temperature': 0.8, 'top_p': 0.9})
        
        result = generate_text(
            model, stoi, itos, prompt,
            max_new_tokens=200,
            temperature=config['temperature'],
            top_p=config['top_p'],
            device=device
        )
        
        continuation = result[len(prompt):]
        print(f"\n【{style}风格】 (temp={config['temperature']}, top_p={config['top_p']}):")
        print(continuation[:120] + ('...' if len(continuation) > 120 else ''))


def interactive_story(model, stoi, itos, starter, rounds=3, device='cpu'):
    """
    交互式故事接龙：模型和用户轮流续写
    
    参数:
        starter: 故事开头
        rounds: 接龙轮数
        device: 计算设备
    """
    story = starter
    print(f"\n{'='*60}")
    print("交互式故事接龙")
    print(f"{'='*60}")
    print(f"【开头】{story}")
    
    for i in range(rounds):
        # AI续写
        ai_continuation = generate_text(
            model, stoi, itos, story,
            max_new_tokens=100,
            temperature=0.9,
            top_p=0.92,
            device=device
        )
        ai_part = ai_continuation[len(story):]
        story += ai_part
        print(f"\n【AI续写 {i+1}】{ai_part}")
        
        # 用户输入
        user_input = input(f"\n【你的续写 {i+1}】(直接回车结束): ").strip()
        if user_input:
            story += user_input
        else:
            break
    
    print(f"\n{'='*60}")
    print("【完整故事】")
    print(story)
    print(f"{'='*60}")


def generate_with_constraints(model, stoi, itos, prompt, must_include, device='cpu'):
    """
    约束生成：生成的文本必须包含指定关键词
    
    参数:
        prompt: 提示文本
        must_include: 必须包含的字符列表
        device: 计算设备
    """
    print(f"\n{'='*60}")
    print(f"约束生成: '{prompt}'")
    print(f"必须包含: {must_include}")
    print(f"{'='*60}")
    
    max_attempts = 10
    for attempt in range(max_attempts):
        result = generate_text(
            model, stoi, itos, prompt,
            max_new_tokens=300,
            temperature=1.0 + attempt * 0.1,  # 逐渐增加随机性
            top_p=0.95,
            device=device
        )
        
        continuation = result[len(prompt):]
        
        # 检查是否包含所有关键词
        missing = [c for c in must_include if c not in continuation]
        
        if not missing:
            print(f"\n✓ 尝试 {attempt+1} 成功！")
            print(f"生成结果: {continuation[:200]}")
            return result
        else:
            print(f"尝试 {attempt+1}: 缺少 {missing}")
    
    print(f"\n✗ 未能在{max_attempts}次尝试内满足所有约束")
    return None


def main():
    """主程序：中文生成交互界面"""
    model, stoi, itos, device = load_model()
    
    print("\n" + "="*60)
    print("中文文本生成器 (基于chargpt_cn.py训练)")
    print("="*60)
    print("支持功能:")
    print("  1. 单条生成 - 输入提示语直接生成")
    print("  2. 多路径续写 - 同一提示多种可能")
    print("  3. 风格迁移 - 古典/现代/诗意/幽默/严肃")
    print("  4. 故事接龙 - 人机交互续写")
    print("  5. 约束生成 - 必须包含指定字符")
    print("  6. 对比实验 - 不同采样策略对比")
    print("  q. 退出")
    print("="*60)
    
    while True:
        choice = input("\n选择功能 (1-6/q): ").strip()
        
        if choice.lower() in ['q', 'quit', 'exit']:
            print("再见！")
            break
        
        if choice == '1':
            # 单条生成
            prompt = input("输入提示语: ").strip()
            if not prompt:
                continue
            
            try:
                length = int(input("生成长度 (默认200): ") or "200")
                temp = float(input("温度 (默认0.8): ") or "0.8")
                
                result = generate_text(
                    model, stoi, itos, prompt,
                    max_new_tokens=length,
                    temperature=temp,
                    top_k=10,
                    top_p=0.9,
                    device=device
                )
                
                print(f"\n【生成结果】:")
                print(result)
            except Exception as e:
                print(f"错误: {e}")
        
        elif choice == '2':
            # 多路径续写
            prompt = input("输入起始文本: ").strip()
            if not prompt:
                continue
            
            num = int(input("续写数量 (默认3): ") or "3")
            continue_writing(model, stoi, itos, prompt, num, device)
        
        elif choice == '3':
            # 风格迁移
            prompt = input("输入原始文本: ").strip()
            if not prompt:
                continue
            
            print("可选风格: 古典, 现代, 诗意, 幽默, 严肃")
            styles_input = input("选择风格 (用空格分隔): ").strip()
            styles = styles_input.split() if styles_input else ['古典', '现代']
            
            style_transfer(model, stoi, itos, prompt, styles, device)
        
        elif choice == '4':
            # 故事接龙
            starter = input("输入故事开头: ").strip()
            if not starter:
                continue
            
            rounds = int(input("接龙轮数 (默认3): ") or "3")
            interactive_story(model, stoi, itos, starter, rounds, device)
        
        elif choice == '5':
            # 约束生成
            prompt = input("输入提示语: ").strip()
            if not prompt:
                continue
            
            constraints = input("必须包含的字符 (用空格分隔): ").strip().split()
            if not constraints:
                print("请至少输入一个约束字符")
                continue
            
            generate_with_constraints(model, stoi, itos, prompt, constraints, device)
        
        elif choice == '6':
            # 对比实验
            prompt = input("输入测试提示语: ").strip() or "宝玉笑道："
            
            strategies = [
                ("贪婪解码", {"temperature": 1.0, "do_sample": False, "top_k": None, "top_p": None}),
                ("Top-k(k=10)", {"temperature": 1.0, "do_sample": True, "top_k": 10, "top_p": None}),
                ("Top-p(p=0.9)", {"temperature": 1.0, "do_sample": True, "top_k": None, "top_p": 0.9}),
                ("高温创意(1.2)", {"temperature": 1.2, "do_sample": True, "top_k": None, "top_p": 0.95}),
            ]
            
            print(f"\n{'='*60}")
            print(f"对比实验: '{prompt}'")
            print(f"{'='*60}")
            
            for name, kwargs in strategies:
                torch.manual_seed(42)
                result = generate_text(model, stoi, itos, prompt, max_new_tokens=200, device=device, **kwargs)
                continuation = result[len(prompt):]
                print(f"\n【{name}】:")
                print(continuation[:100] + ('...' if len(continuation) > 100 else ''))
        
        else:
            print("无效选择，请重新输入")


if __name__ == '__main__':
    main()
