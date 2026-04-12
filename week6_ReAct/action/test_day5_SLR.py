import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from llm_client import KimiClient
from react_agent_v3 import ReActAgentV3
from tools.base import ToolRegistry
from tools.real_tools import get_current_year, python_calculator
from tools.python_sandbox import python_sandbox

def test_day5_scenario_1():
    """场景1：短期记忆测试（基础计算）"""
    print("=" * 60)
    print("场景1：短期记忆 - 连续计算")
    print("=" * 60)
    
    client = KimiClient(api_key=os.getenv("KIMI_API_KEY"))
    registry = ToolRegistry()
    registry.register(get_current_year).register(python_calculator)
    
    agent = ReActAgentV3(client, registry, memory_k=3)
    
    print("\n--- 第一轮：获取年份 ---")  
    q1 = "今年是哪一年？"
    print(f"用户：{q1}")
    r1 = agent.run(q1)
    print(f"Agent：{r1}")
    
    print("\n--- 第二轮：计算出生年份 ---")
    q2 = "我今年30岁，哪年出生的？"
    print(f"用户：{q2}")
    r2 = agent.run(q2)
    print(f"Agent：{r2}")
    
    print("\n--- 第三轮：计算未来年龄 ---")
    q3 = "到2050年我多少岁？"
    print(f"用户：{q3}")
    r3 = agent.run(q3)
    print(f"Agent：{r3}")
    
    print(f"\n[记忆状态]\n{agent.get_memory_debug()}")

def test_day5_scenario_2():
    """场景2：长期记忆测试（跨会话持久化 + 动态开关）"""
    print("\n" + "=" * 60)
    print("场景2：长期记忆 - 跨会话持久化与动态开关")
    print("=" * 60)
    
    client = KimiClient(api_key=os.getenv("KIMI_API_KEY"))
    registry = ToolRegistry()
    registry.register(python_calculator)
    
    # 使用优化参数：更高阈值，更少返回数量
    agent1 = ReActAgentV3(client, registry, memory_k=2, ltm_threshold=0.4, ltm_max_facts=3)
    
    print("\n--- 会话1：建立个人信息 ---")
    q1 = "我叫张三，今年30岁，出生于1995年，在腾讯工作，喜欢编程"
    print(f"用户：{q1}")
    r1 = agent1.run(q1)
    print(f"Agent：{r1}")
    
    # 演示动态开关长期记忆
    print("\n--- 测试：关闭长期记忆 ---")
    q_switch = "关闭长期记忆"
    print(f"用户：{q_switch}")
    r_switch = agent1.run(q_switch)
    print(f"Agent：{r_switch}")
    
    # 再开启
    print("\n--- 测试：开启长期记忆 ---")
    q_switch2 = "开启长期记忆"
    print(f"用户：{q_switch2}")
    r_switch2 = agent1.run(q_switch2)
    print(f"Agent：{r_switch2}")
    
    # 模拟新会话（新建代理，但共享长期记忆）
    agent2 = ReActAgentV3(client, registry, memory_k=2)
    agent2.clear_short_memory()  # 清空短期记忆
    
    print("\n--- 会话2：验证长期记忆（新会话） ---")
    q2 = "你还记得我叫什么名字吗？"
    print(f"用户：{q2}")
    r2 = agent2.run(q2)
    print(f"Agent：{r2}")
    
    q3 = "我是哪年出生的？"
    print(f"用户：{q3}")
    r3 = agent2.run(q3)
    print(f"Agent：{r3}")
    
    q4 = "我在哪里工作？"
    print(f"用户：{q4}")
    r4 = agent2.run(q4)
    print(f"Agent：{r4}")

def test_day5_scenario_3():
    """场景3：反思引擎测试（自动提取事实 + 冲突检测）"""
    print("\n" + "=" * 60)
    print("场景3：反思引擎 - 自动提取关键事实与冲突检测")
    print("=" * 60)
    
    client = KimiClient(api_key=os.getenv("KIMI_API_KEY"))
    registry = ToolRegistry()
    registry.register(get_current_year).register(python_calculator)
    
    agent = ReActAgentV3(client, registry, memory_k=3)
    
    print("\n--- 第一轮：提供个人信息 ---")
    q1 = "我今年32岁，在腾讯工作，月薪5万元，喜欢打篮球"
    print(f"用户：{q1}")
    r1 = agent.run(q1)
    print(f"Agent：{r1}")
    
    print("\n--- 第二轮：验证提取效果 ---")
    q2 = "我的月薪是多少？"
    print(f"用户：{q2}")
    r2 = agent.run(q2)
    print(f"Agent：{r2}")
    
    q3 = "我在哪里工作？"
    print(f"用户：{q3}")
    r3 = agent.run(q3)
    print(f"Agent：{r3}")
    
    print("\n--- 第三轮：测试冲突检测（提供矛盾信息） ---")
    q4 = "其实我今年35岁"  # 与之前的32岁冲突
    print(f"用户：{q4}")
    r4 = agent.run(q4)
    print(f"Agent：{r4}")
    
    print("\n--- 第四轮：验证冲突处理 ---")
    q5 = "我今年多少岁？"
    print(f"用户：{q5}")
    r5 = agent.run(q5)
    print(f"Agent：{r5}")
    
    print(f"\n[完整记忆状态]\n{agent.get_memory_debug()}")

def test_day5_scenario_4():
    """场景4：记忆窗口测试（旧信息遗忘 + 长期记忆保留）"""
    print("\n" + "=" * 60)
    print("场景4：短期记忆窗口 - 超过k后遗忘，但长期记忆保留")
    print("=" * 60)
    
    client = KimiClient(api_key=os.getenv("KIMI_API_KEY"))
    registry = ToolRegistry()
    registry.register(python_calculator)
    
    agent = ReActAgentV3(client, registry, memory_k=2)  # 只记2轮
    
    queries = [
        "请记住：我的临时密码是abc123",
        "我的生日是1990年",
        "计算5*5"
    ]
    
    for i, q in enumerate(queries, 1):
        print(f"\n--- 第{i}轮 ---")
        print(f"用户：{q}")
        agent.run(q)
    
    print("\n--- 测试1：验证短期记忆遗忘 ---")
    q4 = "我的临时密码是多少？"
    print(f"用户：{q4}")
    r4 = agent.run(q4)
    print(f"Agent：{r4}")
    
    print("\n--- 测试2：验证长期记忆保留 ---")
    q5 = "我是哪年出生的？"
    print(f"用户：{q5}")
    r5 = agent.run(q5)
    print(f"Agent：{r5}")

if __name__ == "__main__":
    if not os.getenv("KIMI_API_KEY"):
        print("⚠️ 警告：未设置 KIMI_API_KEY")
    
    print("🚀 Day 5 完整记忆系统测试开始（优化版）")
    print("[优化特性] 动态开关、阈值过滤、冲突检测、智能反思")
    
    try:
        test_day5_scenario_1()
    except Exception as e:
        print(f"\n❌ 场景1失败: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test_day5_scenario_2()
    except Exception as e:
        print(f"\n❌ 场景2失败: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test_day5_scenario_3()
    except Exception as e:
        print(f"\n❌ 场景3失败: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test_day5_scenario_4()
    except Exception as e:
        print(f"\n❌ 场景4失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 Day 5 测试完成")
