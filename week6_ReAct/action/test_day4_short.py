import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from llm_client import KimiClient
from react_agent_v2 import ReActAgentV2
from tools.base import ToolRegistry
from tools.real_tools import get_current_year, python_calculator
from tools.python_sandbox import python_sandbox

def test_day4_scenario_1():
    """场景1：记忆重用（不再重复查询年份）"""
    print("=" * 60)
    print("场景1：记忆重用（关键事实提取）")
    print("=" * 60)
    
    client = KimiClient(api_key=os.getenv("KIMI_API_KEY"))
    registry = ToolRegistry()
    registry.register(get_current_year).register(python_calculator)
    
    agent = ReActAgentV2(client, registry, memory_k=3)
    
    print("\n--- 第一轮：获取年份 ---")
    q1 = "今年是哪一年？"
    print(f"用户：{q1}")
    r1 = agent.run(q1)
    print(f"Agent：{r1}")
    print(f"[记忆状态] {agent.get_memory_debug()}")
    
    print("\n--- 第二轮：使用记忆中的年份 ---")
    q2 = "我今年30岁，哪年出生的？"
    print(f"用户：{q2}")
    r2 = agent.run(q2)
    print(f"Agent：{r2}")
    
    print("\n--- 第三轮：继续利用历史 ---") 
    q3 = "到2050年我多少岁？"
    print(f"用户：{q3}")
    r3 = agent.run(q3)
    print(f"Agent：{r3}")

def test_day4_scenario_2():
    """场景2：记忆窗口（旧对话被遗忘）"""
    print("\n" + "=" * 60)
    print("场景2：记忆窗口限制（k=2）")
    print("=" * 60)
    
    client = KimiClient(api_key=os.getenv("KIMI_API_KEY"))
    registry = ToolRegistry()
    registry.register(python_calculator)
    
    agent = ReActAgentV2(client, registry, memory_k=2)
    
    queries = [
        "请记住：我的密码是123456",
        "我的生日是1995年",
        "计算10+20"
    ]
    
    for i, q in enumerate(queries, 1):
        print(f"\n--- 第{i}轮 ---")
        print(f"用户：{q}")
        r = agent.run(q)
        print(f"Agent：{r}")
    
    print("\n--- 第四轮：测试是否遗忘密码 ---")
    r4 = agent.run("我的密码是多少？")
    print(f"Agent：{r4}")

if __name__ == "__main__":
    if not os.getenv("KIMI_API_KEY"):
        print("⚠️ 警告：未设置 KIMI_API_KEY")
        print("测试将使用 Mock 模式运行")
    
    print("🚀 Day 4 进阶记忆测试开始")
    
    try:
        test_day4_scenario_1()
    except Exception as e:
        print(f"场景1失败: {e}")
    
    try:
        test_day4_scenario_2()
    except Exception as e:
        print(f"场景2失败: {e}")
    
    print("\n🎉 Day 4 测试完成")