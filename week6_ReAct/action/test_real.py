import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from llm_client import KimiClient
from react_agent_v3 import ReActAgentV3
from tools.base import ToolRegistry
from tools.real_tools import get_current_year, python_calculator
from tools.python_sandbox import python_sandbox

def test_real_world_scenario():
    """真实场景测试：用户请求Python代码并迭代修改"""
    print("🚀 真实场景测试开始：用户请求Python代码开发")
    print("=" * 70)
    
    client = KimiClient(api_key=os.getenv("KIMI_API_KEY"))
    registry = ToolRegistry()
    registry.register(get_current_year).register(python_calculator).register(python_sandbox)
    
    # 使用优化参数初始化
    agent = ReActAgentV3(client, registry, memory_k=5, ltm_threshold=0.4, ltm_max_facts=3)
    
    # === 第一轮：用户提出需求 ===
    print("\n--- 第一轮：用户提出需求 ---")
    q1 = "帮我写一个Python脚本，读取CSV文件并绘制柱状图"
    print(f"用户：{q1}")
    r1 = agent.run(q1)
    print(f"Agent：{r1[:100]}...")  # 截断显示
    
    # === 第二轮：用户提供具体需求 ===
    print("\n--- 第二轮：提供具体需求 ---")
    q2 = "数据格式是：日期,销售额\n2024-01,1000\n2024-02,1500\n2024-03,1200\n请用matplotlib绘制"
    print(f"用户：{q2}")
    r2 = agent.run(q2)
    print(f"Agent：{r2[:100]}...")
    
    # === 第三轮：测试长期记忆开关 ===
    print("\n--- 第三轮：测试长期记忆开关 ---")
    q_switch = "关闭长期记忆"
    print(f"用户：{q_switch}")
    r_switch = agent.run(q_switch)
    print(f"Agent：{r_switch}")
    
    # === 第四轮：用户提出修改需求（关闭记忆状态）===
    print("\n--- 第四轮：修改需求（记忆关闭） ---")
    q3 = "请把柱状图改成折线图，并添加标题和网格"
    print(f"用户：{q3}")
    r3 = agent.run(q3)
    print(f"Agent：{r3[:100]}...")
    
    # === 第五轮：重新开启长期记忆 ===
    print("\n--- 第五轮：重新开启长期记忆 ---")
    q_switch2 = "开启长期记忆"
    print(f"用户：{q_switch2}")
    r_switch2 = agent.run(q_switch2)
    print(f"Agent：{r_switch2}")
    
    print(f"\n{'-' * 70}")
    print("🎉 真实场景测试完成")

def test_data_analysis_scenario():
    """数据分析师场景：多轮数据分析任务（带优化功能测试）"""
    print("\n" + "=" * 70)
    print("数据分析师场景：多轮数据分析任务")
    print("=" * 70)
    
    client = KimiClient(api_key=os.getenv("KIMI_API_KEY"))
    registry = ToolRegistry()
    registry.register(get_current_year).register(python_calculator).register(python_sandbox)
    
    # 使用优化参数：更高阈值过滤低质量匹配
    agent = ReActAgentV3(client, registry, memory_k=5, ltm_threshold=0.5, ltm_max_facts=3)
    
    # === 第一轮：导入数据 ===
    print("\n--- 第一轮：导入数据 ---")
    q1 = "我有一组销售数据：[120, 150, 180, 200, 220, 190, 210, 250, 280, 300, 320, 350]，代表12个月的销售额"
    print(f"用户：{q1}")
    r1 = agent.run(q1)
    print(f"Agent：{r1}")
    
    # === 第二轮：计算统计量 ===
    print("\n--- 第二轮：计算统计量 ---")   
    q2 = "帮我计算一下总销售额、平均销售额和最大销售额"
    print(f"用户：{q2}")
    r2 = agent.run(q2)
    print(f"Agent：{r2}")
    
    # === 第三轮：测试冲突数据（提供矛盾信息）===
    print("\n--- 第三轮：测试冲突数据 ---")
    q_conflict = "其实我的销售数据应该是从100开始的"
    print(f"用户：{q_conflict}")
    r_conflict = agent.run(q_conflict)
    print(f"Agent：{r_conflict}")
    
    # === 第四轮：验证冲突处理后的结果 ===
    print("\n--- 第四轮：验证数据准确性 ---")
    q_verify = "总销售额是多少？"
    print(f"用户：{q_verify}")
    r_verify = agent.run(q_verify)
    print(f"Agent：{r_verify}")
    
    # === 第五轮：趋势分析 ===
    print("\n--- 第五轮：趋势分析 ---")
    q3 = "分析一下销售趋势，判断是增长还是下降"
    print(f"用户：{q3}")
    r3 = agent.run(q3)
    print(f"Agent：{r3}")
    
    # === 第六轮：预测 ===
    print("\n--- 第六轮：预测 ---")
    q4 = "用线性回归预测下一个月的销售额"
    print(f"用户：{q4}")
    r4 = agent.run(q4)
    print(f"Agent：{r4}")

def test_memory_control_scenario():
    """记忆控制场景：测试各种优化功能"""
    print("\n" + "=" * 70)
    print("记忆控制场景：测试优化功能")
    print("=" * 70)
    
    client = KimiClient(api_key=os.getenv("KIMI_API_KEY"))
    registry = ToolRegistry()
    registry.register(python_calculator)
    
    agent = ReActAgentV3(client, registry, memory_k=2, ltm_threshold=0.3, ltm_max_facts=2)
    
    print("\n--- 测试1：设置个人信息 ---")
    q1 = "我叫李四，今年28岁，在阿里巴巴工作"
    print(f"用户：{q1}")
    agent.run(q1)
    
    print("\n--- 测试2：关闭长期记忆后查询 ---")
    agent.set_long_term_enabled(False)
    q2 = "我叫什么名字？"
    print(f"用户：{q2}")
    r2 = agent.run(q2)
    print(f"Agent（记忆关闭）：{r2}")
    
    print("\n--- 测试3：开启长期记忆后查询 ---")
    agent.set_long_term_enabled(True)
    q3 = "我叫什么名字？"
    print(f"用户：{q3}")
    r3 = agent.run(q3)
    print(f"Agent（记忆开启）：{r3}")
    
    print("\n--- 测试4：验证阈值过滤（低相关查询） ---")
    q4 = "今天天气怎么样？"
    print(f"用户：{q4}")
    r4 = agent.run(q4)
    print(f"Agent：{r4}")

if __name__ == "__main__":
    if not os.getenv("KIMI_API_KEY"):
        print("⚠️ 警告：未设置 KIMI_API_KEY")
    
    print("🚀 真实场景测试开始（优化版）")
    print("[优化特性测试] 动态开关、阈值过滤、冲突检测")
    
    # # 测试1：真实代码开发场景
    # try:
    #     test_real_world_scenario()
    # except Exception as e:
    #     print(f"\n❌ 真实场景测试失败: {e}")
    #     import traceback
    #     traceback.print_exc()
    
    # 测试2：数据分析场景
    try:
        test_data_analysis_scenario()
    except Exception as e:
        print(f"\n❌ 数据分析场景测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试3：记忆控制场景
    try:
        test_memory_control_scenario()
    except Exception as e:
        print(f"\n❌ 记忆控制场景测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 所有测试完成")
