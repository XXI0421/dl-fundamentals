import os
import sys
from pathlib import Path

# 确保路径正确
sys.path.insert(0, str(Path(__file__).parent))

from llm_client import KimiClient
from react_agent_v2 import ReActAgentV2
from tools.base import ToolRegistry
from tools.real_tools import search_duckduckgo, get_current_year
from tools.retriever_tool import retriever_tool, knowledge_base_status
from tools.python_sandbox import python_sandbox, calculator

def test_day3_scenario_1():
    """场景1：知识库检索 + 计算（Week 5 RAG + Week 6 Agent）"""
    print("=" * 60)
    print("场景1：从知识库检索 ReAct 定义，并计算相关数值")
    print("=" * 60)
    
    # 1. 初始化
    client = KimiClient(api_key=os.getenv("KIMI_API_KEY"))
    registry = ToolRegistry()
    
    # 2. 注册 Day 3 工具集
    registry.register(retriever_tool) \
            .register(calculator) \
            .register(python_sandbox) \
            .register(knowledge_base_status)
    
    print(f"\n已注册工具：{registry.list_tools()}")
    
    # 3. 先检查知识库状态（诊断）
    print("\n[诊断] 检查知识库状态...")
    status = knowledge_base_status.func()  # 使用默认路径
    print(status)
    
    # 4. 执行复杂任务
    agent = ReActAgentV2(llm_client=client, tool_registry=registry, max_iterations=5)
    
    query = """
    请完成以下任务：
    1. 从知识库检索 "ReAgent" 相关的定义和描述（top_k=3）
    2. 如果检索到相关内容，统计描述中出现了多少次 "Agent" 这个词（使用 Python 代码计算）
    3. 给出最终总结：ReAct 是什么？出现的 Agent 次数是多少？
    """
    
    print(f"\n[任务] {query}")
    result = agent.run(query)
    print(f"\n[最终结果] {result}")

def test_day3_scenario_2():
    """场景2：安全沙箱测试（危险代码拦截）"""
    print("\n" + "=" * 60)
    print("场景2：安全沙箱拦截测试")
    print("=" * 60)
    
    # 测试危险代码
    dangerous_codes = [
        "import os; os.system('ls')",
        "open('/etc/passwd').read()",
        "while True: pass",  # 超时测试
        "__import__('subprocess').call(['whoami'])"
    ]
    
    for code in dangerous_codes:
        print(f"\n[测试代码] {code[:50]}...")
        result = python_sandbox.func(code, timeout=3)
        if "安全拦截" in result or "超时" in result:
            print("✅ 安全机制正常拦截")
        else:
            print(f"⚠️ 未拦截: {result[:100]}")

def test_day3_scenario_3():
    """场景3：多工具协作（搜索 + 检索 + 计算）"""
    print("\n" + "=" * 60)
    print("场景3：多工具协作验证")
    print("=" * 60)
    
    client = KimiClient(api_key=os.getenv("KIMI_API_KEY"))
    registry = ToolRegistry()
    
    # 同时注册搜索和检索（网络 + 本地知识库）
    registry.register(search_duckduckgo) \
            .register(retriever_tool) \
            .register(calculator)
    
    agent = ReActAgentV2(llm_client=client, tool_registry=registry, max_iterations=6)
    
    query = """
    对比分析任务：
    1. 从本地知识库检索 "ReAct" 的定义
    2. 同时搜索网络上 "ReAct Agent" 的最新应用案例
    3. 总结两者的信息差异（如果有）
    """
    
    print(f"[任务] {query}")
    result = agent.run(query)
    print(f"\n[最终结果] {result}")

if __name__ == "__main__":
    # 检查环境
    if not os.getenv("KIMI_API_KEY"):
        print("⚠️ 警告：未设置 KIMI_API_KEY，请设置环境变量")
        print("测试将使用 Mock 模式运行（仅验证工具逻辑）")
    
    print("🚀 Day 3 集成测试开始")
    
    try:
        test_day3_scenario_1()
    except Exception as e:
        print(f"场景1失败: {e}")
    
    try:
        test_day3_scenario_2()
    except Exception as e:
        print(f"场景2失败: {e}")
    
    try:
        test_day3_scenario_3()
    except Exception as e:
        print(f"场景3失败: {e}")
    
    print("\n🎉 Day 3 测试完成")
