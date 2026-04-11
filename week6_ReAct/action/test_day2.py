from llm_client import KimiClient
from react_agent_v2 import ReActAgentV2
from tools.base import ToolRegistry
from tools.real_tools import search_duckduckgo, python_calculator, get_current_year, master_test
import os

def test_function_calling_agent():
    print("=" * 60)
    print("Day 2 测试：Kimi Function Calling + 真实工具")
    print("=" * 60)
    
    # 1. 初始化 Kimi 客户端
    client = KimiClient(
        api_key=os.getenv("KIMI_API_KEY"),
        model="moonshot-v1-32k"  # 32k 足够应对 ReAct 多轮
    )
    
    # 2. 注册工具
    registry = ToolRegistry()
    registry.register(search_duckduckgo) \
            .register(python_calculator) \
            .register(get_current_year) \
            .register(master_test)
    
    print(f"已注册工具：{list(registry._tools.keys())}")
    
    # 3. 初始化 Agent
    agent = ReActAgentV2(
        llm_client=client,
        tool_registry=registry,
        max_iterations=5
    )
    
    # 4. 测试：需要多步推理的问题
    """query = "Python 创始人 Guido van Rossum 今年多少岁？请搜索他的出生年份和当前年份，然后计算。"
    
    print(f"\n问题：{query}")
    print("开始执行...\n")
    
    result = agent.run(query)
    print(f"\n最终答案：{result}")
    
    # 5. 验证并行调用（如果 Kimi 支持一次返回多个 tool_calls）
    print("\n" + "=" * 60)
    print("测试并行工具调用：")
    query2 = "同时搜索 'Python 语言特点' 和计算 '256 * 16'，然后总结"
    result2 = agent.run(query2)
    print(f"并行测试结果：{result2}")"""
    
    # 6. 测试：主工具调用
    query3 = "请测试主工具"
    result3 = agent.run(query3)
    print(f"主工具测试结果：{result3}")

if __name__ == "__main__":
    test_function_calling_agent()
