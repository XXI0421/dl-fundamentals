"""
ReAct Agent测试
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from react_agent import ReActAgent
from tools.base import ToolRegistry
from tools.real_tools import init_default_tools
from tools.python_sandbox import init_sandbox_tools
from llm_client import get_llm_client
from config import Config

def test_xml_parsing():
    """测试XML解析功能"""
    print("测试XML解析功能...")
    
    # 创建Agent实例（不调用LLM）
    registry = ToolRegistry()
    init_default_tools(registry)
    
    # 创建一个模拟Agent来测试解析功能
    agent = ReActAgent(
        llm_client=None,
        tools=registry,
        max_iterations=1
    )
    
    # 测试1: 标准XML格式（正常情况）
    test_content1 = '<tool_call><function name="save_file"><arguments><content>Hello World</content><filename>test.txt</filename></arguments></function></tool_call>'
    tool_calls1 = agent._parse_tool_calls(test_content1)
    print(f"测试1 - 标准XML: 解析出 {len(tool_calls1)} 个工具调用")
    
    # 测试2: 包含特殊字符的XML
    test_content2 = '<tool_call><function name="save_file"><arguments><content><div id="app">Hello</div></content><filename>test.html</filename></arguments></function></tool_call>'
    tool_calls2 = agent._parse_tool_calls(test_content2)
    print(f"测试2 - 特殊字符: 解析出 {len(tool_calls2)} 个工具调用")
    
    # 测试3: 多个连续工具调用
    test_content3 = '<tool_call><function name="save_file"><arguments><content>File 1</content><filename>file1.txt</filename></arguments></function></tool_call> <tool_call><function name="save_file"><arguments><content>File 2</content><filename>file2.txt</filename></arguments></function></tool_call>'
    tool_calls3 = agent._parse_tool_calls(test_content3)
    print(f"测试3 - 连续工具调用: 解析出 {len(tool_calls3)} 个工具调用")
    
    if tool_calls1:
        print(f"✅ 工具名称: {tool_calls1[0].get('name')}")
        args = tool_calls1[0].get('arguments', {})
        print(f"✅ 参数filename: {args.get('filename')}")
        print(f"✅ 参数content: {args.get('content')[:30]}...")
    
    print("XML解析测试完成！")

def test_react_agent():
    """测试ReAct Agent"""
    print("测试ReAct Agent...")
    
    # 测试XML解析
    test_xml_parsing()
    print()
    
    # 检查API密钥
    if not Config.is_llm_configured():
        print("⚠️ 未配置LLM API密钥，跳过LLM测试")
        return
    
    try:
        # 创建工具注册表
        registry = ToolRegistry()
        init_default_tools(registry)
        init_sandbox_tools(registry)
        
        # 创建LLM客户端
        llm_client = get_llm_client()
        
        # 创建Agent
        agent = ReActAgent(
            llm_client=llm_client,
            tools=registry,
            max_iterations=3,
            verbose=True
        )
        
        print(f"✅ Agent创建成功")
        
        # 测试简单问题
        response = agent.run("你好，请问你是谁？")
        print(f"✅ 测试响应: {response[:100]}...")
        
        # 测试工具调用
        response = agent.run("计算 2 的 10 次方")
        print(f"✅ 测试工具调用: {response[:100]}...")
        
        # 测试记忆调试
        debug_info = agent.get_memory_debug()
        print(f"✅ 记忆调试信息: {debug_info}")
        
        # 测试统计信息
        stats = agent.get_stats()
        print(f"✅ 工具调用统计: {stats}")
        
        print("\n🎉 ReAct Agent测试通过！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_react_agent()
