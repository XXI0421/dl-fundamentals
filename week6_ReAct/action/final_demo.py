"""
ReAct Agent V4 交互式演示程序
支持会话管理、实时对话和文件读写操作
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from llm_client import KimiClient
from react_agent_v4 import MultiToolAgent
from tools.base import ToolRegistry
from tools.real_tools import get_current_year, python_calculator, save_file, read_file
from tools.python_sandbox import python_sandbox


def print_help():
    """显示帮助信息"""
    print("=" * 70)
    print("ReAct Agent V4 交互式演示 - 支持文件读写")
    print("=" * 70)
    print("命令列表:")
    print("  q           - 退出程序")
    print("  n           - 创建新会话（重置所有记忆）")
    print("  o --id xxx  - 切换到指定会话ID")
    print("  ls          - 列出所有可用会话")
    print("  info        - 显示当前会话信息和记忆状态")
    print("  help        - 显示此帮助信息")
    print("=" * 70)
    print("可用工具:")
    print("  • save_file     - 保存内容到文件")
    print("  • read_file     - 读取文件内容")
    print("  • python_sandbox - 执行Python代码（支持图表生成）")
    print("  • python_calculator - 数学计算")
    print("  • get_current_year - 获取当前年份")
    print("=" * 70)


def main():
    # 检查API密钥
    if not os.getenv("KIMI_API_KEY"):
        print("⚠️ 警告：未设置 KIMI_API_KEY")
        api_key = input("请输入API密钥: ").strip()
        if not api_key:
            print("错误：API密钥不能为空")
            return
        os.environ["KIMI_API_KEY"] = api_key
    
    # 初始化工具注册表
    registry = ToolRegistry()
    registry.register(get_current_year)
    registry.register(python_calculator)
    registry.register(save_file)
    registry.register(read_file)
    registry.register(python_sandbox)
    
    print(f"📦 已注册工具: {', '.join(registry.list_tools())}")
    
    # 初始化Agent
    client = KimiClient(api_key=os.getenv("KIMI_API_KEY"), model="moonshot-v1-128k")
    agent = MultiToolAgent(
        client, 
        registry, 
        max_iterations=8,
        memory_k=5, 
        ltm_threshold=0.3, 
        ltm_max_facts=3,
        max_retries=3,
        retry_base_delay=1.0,
        retry_multiplier=2.0
    )
    
    print_help()
    print(f"\n🚀 已创建新会话，会话ID: {agent.get_session_id()}")
    
    while True:
        try:
            user_input = input("\n你: ").strip()
            
            if not user_input:
                print("请输入命令或问题（输入 help 查看帮助）")
                continue
            
            # 处理命令
            if user_input.lower() == 'q':
                print("👋 再见！")
                break
            
            elif user_input.lower() == 'n':
                agent.new_session()
                print(f"✅ 已创建新会话，会话ID: {agent.get_session_id()}")
                continue
            
            elif user_input.lower().startswith('o --id '):
                session_id = user_input[7:].strip()
                if session_id:
                    sessions = agent.list_available_sessions()
                    if session_id in sessions:
                        agent.switch_session(session_id)
                        print(f"✅ 已切换到会话: {session_id}")
                    else:
                        print(f"❌ 未找到会话: {session_id}")
                        print(f"可用会话: {', '.join(sessions)}")
                else:
                    print("❌ 请指定会话ID，格式: o --id session_xxx")
                continue
            
            elif user_input.lower() == 'ls':
                sessions = agent.list_available_sessions()
                if sessions:
                    print("📋 可用会话列表:")
                    for s in sessions:
                        marker = " *" if s == agent.get_session_id() else ""
                        print(f"  - {s}{marker}")
                else:
                    print("暂无保存的会话")
                continue
            
            elif user_input.lower() == 'info':
                print("📊 当前会话信息:")
                print(f"  会话ID: {agent.get_session_id()}")
                print(f"  长期记忆: {'开启' if agent.use_long_term else '关闭'}")
                print(f"  反思引擎: {'开启' if agent.use_reflection else '关闭'}")
                print(f"  最大重试次数: {agent.max_retries}")
                print("\n记忆状态:")
                print(agent.get_memory_debug())
                continue
            
            elif user_input.lower() == 'help':
                print_help()
                continue
            
            # 处理普通问题
            print(f"\n🤔 正在处理...")
            response = agent.run(user_input)
            print(f"\n🤖 Agent: {response}")
            
            # 显示工具调用统计
            stats = agent.get_tool_call_stats()
            if stats['total_steps'] > 0:
                print(f"\n📈 本次调用统计:")
                print(f"  步骤数: {stats['total_steps']}, 成功: {stats['successful_steps']}, 失败: {stats['failed_steps']}")
                if stats['tools_used']:
                    print(f"  使用工具: {', '.join(stats['tools_used'])}")
            
        except KeyboardInterrupt:
            print("\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()