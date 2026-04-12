"""
ReAct Agent V3 交互式演示程序
支持会话管理和实时对话
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from llm_client import KimiClient
from react_agent_v3 import ReActAgentV3
from tools.base import ToolRegistry
from tools.real_tools import get_current_year, python_calculator

def print_help():
    """显示帮助信息"""
    print("=" * 60)
    print("ReAct Agent V3 交互式演示")
    print("=" * 60)
    print("命令列表:")
    print("  q           - 退出程序")
    print("  n           - 创建新会话（重置所有记忆）")
    print("  o --id xxx  - 切换到指定会话ID")
    print("  ls          - 列出所有可用会话")
    print("  info        - 显示当前会话信息和记忆状态")
    print("  help        - 显示此帮助信息")
    print("  其他内容    - 作为问题发送给Agent")
    print("=" * 60)

def main():
    # 检查API密钥
    if not os.getenv("KIMI_API_KEY"):
        print("⚠️ 警告：未设置 KIMI_API_KEY")
        api_key = input("请输入API密钥: ").strip()
        if not api_key:
            print("错误：API密钥不能为空")
            return
        os.environ["KIMI_API_KEY"] = api_key
    
    # 初始化Agent
    client = KimiClient(api_key=os.getenv("KIMI_API_KEY"))
    registry = ToolRegistry()
    registry.register(get_current_year).register(python_calculator)
    
    agent = ReActAgentV3(client, registry, memory_k=5, ltm_threshold=0.3, ltm_max_facts=3)
    
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
            
        except KeyboardInterrupt:
            print("\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
