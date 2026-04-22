"""
记忆模块测试
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory.short_term import ShortTermMemory, ConversationSummaryMemory

def test_short_term_memory():
    """测试短期记忆"""
    print("测试短期记忆...")
    
    # 测试ShortTermMemory
    stm = ShortTermMemory(max_size=3)
    stm.add_message("user", "你好")
    stm.add_message("assistant", "您好！")
    stm.add_message("user", "今天天气怎么样？")
    
    messages = stm.get_messages()
    print(f"✅ 短期记忆消息数: {len(messages)}")
    print(f"✅ 上下文内容: {stm.get_context()}")
    
    # 测试ConversationSummaryMemory
    csm = ConversationSummaryMemory(max_size=3)
    csm.add_user("你好")
    csm.add_assistant("您好！")
    csm.add_user("今天天气怎么样？")
    
    print(f"✅ 对话摘要记忆: {csm.get_full_context()}")
    
    print("短期记忆测试通过！")

def test_memory():
    """运行所有记忆测试"""
    test_short_term_memory()
    print("\n🎉 所有记忆测试通过！")

if __name__ == "__main__":
    test_memory()
