"""
LLM客户端测试
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_client import KimiClient, OpenAIClient, get_llm_client
from config import Config

def test_llm_client():
    """测试LLM客户端"""
    print("测试LLM客户端...")
    
    # 检查API密钥
    if not Config.is_llm_configured():
        print("⚠️ 未配置LLM API密钥，跳过测试")
        return
    
    try:
        client = get_llm_client()
        print(f"✅ 成功创建客户端: {client.__class__.__name__}")
        
        # 测试generate方法
        response = client.generate("你好，请问你是谁？")
        print(f"✅ 测试generate成功: {response[:50]}...")
        
        # 测试chat_completion方法
        messages = [{"role": "user", "content": "请用一个词描述人工智能"}]
        response = client.chat_completion(messages)
        print(f"✅ 测试chat_completion成功: {response.get('content', '')[:50]}...")
        
        print("\n🎉 所有LLM客户端测试通过！")
    except Exception as e:
        print(f"❌ 测试失败: {e}")

if __name__ == "__main__":
    test_llm_client()
