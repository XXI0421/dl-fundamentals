# test_kimi_connection.py
from llm_client import KimiClient
import os

# $env:KIMI_API_KEY = "your_api_key_here"
def test_connection():
    client = KimiClient(
        api_key=os.getenv("KIMI_API_KEY"),
        model="moonshot-v1-32k"  # Day 2 推荐 32k，足够应对多轮 ReAct
    )
    
    # 简单对话测试
    resp = client.chat_completion([
        {"role": "system", "content": "你是一个乐于助人的助手"},
        {"role": "user", "content": "你好，请用一句话证明你在线"}
    ])
    
    print("Kimi 响应：", resp["content"])
    assert "error" not in resp, "连接失败"
    print("✅ Kimi API 连接正常")

if __name__ == "__main__":
    test_connection()
