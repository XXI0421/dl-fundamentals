import openai
from typing import List, Dict, Any, Optional

class KimiClient:
    def __init__(self, api_key: str, model: str = "moonshot-v1-32k"):
        """
        模型选项：
        - moonshot-v1-8k: 轻量任务
        - moonshot-v1-32k: 中等复杂任务（推荐）
        - moonshot-v1-128k: 长上下文，适合 ReAct 多轮历史
        """
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.moonshot.cn/v1"
        )
        self.model = model
    
    def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        temperature: float = 0.3  # ReAct 需要确定性，不宜过高
    ) -> Dict[str, Any]:
        """
        调用 Kimi API，支持 Function Calling
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice if tools else None,
                temperature=temperature
            )
            return self._parse_response(response)
        except Exception as e:
            return {"error": str(e)}
    
    def _parse_response(self, response) -> Dict[str, Any]:
        """统一输出格式，兼容 ReActAgent"""
        message = response.choices[0].message
        result = {
            "content": message.content,
            "tool_calls": None,
            "finish_reason": response.choices[0].finish_reason
        }
        
        # 提取 Function Calling 信息
        if hasattr(message, 'tool_calls') and message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments  # JSON 字符串
                }
                for tc in message.tool_calls
            ]
        
        return result
    
    def generate(self, prompt: str) -> str:
        """兼容 Day 1 的接口（非 Function Calling 模式）"""
        response = self.chat_completion(
            messages=[{"role": "user", "content": prompt}]
        )
        return response.get("content", "")
