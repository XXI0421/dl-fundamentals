"""
LLM客户端 - 支持多种LLM提供商的统一接口
"""
import openai
from typing import List, Dict, Any, Optional, Union
from config import Config

class BaseLLMClient:
    """基础LLM客户端接口"""
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        temperature: float = 0.3,
        **kwargs
    ) -> Dict[str, Any]:
        raise NotImplementedError
    
    def generate(self, prompt: str, **kwargs) -> str:
        kwargs['tool_choice'] = "none"
        response = self.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        if response is None:
            return ""
        content = response.get("content", "")
        return content if content is not None else ""

class KimiClient(BaseLLMClient):
    """Kimi API客户端"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "moonshot-v1-128k"):
        self.api_key = api_key or Config.KIMI_API_KEY
        self.model = model or Config.KIMI_MODEL
        
        if not self.api_key:
            raise ValueError("Kimi API密钥未配置，请设置环境变量 KIMI_API_KEY")
        
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://api.moonshot.cn/v1"
        )
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        temperature: float = 0.3,
        max_tokens: int = 4096,
        **kwargs
    ) -> Dict[str, Any]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice if tools else None,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return self._parse_response(response)
        except Exception as e:
            return {"error": str(e), "content": None, "tool_calls": None, "finish_reason": "error"}
    
    def _parse_response(self, response) -> Dict[str, Any]:
        message = response.choices[0].message
        result = {
            "content": message.content,
            "tool_calls": None,
            "finish_reason": response.choices[0].finish_reason
        }
        
        if hasattr(message, 'tool_calls') and message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                }
                for tc in message.tool_calls
            ]
        
        return result

class OpenAIClient(BaseLLMClient):
    """OpenAI API客户端（兼容OpenAI格式）"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini"
    ):
        self.api_key = api_key or Config.OPENAI_API_KEY
        self.base_url = base_url or Config.OPENAI_BASE_URL
        self.model = model or Config.OPENAI_MODEL
        
        if not self.api_key:
            raise ValueError("OpenAI API密钥未配置，请设置环境变量 OPENAI_API_KEY")
        
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url if self.base_url else None
        )
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        temperature: float = 0.3,
        max_tokens: int = 4096,
        **kwargs
    ) -> Dict[str, Any]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice if tools else None,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return self._parse_response(response)
        except Exception as e:
            return {"error": str(e), "content": None, "tool_calls": None, "finish_reason": "error"}
    
    def _parse_response(self, response) -> Dict[str, Any]:
        message = response.choices[0].message
        result = {
            "content": message.content,
            "tool_calls": None,
            "finish_reason": response.choices[0].finish_reason
        }
        
        if hasattr(message, 'tool_calls') and message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                }
                for tc in message.tool_calls
            ]
        
        return result

def get_llm_client(provider: Optional[str] = None) -> BaseLLMClient:
    """
    获取LLM客户端实例
    
    Args:
        provider: "kimi", "openai", 或None（自动检测）
    
    Returns:
        LLM客户端实例
    
    Raises:
        ValueError: 如果未配置任何LLM API
    """
    if provider is None:
        provider = Config.get_default_llm_provider()
    
    if provider == "kimi":
        return KimiClient()
    elif provider == "openai":
        return OpenAIClient()
    else:
        raise ValueError(
            "未配置有效的LLM提供商。请设置以下环境变量之一:\n"
            "- KIMI_API_KEY (推荐)\n"
            "- OPENAI_API_KEY\n"
            "然后重新运行程序。"
        )
