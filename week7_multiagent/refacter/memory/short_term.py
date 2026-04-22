"""
短期记忆模块 - 管理对话历史和上下文
"""
from typing import List, Dict, Any, Optional
import time
from collections import deque

class ShortTermMemory:
    """短期记忆 - 存储最近的对话历史"""
    
    def __init__(self, max_size: int = 10):
        """
        Args:
            max_size: 最大存储的对话轮数
        """
        self.max_size = max_size
        self.messages: deque = deque(maxlen=max_size)
    
    def add_message(self, role: str, content: str):
        """添加消息到短期记忆"""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })
    
    def get_messages(self) -> List[Dict[str, Any]]:
        """获取所有消息"""
        return list(self.messages)
    
    def get_context(self, k: Optional[int] = None) -> str:
        """
        获取上下文字符串
        
        Args:
            k: 返回最近k条消息，None表示全部
        
        Returns:
            格式化的上下文字符串
        """
        messages = list(self.messages)
        if k is not None:
            messages = messages[-k:]
        
        context_parts = []
        for msg in messages:
            context_parts.append(f"{msg['role']}: {msg['content']}")
        
        return "\n".join(context_parts)
    
    def clear(self):
        """清空短期记忆"""
        self.messages.clear()
    
    def get_length(self) -> int:
        """获取消息数量"""
        return len(self.messages)

class ConversationSummaryMemory:
    """带摘要功能的对话记忆"""
    
    def __init__(self, llm_client=None, max_size: int = 10, summary_threshold: int = 5):
        """
        Args:
            llm_client: LLM客户端，用于生成摘要
            max_size: 最大存储的对话轮数
            summary_threshold: 超过此阈值时生成摘要
        """
        self.short_term = ShortTermMemory(max_size=max_size)
        self.llm_client = llm_client
        self.summary_threshold = summary_threshold
        self.summary: str = ""
    
    def add_user(self, content: str):
        """添加用户消息"""
        self.short_term.add_message("user", content)
        self._check_summary()
    
    def add_assistant(self, content: str):
        """添加助手消息"""
        self.short_term.add_message("assistant", content)
        self._check_summary()
    
    def add_system(self, content: str):
        """添加系统消息"""
        self.short_term.add_message("system", content)
    
    def _check_summary(self):
        """检查是否需要生成摘要"""
        if self.short_term.get_length() >= self.summary_threshold and self.llm_client:
            self._generate_summary()
    
    def _generate_summary(self):
        """生成对话摘要"""
        context = self.short_term.get_context()
        prompt = f"""请总结以下对话内容，保持关键信息：

{context}

总结："""
        
        self.summary = self.llm_client.generate(prompt)
        # 清空短期记忆，保留摘要
        self.short_term.clear()
    
    def get_full_context(self) -> str:
        """获取完整上下文（摘要+近期消息）"""
        parts = []
        if self.summary:
            parts.append(f"【对话摘要】{self.summary}")
        if self.short_term.get_length() > 0:
            parts.append(self.short_term.get_context())
        return "\n\n".join(parts)
    
    def clear(self):
        """清空所有记忆"""
        self.short_term.clear()
        self.summary = ""
