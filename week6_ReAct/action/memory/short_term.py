from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import deque
import re

@dataclass
class Message:
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

class ConversationBufferWindowMemory:
    def __init__(self, k: int = 3, content_limit: int = 500, tool_result_limit: int = 300):
        self.k = k
        self.content_limit = content_limit
        self.tool_result_limit = tool_result_limit
        self.messages: deque = deque(maxlen=k * 2)
    
    def add_user(self, content: str):
        self.messages.append(Message(role="user", content=content[:self.content_limit]))
    
    def add_assistant(self, content: Optional[str] = None, tool_calls: Optional[List] = None):
        self.messages.append(Message(role="assistant", content=content[:self.content_limit] if content else None, tool_calls=tool_calls))
    
    def add_tool_result(self, tool_call_id: str, name: str, content: str):
        self.messages.append(Message(role="tool", tool_call_id=tool_call_id, name=name, content=content[:self.tool_result_limit]))
    
    def get_messages(self) -> List[Dict[str, Any]]:
        result = []
        for msg in self.messages:
            entry = {"role": msg.role}
            if msg.content: entry["content"] = msg.content
            if msg.tool_calls: entry["tool_calls"] = msg.tool_calls
            if msg.tool_call_id:
                entry["tool_call_id"] = msg.tool_call_id
                entry["name"] = msg.name
            result.append(entry)
        return result
    
    def clear(self):
        """清空所有消息"""
        self.messages.clear()

class ConversationSummaryMemory(ConversationBufferWindowMemory):
    def __init__(self, k: int = 3, summarizer_llm=None, content_limit: int = 500, tool_result_limit: int = 300):
        super().__init__(k=k, content_limit=content_limit, tool_result_limit=tool_result_limit)
        self.summarizer = summarizer_llm
        self.summary: str = ""
        self.conversations: List[Dict] = []
        self.current_conversation: Dict = {}
    
    def add_user(self, content: str):
        super().add_user(content)
        self.current_conversation = {"user": content[:self.content_limit], "agent_response": "", "tools_used": []}
    
    def add_assistant(self, content: Optional[str] = None, tool_calls: Optional[List] = None):
        super().add_assistant(content=content, tool_calls=tool_calls)
        if tool_calls:
            self.current_conversation["tools_used"] = [tc["name"] for tc in tool_calls]
    
    def add_tool_result(self, tool_call_id: str, name: str, content: str):
        super().add_tool_result(tool_call_id, name, content)
        if "tool_results" not in self.current_conversation:
            self.current_conversation["tool_results"] = []
        self.current_conversation["tool_results"].append({name: content[:self.tool_result_limit]})
    
    def add_conversation(self, conversation: Dict[str, str]):
        self.conversations.append(conversation)
        if len(self.conversations) > self.k:
            self.conversations = self.conversations[-self.k:]
    
    def get_summary(self) -> str:
        """精简版摘要：只保留关键事实（数字、日期）"""
        if not self.conversations:
            return ""
        
        key_facts = set()
        
        for conv in self.conversations[-self.k:]:
            text = conv.get('user', '') + " " + conv.get('agent_response', '')
            facts = self._extract_facts(text)
            key_facts.update(facts)
        
        if key_facts:
            return "；".join(sorted(key_facts))
        return ""
    
    def _extract_facts(self, text: str) -> List[str]:
        """提取关键事实（只保留有价值的数字信息）"""
        facts = []
        
        years = re.findall(r'(20\d{2}|19\d{2})\s*年?', text)
        for y in years: facts.append(f"{y}年")
        
        ages = re.findall(r'(\d+)\s*岁', text)
        for a in ages: facts.append(f"{a}岁")
        
        numbers = re.findall(r'(\d+)\s*(万|千|百|元|人|个|次)', text)
        for n, u in numbers: facts.append(f"{n}{u}")
        
        return facts
    
    def get_messages(self) -> List[Dict[str, Any]]:
        return super().get_messages()
    
    def get_full_summary(self) -> str:
        """完整版摘要（包含对话概览）- 用于调试"""
        if not self.conversations:
            return ""
        
        key_facts = []
        summaries = []
        
        for i, conv in enumerate(self.conversations[-self.k:], 1):
            user = conv.get('user', '').strip()
            agent = conv.get('agent_response', '').strip()
            
            facts = self._extract_facts(user + " " + agent)
            key_facts.extend(facts)
            
            user_short = user[:30] + '...' if len(user) > 30 else user
            agent_short = agent[:40] + '...' if len(agent) > 40 else agent
            
            summaries.append(f"【对话{i}】{user_short} | {agent_short}")
        
        unique_facts = list(dict.fromkeys(key_facts))
        
        if unique_facts:
            return "【关键事实】" + "；".join(unique_facts[:5]) + "\n" + "\n".join(summaries)
        return "\n".join(summaries)
    
    def try_summarize(self):
        if not self.summarizer or len(self.conversations) < 2:
            return
        text = "\n".join([f"Q: {c['user']}\nA: {c.get('agent_response', '')}" for c in self.conversations])
        prompt = f"提取关键数字事实（年份/年龄/数值）：\n{text}\n事实："
        try:
            self.summary = self.summarizer.generate(prompt)
        except:
            pass
    
    def clear(self):
        super().clear()
        self.conversations.clear()
        self.current_conversation = {}
        self.summary = ""