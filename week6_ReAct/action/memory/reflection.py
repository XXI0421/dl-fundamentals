import json
import re
from typing import List, Dict, Optional
from dataclasses import dataclass
from llm_client import KimiClient
from memory.long_term import LongTermMemory, get_long_term_memory

@dataclass
class ReflectionResult:
    """反思结果"""
    facts: List[Dict]  # 提取的事实列表
    summary: str       # 本轮对话总结
    should_save: bool  # 是否值得保存

class ReflectionEngine:
    """记忆反思引擎：从对话中提取永久记忆"""
    
    def __init__(self, llm_client: KimiClient, session_id: Optional[str] = None):
        self.llm = llm_client
        self.session_id = session_id
        self.ltm = get_long_term_memory(session_id=session_id)
    
    def reflect(self, 
                conversation_history: List[Dict], 
                current_facts: List[str]) -> ReflectionResult:
        """
        执行反思：分析对话，提取值得长期保存的事实
        
        Args:
            conversation_history: 本轮完整对话历史
            current_facts: 当前短期记忆中的事实字符串
        """
        # 构建反思 Prompt
        dialog_text = self._format_dialog(conversation_history)
        
        # 从对话中提取用户姓名
        user_name = self._extract_user_name(conversation_history, current_facts)
        
        prompt = f"""分析以下对话，提取应该长期保存的关键事实。

用户姓名: {user_name if user_name else '未知'}

已知当前事实: {', '.join(current_facts) if current_facts else '无'}

对话历史:
{dialog_text}

任务：
1. 识别用户提到的永久信息（姓名、生日、年龄、职业、偏好、重要结论等）
2. 排除临时计算结果（如"2026-30=1996"这种计算过程，只保留"出生于1996"）
3. 如果知道用户姓名，所有事实都要使用真实姓名（如"张三出生于1996年"，不要用"用户"）
4. 为每个事实标注类别：
   - identity: 身份信息（姓名、生日、年龄等）
   - fact: 客观事实（职业、工作单位等）
   - preference: 偏好爱好
   - other: 其他

输出格式（JSON）：
{{
    "summary": "一句话总结对话",
    "facts": [
        {{"text": "张三出生于1996年", "category": "identity", "importance": 0.9}},
        {{"text": "张三是Python开发者", "category": "fact", "importance": 0.8}},
        {{"text": "张三喜欢编程", "category": "preference", "importance": 0.8}}
    ],
    "should_save": true
}}

如果对话无重要信息（闲聊、临时计算），返回 should_save: false"""

        try:
            response = self.llm.generate(prompt)
            result = json.loads(response)
            
            # 验证并存储到长期记忆
            saved_facts = []
            for fact in result.get("facts", []):
                if fact.get("text") and len(fact["text"]) > 5:
                    fid = self.ltm.add_fact(
                        text=fact["text"],
                        category=fact.get("category", "general"),
                        importance=fact.get("importance", 0.5)
                    )
                    saved_facts.append({"id": fid, **fact})
            
            return ReflectionResult(
                facts=saved_facts,
                summary=result.get("summary", ""),
                should_save=result.get("should_save", False)
            )
            
        except Exception as e:
            print(f"[Reflection] 反思失败: {e}")
            return ReflectionResult(facts=[], summary="", should_save=False)
    
    def _extract_user_name(self, history: List[Dict], facts: List[str]) -> str:
        """从对话历史和事实中提取用户姓名"""
        # 先从对话历史中查找
        for msg in history:
            content = msg.get("content", "")
            if "我叫" in content:
                # 提取 "我叫XXX" 中的名字
                idx = content.find("我叫") + 2
                name = ""
                for char in content[idx:]:
                    if char in "，。！？、 ":
                        break
                    name += char
                if name:
                    return name
        
        # 从当前事实中查找
        for fact in facts:
            if "叫" in fact or "姓名" in fact:
                # 提取名字部分
                match = re.search(r'(?:叫|姓名)[：:]?\s*([\u4e00-\u9fa5]{2,4})', fact)
                if match:
                    return match.group(1)
        
        # 从长期记忆中查找
        profile = self.ltm.get_user_profile()
        for fact in profile["fact"]:
            match = re.search(r'^([\u4e00-\u9fa5]{2,4})\s', fact)
            if match:
                return match.group(1)
        
        return ""
    
    def _format_dialog(self, history: List[Dict]) -> str:
        """格式化对话为文本"""
        lines = []
        for msg in history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                lines.append(f"User: {content}")
            elif role == "assistant" and content:
                lines.append(f"Agent: {content[:100]}...")
        return "\n".join(lines[-10:])  # 只取最近10条防止过长
    
    def periodic_reflect(self, agent_memory_summary: str) -> List[str]:
        """
        定期反思：基于记忆摘要主动询问用户以完善画像
        （可选高级功能）
        """
        # 如果长期记忆很少，主动询问关键信息
        profile = self.ltm.get_user_profile()
        missing = []
        
        if not any("生日" in f or "出生" in f for f in profile["fact"]):
            missing.append("我还不知道您的生日或年龄")
        if not any("职业" in f or "工作" in f for f in profile["preference"]):
            missing.append("我还不知道您的职业")
        
        return missing
