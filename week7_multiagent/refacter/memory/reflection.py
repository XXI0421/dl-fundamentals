"""
反思引擎模块 - 实现Agent的自我反思能力
"""
from typing import List, Dict, Any, Optional
from llm_client import BaseLLMClient

class ReflectionEngine:
    """反思引擎 - 分析对话历史并生成改进建议"""
    
    def __init__(self, llm_client: BaseLLMClient):
        """
        Args:
            llm_client: LLM客户端
        """
        self.llm_client = llm_client
    
    def reflect_on_task(self, task: str, execution_log: List[Dict]) -> str:
        """
        对任务执行过程进行反思
        
        Args:
            task: 原始任务描述
            execution_log: 执行日志
        
        Returns:
            反思总结
        """
        log_text = "\n".join([
            f"步骤{i+1}: {step.get('action', '')} -> {step.get('result', '')}"
            for i, step in enumerate(execution_log)
        ])
        
        prompt = f"""请分析以下任务执行过程，进行深度反思：

任务：{task}

执行日志：
{log_text}

请从以下角度进行反思：
1. 执行过程中的优点
2. 执行过程中的不足
3. 可以改进的地方
4. 下次执行类似任务的建议

反思报告："""
        
        return self.llm_client.generate(prompt)
    
    def analyze_error(self, error: str, context: str) -> Dict[str, Any]:
        """
        分析错误并提供修复建议
        
        Args:
            error: 错误信息
            context: 上下文信息
        
        Returns:
            包含分析结果和建议的字典
        """
        prompt = f"""请分析以下错误并提供修复建议：

错误信息：{error}

上下文：{context}

请输出JSON格式，包含：
- analysis: 错误原因分析
- suggestion: 修复建议
- severity: 严重程度（high/medium/low）
"""
        
        response = self.llm_client.generate(prompt)
        
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            return json.loads(response)
        except:
            return {
                "analysis": "无法解析错误",
                "suggestion": "请检查代码逻辑",
                "severity": "high"
            }
    
    def summarize_conversation(self, messages: List[Dict]) -> str:
        """
        总结对话内容
        
        Args:
            messages: 消息列表
        
        Returns:
            对话摘要
        """
        conversation = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in messages
        ])
        
        prompt = f"""请总结以下对话：

{conversation}

总结："""
        
        return self.llm_client.generate(prompt)
