import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from llm_client import KimiClient

@dataclass
class Step:
    thought: Optional[str]    # Function Calling 模式下可能为 None（由 LLM 内部处理）
    action: str               # 工具名
    action_input: Dict[str, Any]  # 结构化参数
    observation: str
    step_num: int

class ReActAgentV2:
    def __init__(
        self,
        llm_client: KimiClient,
        tool_registry,           # Day 2 使用注册器
        max_iterations: int = 10
    ):
        self.llm = llm_client
        self.tools = tool_registry
        self.max_iterations = max_iterations
        self.trajectory: List[Step] = []
        
    def run(self, query: str) -> str:
        """基于 Function Calling 的 ReAct 循环"""
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": query}
        ]
        
        for i in range(self.max_iterations):
            print(f"\n--- 第 {i+1} 轮 ---")
            
            # 1. 调用 LLM（传入工具定义）
            response = self.llm.chat_completion(
                messages=messages,
                tools=self.tools.get_schemas(),  # 获取所有工具的 JSON Schema
                tool_choice="auto"
            )
            
            if "error" in response:
                return f"API 错误：{response['error']}"
            
            # 2. 检查是否需要调用工具
            if response["finish_reason"] == "tool_calls" and response["tool_calls"]:
                # 处理工具调用（支持并行）
                observations = self._execute_tools(response["tool_calls"])
                
                # 记录轨迹
                for j, tc in enumerate(response["tool_calls"]):
                    step = Step(
                        thought=None,  # Kimi 在 function calling 模式下不输出 Thought 文本
                        action=tc["name"],
                        action_input=json.loads(tc["arguments"]),
                        observation=observations[j],
                        step_num=i+1
                    )
                    self.trajectory.append(step)
                    print(f"🔧 工具：{tc['name']}, 参数：{tc['arguments']}")
                    print(f"👁 结果：{observations[j][:100]}...")
                
                # 将工具结果加入上下文（OpenAI 格式）
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": tc["arguments"]
                            }
                        } for tc in response["tool_calls"]
                    ]
                })
                
                # 添加工具返回结果
                for j, tc in enumerate(response["tool_calls"]):
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "name": tc["name"],
                        "content": observations[j]
                    })
            
            else:
                # 直接输出答案
                answer = response["content"]
                print(f"✅ 最终答案：{answer}")
                return answer
        
        return "达到最大迭代次数"
    
    def _execute_tools(self, tool_calls: List[Dict]) -> List[str]:
        """并行执行多个工具调用"""
        results = []
        for tc in tool_calls:
            tool_name = tc["name"]
            arguments = json.loads(tc["arguments"])
            
            if tool_name not in self.tools._tools:
                results.append(f"错误：未找到工具 {tool_name}")
                continue
            
            try:
                tool_obj = self.tools._tools[tool_name]
                result = tool_obj.func(**arguments)  # 使用 ** 解包字典参数
                results.append(str(result))
            except Exception as e:
                results.append(f"执行错误：{str(e)}")
        
        return results
    
    def _build_system_prompt(self) -> str:
        """系统提示：约束 ReAct 行为"""
        return """你是一个智能 Agent，通过调用工具解决问题。
可用工具已提供给你。请遵循以下规则：
1. 分析用户需求，选择合适的工具调用
2. 如果一次需要多个信息，可以同时调用多个工具（并行）
3. 当获得足够信息，直接回答用户，不再调用工具
4. 如果工具返回错误，分析是否换用其他工具或调整参数"""
