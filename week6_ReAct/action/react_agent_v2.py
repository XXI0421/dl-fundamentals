import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from llm_client import KimiClient
from memory.short_term import ConversationSummaryMemory

@dataclass
class Step:
    action: str
    action_input: Dict[str, Any]
    observation: str
    step_num: int

class ReActAgentV2:
    def __init__(
        self,
        llm_client: KimiClient,
        tool_registry,
        max_iterations: int = 5,
        memory_k: int = 3
    ):
        self.llm = llm_client
        self.tools = tool_registry
        self.max_iterations = max_iterations
        self.memory = ConversationSummaryMemory(k=memory_k)
        self.trajectory: List[Step] = []
        
    def run(self, query: str) -> str:
        self.memory.add_user(query)
        
        messages = self._build_messages()
        
        final_answer = ""
        tools_used = []
        last_tool_calls = None  # 记录最后一次工具调用
        
        for i in range(self.max_iterations):
            print(f"\n--- 第 {i+1} 轮 ---")
            
            response = self.llm.chat_completion(
                messages=messages,
                tools=self.tools.get_schemas(),
                tool_choice="auto",
                temperature=0.2
            )
            
            if "error" in response:
                final_answer = f"API错误: {response['error']}"
                break
            
            if response.get("finish_reason") == "stop" or not response.get("tool_calls"):
                final_answer = response.get("content", "")
                self.memory.add_assistant(content=final_answer)
                print(f"✅ 直接回答: {final_answer[:100]}...")
                break
            
            tool_calls = response["tool_calls"]
            tools_used = [tc["name"] for tc in tool_calls]
            last_tool_calls = tool_calls  # 记录当前工具调用
            print(f"🔧 调用: {tools_used}")
            
            assistant_msg = {
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
                    } for tc in tool_calls
                ]
            }
            messages.append(assistant_msg)
            self.memory.add_assistant(tool_calls=tool_calls)
            
            for tc in tool_calls:
                result = self._execute_tool(tc)
                
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": tc["name"],
                    "content": str(result)
                }
                messages.append(tool_msg)
                self.memory.add_tool_result(tc["id"], tc["name"], str(result))
                
                print(f"👁 {tc['name']}: {str(result)[:60]}...")
                
                self.trajectory.append(Step(
                    action=tc["name"],
                    action_input=json.loads(tc["arguments"]),
                    observation=str(result),
                    step_num=i+1
                ))
        
        else:
            # 达到最大迭代次数
            # 检查是否有未完成的工具调用（最后一次迭代执行了工具调用但未总结）
            if last_tool_calls:
                print("\n--- 最后总结 ---")
                # 强制进行一次总结调用（不调用工具）
                response = self.llm.chat_completion(
                    messages=messages,
                    tools=self.tools.get_schemas(),
                    tool_choice="none",  # 强制不调用工具
                    temperature=0.2
                )
                if response.get("content"):
                    final_answer = response["content"]
                    print(f"✅ 总结回答: {final_answer[:100]}...")
                else:
                    final_answer = "达到最大迭代次数，任务未完成"
            else:
                final_answer = "达到最大迭代次数，任务未完成"
            
            self.memory.add_assistant(content=final_answer)
        
        self.memory.add_conversation({
            "user": query,
            "agent_response": final_answer,
            "tools_used": tools_used
        })
        
        return final_answer
    
    def _build_messages(self) -> List[Dict[str, Any]]:
        """精简版上下文构建：只传关键事实，不传完整历史"""
        memory_facts = self.memory.get_summary()
        
        system_content = f"""你是一个严谨的计算器助手。

历史事实：{memory_facts if memory_facts else '无'}

规则：
1. 简单计算（加减乘除）必须心算，禁止调用工具
2. 优先使用历史事实中的数字，不要重复查询
3. 如果历史有年份/年龄，直接使用

回答要简洁。"""

        messages = [{"role": "system", "content": system_content}]
        
        recent_history = self.memory.get_messages()[-4:]
        if recent_history:
            messages.extend(recent_history)
            print(f"[上下文] {len(recent_history)} 条消息")
            if memory_facts:
                print(f"[记忆事实] {memory_facts}")
        
        return messages
    
    def _execute_tool(self, tool_call: Dict) -> str:
        name = tool_call["name"]
        try:
            args = json.loads(tool_call["arguments"])
            if name in self.tools._tools:
                tool_obj = self.tools._tools[name]
                result = tool_obj.func(**args) if hasattr(tool_obj, 'func') else tool_obj(**args)
                return str(result)[:1000]
            return f"错误: 未找到工具 {name}"
        except Exception as e:
            return f"执行错误: {str(e)}"
    
    def get_memory_debug(self) -> str:
        return self.memory.get_full_summary()