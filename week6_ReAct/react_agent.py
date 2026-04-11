import re
from typing import Dict, Callable, List, Tuple
from _types import Step, AgentState

class ReActAgent:
    def __init__(
        self, 
        llm_client,                                    # LLM API 客户端
        tools: Dict[str, Callable],                    # 工具名 → 函数映射
        prompt_template: str,                          # ReAct prompt 模板
        max_iterations: int = 10
    ):
        self.llm = llm_client
        self.tools = tools
        self.prompt_template = prompt_template
        self.max_iterations = max_iterations
        
        # 编译正则：提取 Thought 和 Action（支持多行）
        self.thought_pattern = re.compile(r"Thought\s*\d*\s*:\s*(.*?)(?=\nAction|\Z)", re.DOTALL)
        self.action_pattern = re.compile(r"Action\s*\d*\s*:\s*(.*?)(?=\nObservation|\Z)", re.DOTALL)
    
    def run(self, query: str) -> str:
        """主入口：执行 ReAct 循环"""
        state = AgentState(query=query)
        print(f"🚀 开始处理任务：{query}\n")
        
        for i in range(self.max_iterations):
            state.iteration_count = i + 1
            print(f"--- 第 {i+1} 轮 ---")
            
            # 1. 构造当前 Prompt（包含历史轨迹）
            prompt = self._construct_prompt(state)
            
            # 2. 调用 LLM 生成 Thought + Action
            try:
                llm_output = self.llm.generate(prompt)
                print(f"📝 LLM 输出：\n{llm_output}\n")
            except Exception as e:
                return f"LLM 调用失败：{str(e)}"
            
            # 3. 解析 Thought 和 Action（容错处理）
            thought, action = self._parse_response(llm_output, state.iteration_count)
            if not thought or not action:
                return "解析失败：LLM 未遵循 ReAct 格式"
            
            print(f"💭 Thought：{thought}")
            print(f"🔧 Action：{action}")
            
            # 4. 检查是否结束（Finish[answer]）
            if action.startswith("Finish["):
                answer = self._extract_finish_answer(action)
                state.final_answer = answer
                print(f"✅ 任务完成：{answer}")
                return answer
            
            # 5. 执行工具获取 Observation
            observation = self._execute_action(action)
            print(f"👁 Observation：{observation[:100]}...\n")  # 截断显示
            
            # 6. 记录到轨迹
            step = Step(
                thought=thought.strip(),
                action=action.strip(),
                observation=str(observation),
                step_num=state.iteration_count
            )
            state.trajectory.append(step)
        
        return "达到最大迭代次数，任务未完成"
    
    def _construct_prompt(self, state: AgentState) -> str:
        """构建完整 Prompt：模板 + 工具描述 + 历史轨迹 + 当前问题"""
        # 工具描述生成
        tool_desc = self._format_tool_descriptions()
        
        # 历史轨迹格式化
        trajectory_str = "\n".join([step.to_string() for step in state.trajectory])
        
        # 下一步序号
        next_step = state.iteration_count + 1
        
        # 填充模板
        prompt = self.prompt_template.format(
            tool_descriptions=tool_desc,
            query=state.query,
            trajectory=trajectory_str,
            step_num=next_step
        )
        
        return prompt
    
    def _format_tool_descriptions(self) -> str:
        """生成工具说明文档"""
        descs = []
        for name, func in self.tools.items():
            doc = func.__doc__ or "无描述"
            descs.append(f"- {name}[input]: {doc}")
        return "\n".join(descs)
    
    def _parse_response(self, response: str, step_num: int) -> Tuple[str, str]:
        """从 LLM 输出中提取 Thought 和 Action"""
        # 尝试匹配 "Thought N:" 和 "Action N:" 格式
        thought_match = re.search(rf"Thought\s*{step_num}\s*:\s*(.*?)(?=\nAction|\Z)", response, re.DOTALL)
        action_match = re.search(rf"Action\s*{step_num}\s*:\s*(.*?)(?=\nObservation|\Z|$)", response, re.DOTALL)
        
        # 容错：如果不带序号，匹配任意 Thought/Action
        if not thought_match:
            thought_match = self.thought_pattern.search(response)
        if not action_match:
            action_match = self.action_pattern.search(response)
        
        thought = thought_match.group(1).strip() if thought_match else ""
        action = action_match.group(1).strip() if action_match else ""
        
        return thought, action
    
    def _execute_action(self, action: str) -> str:
        """解析并执行工具调用"""
        # 解析 Action 格式：ToolName[arguments]
        match = re.match(r"(\w+)\[(.*?)\]", action)
        if not match:
            return f"错误：Action 格式不正确，应为 ToolName[args]，实际为：{action}"
        
        tool_name, args_str = match.groups()
        
        # 检查工具是否存在
        if tool_name not in self.tools:
            return f"错误：未知工具 '{tool_name}'，可用工具：{list(self.tools.keys())}"
        
        try:
            # 执行工具（简单字符串参数，复杂参数可改用 JSON）
            tool_func = self.tools[tool_name]
            result = tool_func(args_str)
            return str(result)
        except Exception as e:
            return f"工具执行错误：{str(e)}"
    
    def _extract_finish_answer(self, action: str) -> str:
        """从 Finish[answer] 中提取答案"""
        match = re.match(r"Finish\[(.*?)\]", action, re.DOTALL)
        return match.group(1).strip() if match else action
