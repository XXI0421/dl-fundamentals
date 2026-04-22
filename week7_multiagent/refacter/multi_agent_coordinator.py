"""
多Agent协调器 - 管理多个Agent协作执行任务
支持分步执行和用户验收
"""
import json
import time
from typing import List, Dict, Any, Optional
from llm_client import BaseLLMClient, get_llm_client
from message_bus import MessageBus
from react_agent import ReActAgent
from tools.base import ToolRegistry
from tools.real_tools import init_default_tools
from tools.python_sandbox import init_sandbox_tools

class AgentInfo:
    """Agent信息"""
    def __init__(self, id: str, prompt: str):
        self.id = id
        self.prompt = prompt
        self.agent: Optional[ReActAgent] = None
        self.output: str = ""
        self.accepted: bool = False

class MultiAgentCoordinator:
    """
    多Agent协调器
    
    工作流程：
    1. 用户输入需求
    2. 主Agent分析需求，生成子任务和Agent配置
    3. 依次执行每个子Agent
    4. 每个Agent完成后等待用户验收
    5. 验收通过继续下一个Agent，否则回溯修改
    """
    
    def __init__(self, llm_client: Optional[BaseLLMClient] = None):
        self.llm_client = llm_client or get_llm_client()
        self.bus = MessageBus()
        self.agents: List[AgentInfo] = []
        self.shared_context: Dict[str, Any] = {}
        self.execution_log: List[Dict] = []
        self.current_agent_index: int = 0
        self.analysis_result: Optional[Dict] = None
    
    def add_agent(self, agent_id: str, prompt: str):
        """添加Agent"""
        self.agents.append(AgentInfo(agent_id, prompt))
        self.bus.register(agent_id)
    
    def clear_agents(self):
        """清空所有Agent"""
        self.agents = []
        self.current_agent_index = 0
        self.execution_log = []
        self.shared_context = {}
    
    def _get_default_agents(self, requirement: str = "") -> Dict:
        """获取默认的Agent配置（当LLM解析失败时使用）"""
        return {
            "task_name": "需求实现任务",
            "task_description": requirement,
            "agents": [
                {
                    "id": "分析员",
                    "prompt": "你是一位需求分析专家，负责分析用户需求并输出详细的分析报告。",
                    "task": f"分析用户需求: {requirement}",
                    "output": "analysis_result"
                },
                {
                    "id": "执行员",
                    "prompt": "你是一位执行专家，负责根据分析结果完成具体任务。",
                    "task": "根据分析结果执行任务",
                    "output": "final_output"
                }
            ]
        }
    
    def analyze_requirement_and_generate_agents(self, requirement: str) -> Dict:
        """
        分析用户需求并自动生成Agent配置
        
        Returns:
            包含任务名称、步骤和生成的Agent配置的字典
        """
        prompt = f"""
请分析以下用户需求，生成详细的任务分解和对应的Agent配置：

用户需求：{requirement}

请输出JSON格式的分析结果，包含：
1. task_name: 任务名称（简短描述）
2. task_description: 任务详细描述
3. agents: Agent配置列表，每个Agent包含：
   - id: Agent标识（如"产品经理"）
   - prompt: Agent的角色提示词
   - task: 该Agent需要完成的任务
   - output: 产出物名称

示例输出：
{{
    "task_name": "需求实现任务",
    "task_description": "{requirement}",
    "agents": [
        {{
            "id": "产品经理",
            "prompt": "你是一位经验丰富的产品经理，擅长分析用户需求并撰写PRD文档。",
            "task": "分析用户需求，撰写详细的PRD文档",
            "output": "prd_doc"
        }},
        {{
            "id": "系统架构师",
            "prompt": "你是一位资深系统架构师，擅长设计技术方案。",
            "task": "基于PRD设计技术架构方案",
            "output": "tech_spec"
        }}
    ]
}}

请确保JSON格式正确，不要包含其他文字。
"""
        
        response = self.llm_client.generate(prompt)
        
        try:
            # 检查响应是否为空
            if not response or response.strip() == "":
                print("⚠️ LLM返回空响应，使用默认配置")
                result = self._get_default_agents(requirement)
            else:
                if "```json" in response:
                    response = response.split("```json")[1].split("```")[0].strip()
                
                # 检查是否还有内容
                if not response or response.strip() == "":
                    print("⚠️ JSON提取后为空，使用默认配置")
                    result = self._get_default_agents(requirement)
                else:
                    result = json.loads(response)
            
            # 验证格式
            if "agents" in result and isinstance(result["agents"], list):
                self.analysis_result = result
                
                # 清空并添加生成的Agent
                self.clear_agents()
                for agent_config in result["agents"]:
                    self.add_agent(
                        agent_config.get("id", f"Agent {len(self.agents) + 1}"),
                        agent_config.get("prompt", "")
                    )
                
            return result
        except Exception as e:
            print(f"LLM解析失败，使用默认配置: {e}")
            # 使用默认配置
            default_result = self._get_default_agents(requirement)
            self.analysis_result = default_result
            self.clear_agents()
            for agent_config in default_result["agents"]:
                self.add_agent(agent_config["id"], agent_config["prompt"])
            return default_result
    
    def execute_agent_step(self, requirement: str, step_index: int = 0) -> Dict:
        """
        执行单个Agent步骤
        
        Args:
            requirement: 用户需求
            step_index: 要执行的Agent索引
        
        Returns:
            执行结果
        """
        if step_index >= len(self.agents):
            return {
                "success": False,
                "error": "没有更多Agent需要执行"
            }
        
        self.current_agent_index = step_index
        agent_info = self.agents[step_index]
        
        # 初始化Agent
        if agent_info.agent is None:
            registry = ToolRegistry()
            init_default_tools(registry)
            init_sandbox_tools(registry)
            
            agent_info.agent = ReActAgent(
                llm_client=self.llm_client,
                tools=registry
            )
        
        # 构建任务输入
        task_input = agent_info.prompt + "\n\n任务："
        
        # 如果不是第一个Agent，添加前一个Agent的产出
        if step_index > 0:
            prev_output = self.agents[step_index - 1].output
            if prev_output:
                task_input += f"参考上一步产出：\n{prev_output}\n\n"
        
        # 添加用户需求
        task_input += requirement
        
        # 执行Agent
        result = agent_info.agent.run(task_input)
        agent_info.output = result
        
        # 保存到共享上下文
        if self.analysis_result and step_index < len(self.analysis_result.get("agents", [])):
            output_key = self.analysis_result["agents"][step_index].get("output", f"output_{step_index}")
            self.shared_context[output_key] = result
        
        self.execution_log.append({
            "step": step_index,
            "agent": agent_info.id,
            "output": result[:200] + "..." if len(result) > 200 else result,
            "timestamp": time.time()
        })
        
        # 通过消息总线通知
        self.bus.broadcast(agent_info.id, f"Agent {agent_info.id} 完成任务")
        
        return {
            "success": True,
            "agent_id": agent_info.id,
            "step": step_index,
            "total_steps": len(self.agents),
            "output": result,
            "shared_context": self.shared_context
        }
    
    def get_agent_count(self) -> int:
        """获取Agent数量"""
        return len(self.agents)
    
    def get_current_agent(self) -> Optional[AgentInfo]:
        """获取当前Agent"""
        if self.current_agent_index < len(self.agents):
            return self.agents[self.current_agent_index]
        return None
    
    def set_agent_accepted(self, step_index: int, accepted: bool):
        """设置Agent产出是否被验收"""
        if step_index < len(self.agents):
            self.agents[step_index].accepted = accepted
    
    def reset_to_step(self, step_index: int):
        """回溯到指定步骤"""
        # 重置该步骤及之后的所有Agent
        for i in range(step_index, len(self.agents)):
            self.agents[i].output = ""
            self.agents[i].accepted = False
            self.agents[i].agent = None
        
        self.current_agent_index = step_index
        
        # 清除相关的执行日志
        self.execution_log = self.execution_log[:step_index]
    
    def get_execution_log(self) -> List[Dict]:
        """获取执行日志"""
        return self.execution_log
