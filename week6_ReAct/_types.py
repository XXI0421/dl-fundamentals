from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field

@dataclass
class Step:
    """ReAct 单步记录"""
    thought: str           # 思考内容
    action: str            # 行动指令，格式：ToolName[arguments]
    observation: str       # 观察结果
    step_num: int          # 第几步（从1开始）
    
    def to_string(self) -> str:
        """格式化为 ReAct 标准格式"""
        return (f"Thought {self.step_num}: {self.thought}\n"
                f"Action {self.step_num}: {self.action}\n"
                f"Observation {self.step_num}: {self.observation}")

@dataclass
class AgentState:
    """Agent 全局状态"""
    query: str                                    # 原始问题
    trajectory: List[Step] = field(default_factory=list)  # 执行轨迹
    final_answer: Optional[str] = None            # 最终答案
    iteration_count: int = 0                      # 当前迭代次数
