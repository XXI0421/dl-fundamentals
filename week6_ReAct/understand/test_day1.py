from react_agent import ReActAgent
from react_prompt import REACT_PROMPT_TEMPLATE
from tools.mock_tools import mock_search, mock_calculator, mock_lookup

# 模拟 LLM 客户端（实际使用时替换为 OpenAI/Claude API）
class MockLLM:
    """模拟 LLM 用于测试 ReAct 循环逻辑（非真实调用，用于验证解析逻辑）"""
    def __init__(self, preset_responses=None):
        self.preset = preset_responses or []
        self.call_count = 0
    
    def generate(self, prompt: str) -> str:
        """根据 prompt 中的历史决定下一步（简化版逻辑）"""
        # 实际测试时，这里应该调用真实 LLM
        # 这个 mock 仅用于验证代码流程跑通
        if self.call_count < len(self.preset):
            resp = self.preset[self.call_count]
            self.call_count += 1
            return resp
        return "Thought 1: 测试结束\nAction 1: Finish[测试完成]"

# 测试用例 1：多步推理（Search → Calculation）
def test_multi_step():
    print("=" * 50)
    print("测试 1：多步推理（搜索 + 计算）")
    print("=" * 50)
    
    # 预设 LLM 回复序列（模拟真实 LLM 的 ReAct 思考过程）
    preset_responses = [
        # 第 1 轮：搜索 Python 创始人出生年份
        "Thought 1: 我需要先找到 Python 创始人 Guido 的出生年份。\n"
        "Action 1: Search[Guido van Rossum birth year]",
        
        # 第 2 轮：计算年龄（假设当前 2026 年）
        "Thought 2: Guido 出生于 1956 年，现在 2026 年，需要计算年龄。\n"
        "Action 2: Calculator[2026 - 1956]",
        
        # 第 3 轮：结束
        "Thought 3: 已获得计算结果 70，可以给出最终答案。\n"
        "Action 3: Finish[Guido van Rossum 今年 70 岁]"
    ]
    
    llm = MockLLM(preset_responses)
    tools = {
        "Search": mock_search,
        "Calculator": mock_calculator
    }
    
    agent = ReActAgent(llm, tools, REACT_PROMPT_TEMPLATE)
    result = agent.run("Guido van Rossum 今年多少岁？")
    
    print(f"\n最终结果：{result}")
    assert result == "Guido van Rossum 今年 70 岁", f"结果不符：{result}"
    print("✅ 测试通过：完成 3 轮 ReAct 循环")

# 测试用例 2：错误恢复（工具不存在）
def test_error_handling():
    print("\n" + "=" * 50)
    print("测试 2：错误处理（调用不存在的工具）")
    print("=" * 50)
    
    preset = [
        "Thought 1: 我试试不存在的工具。\n"
        "Action 1: UnknownTool[test]",

        "Thought 2: 工具调用失败，我应该换用可用工具 Search。\n"
        "Action 2: Search[test]",

        "Thought 3: 已获得信息，结束。\n"
        "Action 3: Finish[处理完成]"
    ]
    
    llm = MockLLM(preset)
    tools = {"Search": mock_search}
    agent = ReActAgent(llm, tools, REACT_PROMPT_TEMPLATE)
    result = agent.run("测试错误处理")
    
    print(f"结果：{result}")
    assert "处理完成" in result
    print("✅ 测试通过：Agent 能从工具错误中恢复")

if __name__ == "__main__":
    test_multi_step()
    test_error_handling()
    print("\n🎉 Day 1 基础测试全部通过！")
