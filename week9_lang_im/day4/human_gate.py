"""
Day 4 Step 3/3: 人工干预嫁接
- aggregator 后插入 human_gate（interrupt + Command(resume)）
"""

from typing import TypedDict, Annotated, List
import operator
import os
import json
import re
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.types import Send, interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_classic.schema import SystemMessage, HumanMessage

from tools import get_current_time, python_executor


# ========== 0. 复用 Step 2 基础设施 ==========
def reviews_reducer(old: List[str], new: List[str]) -> List[str]:
    if not new:
        return []
    return old + new


def load_tools_constraints():
    return """【可用工具】
- python_executor: 在受限沙箱中执行 Python 代码
- get_current_time: 返回当前时间

【允许的 Python 模块】
os, csv, io, re, json, math, time, random, urllib, pathlib, collections, datetime, typing, hashlib, base64, string, itertools, functools, statistics, html, xml, requests, pandas, numpy, matplotlib.pyplot

【预注入全局变量】
json, math, re, csv, io, os, time, random, plt, requests, pd, np
"""


class AgentState(TypedDict):
    messages: Annotated[List[dict], operator.add]
    available_tools: str
    prd: str
    code: str
    reviews: Annotated[List[str], reviews_reducer]
    aggregated_review: str
    task_dir: str
    loop_count: int
    human_approved: bool
    human_decision: str  # 新增：存储人类的原始决策文本


class LLMRegistry:
    def __init__(self, base_url: str, api_key: str, model: str = "mimo-v2.5"):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model

    def _create(self, temperature: float) -> ChatOpenAI:
        return ChatOpenAI(
            model_name=self.model,
            openai_api_base=self.base_url,
            openai_api_key=self.api_key,
            temperature=temperature,
        )

    @property
    def pm(self):
        return (self._create(0.7), "你是资深产品经理。将模糊需求转化为结构化PRD。")

    @property
    def engineer(self):
        return (self._create(0.2), "你是资深Python工程师。只输出高质量、带注释、可直接运行的代码。禁止输出解释性文字。")

    @property
    def tester_security(self):
        return (self._create(0.3), "你是安全审计专家。检查命令注入、路径遍历、敏感信息硬编码。标记风险等级（高/中/低）。用中文输出。")

    @property
    def tester_performance(self):
        return (self._create(0.3), "你是性能优化专家。检查时间/空间复杂度、I/O效率。给出量化建议。用中文输出。")

    @property
    def tester_logic(self):
        return (self._create(0.3), "你是逻辑审查专家。检查边界条件、异常分支、与PRD一致性。用中文输出。")

    @property
    def aggregator(self):
        return (self._create(0.1), "")

    @property
    def executor(self):
        return (self._create(0.2), "")


# ========== 1. 复用 Step 2 节点工厂（精简版） ==========
def make_pm_node(llm_system: tuple):
    llm, system_prompt = llm_system
    tools_constraints = load_tools_constraints()
    def pm_node(state: AgentState):
        user_msg = state["messages"][-1].get("content", "") if state["messages"] else "写一个爬虫"
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        task_dir = f"./output/task_{timestamp}"
        os.makedirs(task_dir, exist_ok=True)
        resp = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"用户需求：{user_msg}\n\n{tools_constraints}\n\n请输出PRD大纲。")
        ])
        return {
            "messages": [{"role": "assistant", "content": "[PM] PRD已生成"}],
            "available_tools": tools_constraints,
            "prd": resp.content,
            "task_dir": task_dir
        }
    return pm_node


def make_engineer_node(llm_system: tuple):
    llm, system_prompt = llm_system
    def engineer_node(state: AgentState):
        prd = state.get("prd", "")
        tools_info = state.get("available_tools", "")
        review = state.get("aggregated_review", "")
        loop = state.get("loop_count", 0)
        if not prd or prd.startswith("ERROR"):
            return {"messages": [{"role": "assistant", "content": "[Engineer] 跳过：PRD无效"}], "code": "", "loop_count": loop}
        context = f"PRD如下：\n\n{prd}\n\n{tools_info}\n\n"
        if review and loop > 0:
            context += f"【第 {loop} 轮评审意见】\n{review}\n\n请根据以上意见修复代码，输出完整修复后的代码。\n"
        resp = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=context + "请编写完整代码。代码必须用 ```python ... ``` 包裹。")
        ])
        return {
            "messages": [{"role": "assistant", "content": f"[Engineer] 代码已生成（第{loop+1}轮）"}],
            "code": resp.content,
            "loop_count": loop + 1,
            "reviews": []
        }
    return engineer_node


def make_security_tester(llm_system: tuple):
    llm, system_prompt = llm_system
    def tester(state: AgentState):
        code = state.get("code", "")
        if not code or code.startswith("ERROR"):
            return {"reviews": [], "messages": [{"role": "assistant", "content": "[Tester-Security] 跳过"}]}
        resp = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"待审查代码：\n\n```python\n{code}\n```\n\n请给出安全审查意见。")
        ])
        return {"reviews": [f"【安全评审】{resp.content}"], "messages": [{"role": "assistant", "content": "[Tester-Security] 评审完成"}]}
    return tester


def make_performance_tester(llm_system: tuple):
    llm, system_prompt = llm_system
    def tester(state: AgentState):
        code = state.get("code", "")
        if not code or code.startswith("ERROR"):
            return {"reviews": [], "messages": [{"role": "assistant", "content": "[Tester-Performance] 跳过"}]}
        resp = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"待审查代码：\n\n```python\n{code}\n```\n\n请给出性能审查意见。")
        ])
        return {"reviews": [f"【性能评审】{resp.content}"], "messages": [{"role": "assistant", "content": "[Tester-Performance] 评审完成"}]}
    return tester


def make_logic_tester(llm_system: tuple):
    llm, system_prompt = llm_system
    def tester(state: AgentState):
        code = state.get("code", "")
        if not code or code.startswith("ERROR"):
            return {"reviews": [], "messages": [{"role": "assistant", "content": "[Tester-Logic] 跳过"}]}
        resp = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"待审查代码：\n\n```python\n{code}\n```\n\n请给出逻辑审查意见。")
        ])
        return {"reviews": [f"【逻辑评审】{resp.content}"], "messages": [{"role": "assistant", "content": "[Tester-Logic] 评审完成"}]}
    return tester


def make_aggregator_node(llm_system: tuple):
    """LLM-based aggregator that outputs decision string"""
    llm, _ = llm_system

    def aggregator_node(state: AgentState):
        reviews = state.get("reviews", [])
        if not reviews:
            return {"aggregated_review": "", "messages": [{"role": "assistant", "content": "[Aggregator] 无评审意见"}], "reviews": []}

        combined = "\n\n=== 评审分隔 ===\n\n".join(reviews)

        try:
            agg_prompt = f"""你是代码评审聚合专家。请分析以下评审意见，判断代码是否满足要求。

【评审意见】
{combined}

【决策规则】
- 如果所有评审都明确表示代码可以接受、运行正常、无严重问题，输出：满足
- 如果任一评审指出以下严重问题，输出：不满足
  * bug（代码会报错或产生错误结果）
  * 安全漏洞（SQL注入、命令注入、XSS等）
  * 逻辑错误（功能逻辑与需求不符、边界条件处理错误导致程序崩溃）
  * 高危或严重风险

注意：以下情况不应视为"不满足"：
- 性能优化建议（如添加timeout、降低DPI等）
- 代码可读性建议
- 功能扩展建议（如添加交互式图表、支持更多输出格式）
- 边界条件的改进建议（但不导致程序崩溃）
- 低危/中危观察项

【输出格式】
只输出单个词：满足 或 不满足
不要输出其他任何内容。"""

            resp = llm.invoke([HumanMessage(content=agg_prompt)])
            decision = resp.content.strip()
            if decision not in ["满足", "不满足"]:
                decision = "不满足"
        except Exception as e:
            print(f"[Aggregator] LLM 调用失败: {e}")
            decision = "不满足"

        return {
            "aggregated_review": combined,
            "aggregator_decision": decision,
            "messages": [{"role": "assistant", "content": f"[Aggregator] 评审聚合完成，决策: {decision}"}],
            "reviews": []
        }

    return aggregator_node


def make_executor_node(llm_system: tuple):
    """Executor with LLM-based auto-fix on execution failure"""
    llm, _ = llm_system

    def executor_node(state: AgentState):
        code = state.get("code", "")
        task_dir = state.get("task_dir", "")
        available_tools = state.get("available_tools", "")
        loop = state.get("loop_count", 0)

        if not code or code.startswith("ERROR"):
            return {"messages": [{"role": "assistant", "content": "[Executor] 跳过：代码无效"}], "tool_results": ""}
        if not task_dir:
            return {"messages": [{"role": "assistant", "content": "[Executor] 错误：无task_dir"}], "tool_results": "ERROR: 无task_dir"}

        clean_code = extract_code_block(code)
        current_code = clean_code
        max_retries = 2

        for attempt in range(max_retries + 1):
            code_path = f"{task_dir}/code_v{loop}_attempt{attempt}.py"
            with open(code_path, "w", encoding="utf-8") as f:
                f.write(current_code)
            print(f"[Executor] 代码已保存: {code_path} ({len(current_code)} chars)")

            result = python_executor.invoke({"code": current_code})
            print(f"[Executor] 执行结果: {result[:200]}...")

            if not result.startswith("❌"):
                return {
                    "messages": [{"role": "assistant", "content": f"[Executor] 执行成功（第{attempt+1}次尝试）"}],
                    "tool_results": result
                }

            if attempt < max_retries:
                print(f"[Executor] 执行失败，尝试用 LLM 修复（第 {attempt+1}/{max_retries} 次）...")

                fix_prompt = f"""代码执行失败，请修复以下错误：

【错误信息】
{result}

【当前代码】
```python
{current_code}
```

【可用工具约束】
{available_tools}

请直接输出修复后的完整代码，用 ```python ... ``` 包裹。
只修复错误，不要添加额外功能。"""

                try:
                    resp = llm.invoke([HumanMessage(content=fix_prompt)])
                    fixed_code = extract_code_block(resp.content)
                    if fixed_code and fixed_code != current_code:
                        current_code = fixed_code
                        print(f"[Executor] LLM 已修复代码，准备重试")
                        continue
                except Exception as e:
                    print(f"[Executor] LLM 修复失败: {e}")

            break

        final_result = python_executor.invoke({"code": current_code})
        return {
            "messages": [{"role": "assistant", "content": "[Executor] 代码执行完成（修复后）"}],
            "tool_results": final_result
        }

    return executor_node


def extract_code_block(text: str) -> str:
    match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


# ========== 2. 新增：human_gate 节点（核心） ==========
def human_gate(state: AgentState):
    """
    人工审批节点：
    - 首次执行：调用 interrupt() 暂停 Graph，等待外部 POST resume
    - 恢复执行：接收 Command(resume=decision)，解析决策并更新状态
    """
    # 如果已经审批过（例如循环回来的情况），直接放行
    if state.get("human_approved") and state.get("human_decision"):
        return {
            "messages": [{"role": "assistant", "content": "[HumanGate] 已审批，继续执行"}],
            "human_decision": state.get("human_decision"),
            "human_approved": True
        }

    # 触发 interrupt：payload 会返回给前端，供审批 UI 展示
    decision = interrupt({
        "stage": "post_aggregator",
        "aggregated_review": state.get("aggregated_review", ""),
        "loop_count": state.get("loop_count", 0),
        "task_dir": state.get("task_dir", ""),
        "prompt": "请审批聚合评审结果。输入：通过 / 修改_xxx / 拒绝"
    })

    # resume 后执行到这里：解析人类决策
    is_approved = decision in ("通过", "approve", "yes", "y")
    is_modify = decision.startswith("修改") or decision.startswith("modify")

    return {
        "messages": [{"role": "assistant", "content": f"[HumanGate] 人类决策: {decision}"}],
        "human_decision": decision,
        "human_approved": is_approved and not is_modify  # "修改"不算通过，需要回 engineer
    }


# ========== 3. 条件边 ==========
def map_testers(state: AgentState):
    shared = {"code": state.get("code", ""), "available_tools": state.get("available_tools", ""), "prd": state.get("prd", "")}
    return [Send("tester_security", shared), Send("tester_performance", shared), Send("tester_logic", shared)]


def route_after_aggregate(state: AgentState):
    loop = state.get("loop_count", 0)
    review = state.get("aggregated_review", "").lower()
    if loop >= 3:
        return "human_gate"  # 强制进入人工审批
    critical = ["严重", "致命", "高危", "必须修复", "critical", "bug", "错误", "漏洞", "泄漏", "异常", "未通过", "不通过"]
    needs_fix = any(k in review for k in critical)
    return "human_gate"  # 无论是否有问题，都先过人工审批（Step 3 的核心）


def route_human(state: AgentState):
    """human_gate 后的路由：根据人类决策分发"""
    decision = state.get("human_decision", "")
    loop = state.get("loop_count", 0)

    if decision in ("通过", "approve", "yes", "y"):
        return "executor"
    if decision.startswith("修改") or decision.startswith("modify"):
        if loop >= 3:
            return "executor"  # 超过循环上限，强制结束
        return "engineer"  # 回到 engineer 重写，带上 aggregated_review
    return END  # 拒绝或其他，结束


# ========== 4. 组装 StateGraph（关键拓扑变更） ==========
def build_graph(api_key: str):
    registry = LLMRegistry(
        base_url="https://token-plan-cn.xiaomimimo.com/v1",
        api_key=api_key
    )

    builder = StateGraph(AgentState)

    builder.add_node("pm", make_pm_node(registry.pm))
    builder.add_node("engineer", make_engineer_node(registry.engineer))
    builder.add_node("tester_security", make_security_tester(registry.tester_security))
    builder.add_node("tester_performance", make_performance_tester(registry.tester_performance))
    builder.add_node("tester_logic", make_logic_tester(registry.tester_logic))
    builder.add_node("aggregator", make_aggregator_node(registry.aggregator))
    builder.add_node("human_gate", human_gate)
    builder.add_node("executor", make_executor_node(registry.executor))

    builder.set_entry_point("pm")
    builder.add_edge("pm", "engineer")

    builder.add_conditional_edges("engineer", map_testers,
                                  ["tester_security", "tester_performance", "tester_logic"])
    builder.add_edge("tester_security", "aggregator")
    builder.add_edge("tester_performance", "aggregator")
    builder.add_edge("tester_logic", "aggregator")

    builder.add_conditional_edges("aggregator", route_after_aggregate, {"human_gate": "human_gate"})

    builder.add_conditional_edges("human_gate", route_human, {
        "engineer": "engineer",
        "executor": "executor",
        END: END
    })

    builder.add_edge("executor", END)

    return builder


# ========== 5. 单机测试模式（验证 interrupt + human_gate 机制） ==========
def test_interrupt_locally():
    """验证 interrupt + Command(resume) 链路"""
    API_KEY = os.getenv("MIMO_API_KEY", "your-api-key-here")
    memory = MemorySaver()
    graph = build_graph(API_KEY).compile(checkpointer=memory)
    thread_id = "day4-step3-local-test"
    config = {"configurable": {"thread_id": thread_id}}

    inputs = {
        "messages": [{"role": "user", "content": "写一个获取GitHub仓库issues创建时间分布并生成时间轴图表的Python程序"}],
        "loop_count": 0,
        "human_approved": False,
        "human_decision": "",
        "available_tools": "",
        "prd": "",
        "code": "",
        "reviews": [],
        "aggregated_review": "",
        "tool_results": "",
        "task_dir": ""
    }

    print("=" * 60)
    print("阶段 1：首次执行，预期在 human_gate 触发 interrupt")
    print("=" * 60)

    for event in graph.stream(inputs, config=config):
        for node_name, output in event.items():
            if node_name == "__interrupt__":
                interrupt_data = output[0].value if isinstance(output, tuple) else output
                print(f"⏸️ [human_gate] 触发中断 - 等待人工审批")
                print(f"   聚合评审摘要: {interrupt_data.get('aggregated_review', '')[:200]}...")
            elif isinstance(output, dict):
                msg = output.get("messages", [{}])[-1].get("content", "")
                print(f"📍 [{node_name}] {msg}")
            else:
                print(f"📍 [{node_name}] 输出: {type(output).__name__}")

    # 检查中断状态
    state = graph.get_state(config)
    print(f"\n🔍 当前暂停点: {state.next}")
    print(f"🔍 聚合评审: {state.values.get('aggregated_review', '')[:300]}...")
    print(f"🔍 当前轮次: {state.values.get('loop_count', 0)}")

    # 模拟人类审批
    print("\n" + "=" * 60)
    print("阶段 2：模拟人类输入 '修改_增加异常处理' 恢复执行")
    print("=" * 60)

    for event in graph.stream(Command(resume="修改_增加异常处理"), config=config):
        for node_name, output in event.items():
            if node_name == "__interrupt__":
                interrupt_data = output[0].value if isinstance(output, tuple) else output
                print(f"⏸️ [human_gate] 触发中断 - 等待人工审批")
            elif isinstance(output, dict):
                msg = output.get("messages", [{}])[-1].get("content", "")
                print(f"📍 [{node_name}] {msg}")
                if "code" in output:
                    print(f"   Code 长度: {len(output['code'])} chars | 轮次: {output.get('loop_count', 'N/A')}")
            else:
                print(f"📍 [{node_name}] 输出: {type(output).__name__}")

    # 再次检查状态（可能再次停在 human_gate，或已完成）
    state2 = graph.get_state(config)
    print(f"\n🔍 第二次暂停点: {state2.next}")
    print(f"🔍 人类决策: {state2.values.get('human_decision', 'N/A')}")

    # 如果再次中断，模拟通过
    if state2.next:
        print("\n" + "=" * 60)
        print("阶段 3：模拟人类输入 '通过' 最终放行")
        print("=" * 60)
        for event in graph.stream(Command(resume="通过"), config=config):
            for node_name, output in event.items():
                if node_name == "__interrupt__":
                    interrupt_data = output[0].value if isinstance(output, tuple) else output
                    print(f"⏸️ [human_gate] 触发中断 - 等待人工审批")
                elif isinstance(output, dict):
                    msg = output.get("messages", [{}])[-1].get("content", "")
                    print(f"📍 [{node_name}] {msg}")
                else:
                    print(f"📍 [{node_name}] 输出: {type(output).__name__}")

    print("\n✅ 单机 interrupt 测试完成")


# ========== 6. 运行入口 ==========
if __name__ == "__main__":
    test_interrupt_locally()
