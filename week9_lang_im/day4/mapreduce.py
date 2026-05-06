"""
Day 4 Step 2/3: 并行 Map-Reduce 评审 + 自动循环修复
- 3 个 Tester LLM 实例并行（安全/性能/逻辑）
- 聚合后自动路由回 Engineer 修复（上限 3 轮）
- 复用 tools.py，保持工具与编排解耦
"""

from typing import TypedDict, Annotated, List
import operator
import os
import json
import re
from datetime import datetime
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Send
from langchain_openai import ChatOpenAI
from langchain_classic.schema import SystemMessage, HumanMessage

from tools import get_current_time, python_executor


# ========== 0. 自定义 Reducer（并行冲突解决 + 循环清空） ==========
def reviews_reducer(old: List[str], new: List[str]) -> List[str]:
    """
    关键设计：
    - 并行 tester 返回单条评审时：追加合并（old + new）
    - engineer/aggregator 返回 [] 时：清空列表，为下一轮循环做准备
    若用 operator.add，[] 会被忽略，无法实现清空。
    """
    if not new:
        return []
    return old + new


# ========== 1. 状态定义 ==========
class AgentState(TypedDict):
    messages: Annotated[List[dict], operator.add]
    available_tools: str
    prd: str
    code: str
    reviews: Annotated[List[str], reviews_reducer]   # 并行评审结果
    aggregated_review: str
    aggregator_decision: str  # LLM输出的决策：满足/不满足
    task_dir: str
    loop_count: int
    human_approved: bool


# ========== 2. 多 LLM 实例注册表（扩展 3 类 Tester） ==========
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
        return (
            self._create(0.7),
            "你是资深产品经理。将模糊需求转化为结构化PRD，包含功能点、接口定义、异常处理。"
        )

    @property
    def engineer(self):
        return (
            self._create(0.2),
            "你是资深Python工程师。只输出高质量、带注释、可直接运行的代码。禁止输出解释性文字。"
        )

    @property
    def tester_security(self):
        return (
            self._create(0.3),
            "你是安全审计专家。逐行检查代码中的安全漏洞：SQL注入、命令注入、路径遍历、敏感信息硬编码、不安全的反序列化。用中文输出，标记风险等级（高/中/低）。"
        )

    @property
    def tester_performance(self):
        return (
            self._create(0.3),
            "你是性能优化专家。检查代码的时间复杂度、空间复杂度、I/O效率、内存泄漏风险、不必要的循环或重复计算。用中文输出，给出量化建议。"
        )

    @property
    def tester_logic(self):
        return (
            self._create(0.3),
            "你是逻辑审查专家。检查代码的边界条件处理、异常分支覆盖、状态机正确性、并发安全性、与PRD需求的一致性。用中文输出，列出遗漏的功能点。"
        )

    @property
    def aggregator(self):
        return (self._create(0.1), "")

    @property
    def executor(self):
        return (self._create(0.2), "")


# ========== 3. 节点工厂 ==========
def make_pm_node(llm_system: tuple):
    llm, system_prompt = llm_system
    from multi_llm import load_tools_constraints  # 复用 Step 1 的工具约束提取

    def pm_node(state: AgentState):
        user_msg = state["messages"][-1].get("content", "") if state["messages"] else "写一个爬虫"
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        task_dir = f"./output/task_{timestamp}"
        os.makedirs(task_dir, exist_ok=True)

        tools_constraints = load_tools_constraints()

        try:
            resp = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"""用户需求：{user_msg}

{tools_constraints}

请输出PRD大纲，必须包含：
1. 使用的工具（如 python_executor 执行代码）
2. 允许的 Python 模块（只能使用上述列出的模块）
3. 功能点和接口定义""")
            ])
            return {
                "messages": [{"role": "assistant", "content": "[PM] PRD已生成"}],
                "available_tools": tools_constraints,
                "prd": resp.content,
                "task_dir": task_dir
            }
        except Exception as e:
            return {
                "messages": [{"role": "assistant", "content": f"[PM] 异常: {str(e)}"}],
                "available_tools": tools_constraints,
                "prd": f"ERROR: {str(e)}",
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
            return {
                "messages": [{"role": "assistant", "content": "[Engineer] 跳过：PRD无效"}],
                "code": "",
                "loop_count": loop
            }

        # 构造上下文：第一轮只有 PRD，后续轮次加入评审意见
        context = f"PRD如下：\n\n{prd}\n\n{tools_info}\n\n"
        if review and loop > 0:
            context += f"""【第 {loop} 轮评审意见】
{review}

【强制要求】
请根据以上评审意见修复代码，输出完整修复后的代码。
特别注意：只使用白名单中的模块，禁止引入新依赖。
"""

        try:
            resp = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=context + "请编写完整代码。代码必须用 ```python ... ``` 包裹。")
            ])
            return {
                "messages": [{"role": "assistant", "content": f"[Engineer] 代码已生成（第{loop+1}轮）"}],
                "code": resp.content,
                "loop_count": loop + 1,
                "reviews": []  # 触发 reviews_reducer 清空旧评审
            }
        except Exception as e:
            return {
                "messages": [{"role": "assistant", "content": f"[Engineer] 异常: {str(e)}"}],
                "code": f"ERROR: {str(e)}",
                "loop_count": loop
            }

    return engineer_node


# 3 个并行 Tester（闭包绑定独立 LLM 实例）
def make_security_tester(llm_system: tuple):
    llm, system_prompt = llm_system
    def tester(state: AgentState):
        code = state.get("code", "")
        if not code or code.startswith("ERROR"):
            return {"reviews": [], "messages": [{"role": "assistant", "content": "[Security Tester] 跳过：代码无效"}]}
        try:
            resp = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"待审查代码：\n\n```python\n{code}\n```\n\n请给出安全审查意见。标记风险等级（高/中/低）。")
            ])
            return {"reviews": [f"【安全评审】{resp.content}"], "messages": [{"role": "assistant", "content": "[Security Tester] 评审完成"}]}
        except Exception as e:
            return {"reviews": [f"【安全评审】异常: {e}"], "messages": [{"role": "assistant", "content": f"[Security Tester] 异常: {str(e)}"}]}
    return tester


def make_performance_tester(llm_system: tuple):
    llm, system_prompt = llm_system
    def tester(state: AgentState):
        code = state.get("code", "")
        if not code or code.startswith("ERROR"):
            return {"reviews": [], "messages": [{"role": "assistant", "content": "[Performance Tester] 跳过：代码无效"}]}
        try:
            resp = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"待审查代码：\n\n```python\n{code}\n```\n\n请给出性能审查意见。给出量化建议。")
            ])
            return {"reviews": [f"【性能评审】{resp.content}"], "messages": [{"role": "assistant", "content": "[Performance Tester] 评审完成"}]}
        except Exception as e:
            return {"reviews": [f"【性能评审】异常: {e}"], "messages": [{"role": "assistant", "content": f"[Performance Tester] 异常: {str(e)}"}]}
    return tester


def make_logic_tester(llm_system: tuple):
    llm, system_prompt = llm_system
    def tester(state: AgentState):
        code = state.get("code", "")
        if not code or code.startswith("ERROR"):
            return {"reviews": [], "messages": [{"role": "assistant", "content": "[Logic Tester] 跳过：代码无效"}]}
        try:
            resp = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"待审查代码：\n\n```python\n{code}\n```\n\n请给出逻辑审查意见。检查边界条件和 PRD 一致性。")
            ])
            return {"reviews": [f"【逻辑评审】{resp.content}"], "messages": [{"role": "assistant", "content": "[Logic Tester] 评审完成"}]}
        except Exception as e:
            return {"reviews": [f"【逻辑评审】异常: {e}"], "messages": [{"role": "assistant", "content": f"[Logic Tester] 异常: {str(e)}"}]}
    return tester


def make_aggregator_node(llm_system: tuple):
    """LLM-based aggregator that outputs decision string"""
    llm, _ = llm_system

    def aggregator_node(state: AgentState):
        reviews = state.get("reviews", [])
        if not reviews:
            return {
                "aggregated_review": "",
                "aggregator_decision": "满足",
                "messages": [{"role": "assistant", "content": "[Aggregator] 无评审意见"}],
                "reviews": []
            }

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

        if not code or code.startswith("ERROR"):
            return {
                "messages": [{"role": "assistant", "content": "[Executor] 跳过：代码无效"}],
                "tool_results": ""
            }
        if not task_dir:
            return {
                "messages": [{"role": "assistant", "content": "[Executor] 错误：无task_dir"}],
                "tool_results": "ERROR: 无task_dir"
            }

        clean_code = extract_code_block(code)
        current_code = clean_code
        max_retries = 2

        for attempt in range(max_retries + 1):
            code_path = f"{task_dir}/code_v{state.get('loop_count', 0)}_attempt{attempt}.py"
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


# ========== 4. 条件边（Map-Reduce 核心） ==========
def map_testers(state: AgentState):
    """
    Map 阶段：从 Engineer 发出 3 个 Send，LangGraph 并行调度。
    传递当前 state，确保 tester 能读取到 engineer 生成的 code。
    """
    return [
        Send("tester_security", state),
        Send("tester_performance", state),
        Send("tester_logic", state)
    ]


def route_after_aggregate(state: AgentState):
    """
    Reduce 后路由：根据 LLM 决策判断是否需要循环修复。
    - loop_count >= 3 强制进入执行（防止无限循环）
    - aggregator_decision == "不满足" 时返回 engineer 修复
    """
    loop = state.get("loop_count", 0)
    decision = state.get("aggregator_decision", "")

    if loop >= 3:
        return "executor"

    # 根据 LLM 的决策判断："不满足" 则修复，"满足" 则执行
    if decision == "不满足":
        return "engineer"
    else:
        return "executor"


# ========== 5. 辅助函数（复用 Step 1） ==========
def extract_code_block(text: str) -> str:
    match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def save_outputs(state: dict, timestamp: str, user_input: str = ""):
    task_dir = state.get("task_dir", "")
    if not task_dir:
        task_id = timestamp.replace(":", "-").replace(" ", "_")
        task_dir = f"./output/task_{task_id}"
    os.makedirs(task_dir, exist_ok=True)

    outputs = {
        "timestamp": timestamp,
        "task_id": os.path.basename(task_dir),
        "loop_count": state.get("loop_count", 0),
        "available_tools": state.get("available_tools", ""),
        "prd": state.get("prd", ""),
        "code": state.get("code", ""),
        "aggregated_review": state.get("aggregated_review", ""),
        "tool_results": state.get("tool_results", ""),
    }

    if not user_input:
        for msg in state.get("messages", []):
            if msg.get("role") == "user":
                user_input = msg.get("content", "")
                break

    with open(f"{task_dir}/input.txt", "w", encoding="utf-8") as f:
        f.write(user_input)

    with open(f"{task_dir}/result.json", "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)

    if outputs["prd"]:
        with open(f"{task_dir}/prd.md", "w", encoding="utf-8") as f:
            f.write(outputs["prd"])

    if outputs["code"]:
        with open(f"{task_dir}/code.py", "w", encoding="utf-8") as f:
            f.write(extract_code_block(outputs["code"]))

    # 保存聚合评审
    if outputs["aggregated_review"]:
        with open(f"{task_dir}/review.md", "w", encoding="utf-8") as f:
            f.write(outputs["aggregated_review"])

    # 保存所有单轮评审（分三轮）
    all_reviews = state.get("all_reviews", [])
    if all_reviews:
        reviews_md = "# 代码评审记录\n\n"
        reviews_md += f"## 共 {len(all_reviews)} 条评审\n\n"
        
        # 按轮次分组（每轮3条评审）
        for i in range(0, len(all_reviews), 3):
            round_num = (i // 3) + 1
            reviews_md += f"---\n\n## 第 {round_num} 轮评审\n\n"
            for review in all_reviews[i:i+3]:
                reviews_md += f"{review}\n\n"
        
        with open(f"{task_dir}/all_reviews.md", "w", encoding="utf-8") as f:
            f.write(reviews_md)

    if outputs["tool_results"]:
        with open(f"{task_dir}/tool_results.txt", "w", encoding="utf-8") as f:
            f.write(outputs["tool_results"])

    print(f"\n📁 输出已保存到 {task_dir}/")


# ========== 6. 组装 StateGraph ==========
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
    builder.add_node("executor", make_executor_node(registry.executor))

    builder.set_entry_point("pm")
    builder.add_edge("pm", "engineer")

    # 关键：条件边返回 Send 列表 → 并行触发 3 个 tester
    builder.add_conditional_edges("engineer", map_testers,
                                  ["tester_security", "tester_performance", "tester_logic"])

    # 3 个并行节点汇聚到 aggregator（LangGraph 自动等待全部完成）
    builder.add_edge("tester_security", "aggregator")
    builder.add_edge("tester_performance", "aggregator")
    builder.add_edge("tester_logic", "aggregator")

    # 聚合后判断：修复 or 执行
    builder.add_conditional_edges("aggregator", route_after_aggregate, {
        "engineer": "engineer",
        "executor": "executor"
    })

    builder.add_edge("executor", END)

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


# ========== 7. 运行测试 ==========
if __name__ == "__main__":
    API_KEY = os.getenv("MIMO_API_KEY", "your-api-key-here")

    graph = build_graph(API_KEY)
    try:
        png_bytes = graph.get_graph().draw_mermaid_png()
        with open("map_graph.png", "wb") as f:
            f.write(png_bytes)
        print("✅ 已生成可视化图：map_graph.png")
    except Exception as e:
        print(f"⚠️ 无法生成可视化图：{e}")

    config = {"configurable": {"thread_id": "day4-step2-mapreduce"}}

    start_time = get_current_time.invoke({})
    print(f"🚀 启动并行 Map-Reduce 评审测试... [{start_time}]\n")

    inputs = {
        "messages": [{"role": "user", "content": """请执行 Python 代码完成以下任务：
1. 使用 requests 访问 GitHub API，获取 'langchain-ai/langgraph' 仓库的最近 10 个 issues
2. 提取 issue 标题和创建时间
3. 使用 matplotlib 绘制时间线图
4. 保存到 ./langgraph_issues.png
5. 打印每个 issue 的标题
6. 保证中文字符的输出
"""}],
        "loop_count": 0,
        "human_approved": False,
        "available_tools": "",
        "prd": "",
        "code": "",
        "reviews": [],
        "aggregated_review": "",
        "aggregator_decision": "",
        "tool_results": "",
        "task_dir": ""
    }

    final_state = {}
    all_reviews = []  # 收集所有评审

    for event in graph.stream(inputs, config=config):
        for node_name, output in event.items():
            timestamp = get_current_time.invoke({})
            msg = output.get("messages", [{}])[-1].get("content", "")
            print(f"[{timestamp}] 📍 [{node_name}] {msg}")

            if "code" in output:
                print(f"   Code 长度: {len(output['code'])} chars | 轮次: {output.get('loop_count', 'N/A')}")
            if "reviews" in output and output["reviews"]:
                print(f"   新增评审: {len(output['reviews'])} 条")
                all_reviews.extend(output["reviews"])
            if "aggregated_review" in output and output["aggregated_review"]:
                print(f"   聚合评审长度: {len(output['aggregated_review'])} chars")
            if "tool_results" in output:
                print(f"   执行结果: {str(output['tool_results'])[:150]}...")

            final_state.update(output)
    
    # 保存所有评审到单独文件
    final_state["all_reviews"] = all_reviews

    end_time = get_current_time.invoke({})
    print(f"\n✅ 测试完成 [{end_time}]")

    save_outputs(final_state, end_time, user_input=inputs["messages"][0]["content"])
