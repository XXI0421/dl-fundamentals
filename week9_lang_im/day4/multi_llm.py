"""
Day 4: Multi-Agent 骨架
- 3 个独立 LLM 实例（PM / Engineer / Tester）
- PM 节点读取 tools.py，提取工具约束传递给后续节点
- 通过 AgentState 共享 artifact 和工具约束
"""

from typing import TypedDict, Annotated, List
import operator
import os
import json
import re
from datetime import datetime
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_classic.schema import SystemMessage, HumanMessage

from tools import get_current_time, python_executor


# ========== 1. 工具约束提取 ==========
def load_tools_constraints():
    """读取 tools.py，提取可用工具和允许的模块"""
    tools_path = os.path.join(os.path.dirname(__file__), "tools.py")
    tools_path = os.path.abspath(tools_path)

    try:
        with open(tools_path, "r", encoding="utf-8") as f:
            content = f.read()

        tools_info = []
        tool_names = re.findall(r'@tool\s+def (\w+)\(', content)
        for name in tool_names:
            desc_match = re.search(rf'@tool\s+def {name}\([^)]*\)\s*->\s*str:\s*"""([^"]+)"""', content, re.DOTALL)
            if desc_match:
                tools_info.append(f"- {name}: {desc_match.group(1).split('.')[0]}")

        # 提取允许的模块（处理多行字典，可能在函数内部）
        allowed_modules = []
        allowed_match = re.search(r'_ALLOWED_MODULES\s*=\s*\{', content)
        if allowed_match:
            start = allowed_match.end() - 1
            brace_count = 1
            end = start
            for i in range(start + 1, len(content)):
                if content[i] == '{':
                    brace_count += 1
                elif content[i] == '}':
                    brace_count -= 1
                if brace_count == 0:
                    end = i
                    break
            allowed_text = content[start:end + 1]
            allowed_modules = re.findall(r'"([^"]+)"', allowed_text)

        # 提取预注入的全局变量
        pre_injected = re.findall(r'"(\w+)":\s*__import__\(', content)

        return f"""【可用工具】
{chr(10).join(tools_info)}

【允许的 Python 模块】（可直接 import）
{', '.join(sorted(allowed_modules))}

【预注入的全局变量】
{', '.join(pre_injected)}

【重要约束】
- 只能使用上述列出的模块，禁止使用其他模块
- 推荐使用预注入的全局变量（如 plt, requests, json, math）
"""
    except Exception as e:
        return f"【工具约束】（读取失败，使用默认）\n允许模块: os, glob, io, json, math, requests, matplotlib.pyplot\n预注入: plt, requests, json, math"


# ========== 2. 状态定义 ==========
class AgentState(TypedDict):
    messages: Annotated[List[dict], operator.add]
    available_tools: str
    prd: str
    code: str
    review: str
    task_dir: str
    loop_count: int
    human_approved: bool


# ========== 3. 多 LLM 实例注册表 ==========
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
            "你是资深Python工程师。只输出高质量、带注释、可直接运行的代码。"
        )

    @property
    def tester(self):
        return (
            self._create(0.3),
            "你是代码审查专家。逐行审查代码，列出潜在bug、边界条件错误、性能隐患。用中文输出。"
        )


# ========== 4. 节点工厂 ==========
def make_pm_node(llm_system: tuple):
    llm, system_prompt = llm_system
    tools_constraints = load_tools_constraints()

    def pm_node(state: AgentState):
        user_msg = state["messages"][-1].get("content", "") if state["messages"] else "写一个爬虫"

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        task_dir = f"./output/task_{timestamp}"
        os.makedirs(task_dir, exist_ok=True)

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

        if not prd or prd.startswith("ERROR"):
            return {
                "messages": [{"role": "assistant", "content": "[Engineer] 跳过：PRD无效"}],
                "code": ""
            }

        try:
            resp = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"""PRD如下：\n\n{prd}

{tools_info}

【强制要求】
1. 只能使用上述「允许的 Python 模块」列表中的模块
2. 推荐使用预注入的全局变量（plt, requests, json, math, os, glob 等）
3. 代码必须用 ```python ... ``` 包裹
4. 【禁止】使用以下模块：argparse, subprocess, sys.exit, traceback, pickle, marshal 等
5. 【禁止】使用任何未在白名单中的模块

【检查清单】
编写完代码后，逐行检查：
- 每个 import 语句是否在白名单中？
- 是否使用了任何系统级操作（os.system, subprocess 等）？
- 是否有不必要的复杂导入？""")
            ])
            return {
                "messages": [{"role": "assistant", "content": "[Engineer] 代码已生成"}],
                "code": resp.content
            }
        except Exception as e:
            return {
                "messages": [{"role": "assistant", "content": f"[Engineer] 异常: {str(e)}"}],
                "code": f"ERROR: {str(e)}"
            }

    return engineer_node


def make_tester_node(llm_system: tuple):
    llm, system_prompt = llm_system

    def tester_node(state: AgentState):
        code = state.get("code", "")
        tools_info = state.get("available_tools", "")

        if not code or code.startswith("ERROR"):
            return {
                "messages": [{"role": "assistant", "content": "[Tester] 跳过：代码无效"}],
                "review": ""
            }

        try:
            resp = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"""待审查代码：\n\n```python\n{code}\n```\n\n{tools_info}

【审查要求】（必须逐项检查）
1. 【关键】检查代码中 import 的模块是否都在上述「允许的 Python 模块」列表中
2. 如果发现使用了未列出的模块（如 traceback, subprocess, sys.exit 等），必须标记为严重问题
3. 检查是否有潜在 bug、边界条件错误
4. 检查是否符合 PRD 要求

【输出格式】
必须包含：
- 模块使用检查结果（列出代码中所有 import 的模块）
- 发现的问题列表（如有）
- 是否通过审查""")
            ])
            return {
                "messages": [{"role": "assistant", "content": "[Tester] 评审完成"}],
                "review": resp.content
            }
        except Exception as e:
            return {
                "messages": [{"role": "assistant", "content": f"[Tester] 异常: {str(e)}"}],
                "review": f"ERROR: {str(e)}"
            }

    return tester_node


def make_executor_node():
    def executor_node(state: AgentState):
        code = state.get("code", "")
        task_dir = state.get("task_dir", "")

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

        # 保存代码
        code_path = f"{task_dir}/code.py"
        clean_code = extract_code_block(code)
        with open(code_path, "w", encoding="utf-8") as f:
            f.write(clean_code)
        print(f"[Executor] 代码已保存: {code_path} ({len(clean_code)} chars)")

        # 执行代码
        try:
            result = python_executor.invoke({"code": clean_code})
            print(f"[Executor] 执行结果: {result}")
            return {
                "messages": [{"role": "assistant", "content": "[Executor] 代码执行完成"}],
                "tool_results": result
            }
        except Exception as e:
            error_msg = f"ERROR: {str(e)}"
            print(f"[Executor] {error_msg}")
            return {
                "messages": [{"role": "assistant", "content": f"[Executor] 调用异常: {str(e)}"}],
                "tool_results": error_msg
            }

    return executor_node


def extract_code_block(text: str) -> str:
    match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    match = re.search(r"```(.*?)```", text, re.DOTALL)
    if match:
        content = match.group(1).strip()
        if any(keyword in content for keyword in ['import ', 'def ', 'class ', 'print(', '#!/']):
            return content

    lines = text.strip().split('\n')
    if lines and any(lines[0].strip().startswith(x) for x in ['#!/', 'import ', 'from ', 'def ', 'class ']):
        clean_lines = [line for line in lines if not line.strip().startswith('```')]
        return '\n'.join(clean_lines).strip()

    clean_text = text
    while '```' in clean_text:
        start = clean_text.find('```')
        end = clean_text.find('```', start + 3)
        if end == -1:
            clean_text = clean_text[:start] + clean_text[start + 3:]
        else:
            clean_text = clean_text[:start] + clean_text[end + 3:]

    return clean_text.strip()


# ========== 5. 组装 StateGraph ==========
def build_graph(api_key: str):
    registry = LLMRegistry(
        base_url="https://token-plan-cn.xiaomimimo.com/v1",
        api_key=api_key
    )

    builder = StateGraph(AgentState)

    builder.add_node("pm", make_pm_node(registry.pm))
    builder.add_node("engineer", make_engineer_node(registry.engineer))
    builder.add_node("tester", make_tester_node(registry.tester))
    builder.add_node("executor", make_executor_node())

    builder.set_entry_point("pm")
    builder.add_edge("pm", "engineer")
    builder.add_edge("engineer", "tester")
    builder.add_edge("tester", "executor")
    builder.add_edge("executor", END)

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


# ========== 6. 保存输出 ==========
def save_outputs(state: dict, timestamp: str, user_input: str = ""):
    task_dir = state.get("task_dir", "")
    if not task_dir:
        task_id = timestamp.replace(":", "-").replace(" ", "_")
        task_dir = f"./output/task_{task_id}"
    os.makedirs(task_dir, exist_ok=True)

    outputs = {
        "timestamp": timestamp,
        "task_id": os.path.basename(task_dir),
        "available_tools": state.get("available_tools", ""),
        "prd": state.get("prd", ""),
        "code": state.get("code", ""),
        "review": state.get("review", ""),
        "tool_results": state.get("tool_results", ""),
    }

    if not user_input:
        messages = state.get("messages", [])
        for msg in messages:
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
        clean_code = extract_code_block(outputs["code"])
        with open(f"{task_dir}/code.py", "w", encoding="utf-8") as f:
            f.write(clean_code)

    if outputs["review"]:
        with open(f"{task_dir}/review.md", "w", encoding="utf-8") as f:
            f.write(outputs["review"])

    if outputs["tool_results"]:
        with open(f"{task_dir}/tool_results.txt", "w", encoding="utf-8") as f:
            f.write(outputs["tool_results"])

    print(f"\n📁 输出已保存到 {task_dir}/")


# ========== 7. 运行测试 ==========
if __name__ == "__main__":
    API_KEY = os.getenv("MIMO_API_KEY", "your-api-key-here")

    graph = build_graph(API_KEY)
    config = {"configurable": {"thread_id": "day4-multi-llm"}}

    start_time = get_current_time.invoke({})
    print(f"🚀 启动多 LLM 实例协作测试... [{start_time}]\n")

    inputs = {
        "messages": [{"role": "user", "content": """请执行 Python 代码完成以下任务：
1. 使用 requests 访问 GitHub API，获取 Agent 仓库中收藏前 10 个仓库
2. 从返回的 JSON 中提取仓库名（full_name）和 stars 数量（stargazers_count）
3. 使用 matplotlib 绘制水平条形图（barh），标题'Top 10 Agent Repositories on GitHub'
4. 保存到 ./agent_repos.png
5. 打印每个仓库的 stars 数量
"""}],
        "loop_count": 0,
        "human_approved": False,
        "available_tools": "",
        "prd": "",
        "code": "",
        "review": "",
        "tool_results": "",
        "task_dir": ""
    }

    final_state = {}

    for event in graph.stream(inputs, config=config):
        for node_name, output in event.items():
            node_start = get_current_time.invoke({})
            msg = output.get("messages", [{}])[-1].get("content", "")
            print(f"[{node_start}] 📍 [{node_name}] {msg}")

            if "available_tools" in output:
                print(f"   工具约束已加载")
            if "prd" in output:
                print(f"   PRD 长度: {len(output['prd'])} chars")
            if "code" in output:
                print(f"   Code 长度: {len(output['code'])} chars")
            if "review" in output:
                print(f"   Review 长度: {len(output['review'])} chars")
            if "tool_results" in output:
                print(f"   执行结果: {str(output['tool_results'])[:100]}...")

            final_state.update(output)

    end_time = get_current_time.invoke({})
    print(f"\n✅ 测试完成 [{end_time}]")

    save_outputs(final_state, end_time, user_input=inputs["messages"][0]["content"])