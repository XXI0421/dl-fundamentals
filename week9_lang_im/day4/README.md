# LangGraph 多Agent协作与人工干预教程（Day 4）

## 概述

本目录包含 LangGraph 多 Agent 协作的高级特性教学材料：多 LLM 实例注册表、Map-Reduce 并行评审、人工干预嫁接（interrupt + Command）以及向量数据库集成。通过这些特性，可以实现复杂的代码生成-评审-修复工作流，支持人工审批和自动修复。

**学习路径（递进关系）：**
```
Day 3 (Time Travel + FastAPI)
         ↓
Day 4 Step 1 (multi_llm.py)      Day 4 Step 2 (mapreduce.py)      Day 4 Step 3 (human_gate.py)
         ↓                              ↓                              ↓
  多 LLM 实例注册表              Map-Reduce 并行评审              人工干预嫁接
  PM/Engineer/Tester 分工         3 类评审并行 + 自动循环           interrupt + Command(resume)
```

## 核心概念

### 什么是多 LLM 实例注册表？

多 LLM 实例注册表允许为不同节点配置独立的 LLM 实例，每个实例可以有不同的：

- 模型选择（如 GPT-4、Claude、国产模型）
- Temperature 参数（创意型 vs 精确型）
- System Prompt（角色定义）

### 什么是 Map-Reduce 并行评审？

Map-Reduce 是一种分布式编程模式，在 LangGraph 中通过 `Send` 类实现：

- **Map 阶段**：一个输入同时发送给多个评审节点并行处理
- **Reduce 阶段**：聚合所有评审结果，统一判断

### 什么是人工干预嫁接？

人工干预嫁接是将人工审批节点插入自动化流程的技术：

- 使用 `interrupt()` 暂停图执行
- 等待外部人工决策
- 使用 `Command(resume=decision)` 恢复执行

## 文件说明

### 1. tools.py - 工具定义模块

**功能说明：**
- 定义 Agent 可用的工具集
- 实现安全的 Python 代码执行沙箱
- 提供 RAG 检索、数学计算、时间获取等功能

**核心工具：**

| 工具 | 说明 |
|------|------|
| `python_executor` | 在受限沙箱中执行 Python 代码，支持数据处理和图表生成 |
| `get_current_time` | 返回当前时间 |
| `search` | 从向量库检索相关信息 |
| `calculate` | 执行数学计算 |

**Python 执行器约束：**
- 允许的模块：`matplotlib`, `requests`, `json`, `math`, `numpy`, `pandas`, `os`, `glob`, `io`, `pathlib`, `datetime`, `typing`, `time`, `re` 等
- 禁止：`subprocess`, `argparse`, `sys.exit`, `pickle`, `marshal` 等

---

### 2. vectorstore.py - 向量数据库模块

**功能说明：**
- 创建和管理向量数据库
- 支持多种检索策略：Base、Multi-Query、Ensemble、Compression、Rerank

**核心函数：**

```python
from vectorstore import load_and_split, build_vectorstore, get_retriever

chunks = load_and_split(data_dir="data")
vectorstore = build_vectorstore(chunks)
retriever = get_retriever("base", vectorstore, chunks)
```

**检索策略说明：**

| 策略 | 说明 |
|------|------|
| `base` | 基础向量检索 |
| `multi` | 多查询检索（使用 LLM 生成多个查询） |
| `ensemble` | 混合检索（BM25 + 向量） |
| `compress` | 上下文压缩检索 |
| `rerank` | 重排序检索（先海选再精排） |

---

### 3. multi_llm.py - Step 1：多 LLM 实例注册表

**功能说明：**
- 演示多 LLM 实例注册表的使用方法
- 为 PM/Engineer/Tester 配置独立 LLM 实例
- PM 节点提取工具约束传递给后续节点

**工作流程：**
```
PM（产品经理） → Engineer（工程师） → Tester（评审） → Executor（执行）
     ↓
  提取工具约束
```

**核心代码：**

```python
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
        return (self._create(0.7), "你是资深产品经理...")

    @property
    def engineer(self):
        return (self._create(0.2), "你是资深Python工程师...")

    @property
    def tester(self):
        return (self._create(0.3), "你是代码审查专家...")
```

**运行方式：**
```bash
python multi_llm.py
```

**输出示例：**
```
[2025-05-06 10:30:00] 📍 [pm] [PM] PRD已生成
[2025-05-06 10:30:05] 📍 [engineer] [Engineer] 代码已生成
[2025-05-06 10:30:10] 📍 [tester] [Tester] 评审完成
[2025-05-06 10:30:15] 📍 [executor] [Executor] 代码执行完成

✅ 测试完成 [2025-05-06 10:30:20]
📁 输出已保存到 ./output/task_2025-05-06_10-30-00/
```

---

### 4. mapreduce.py - Step 2：Map-Reduce 并行评审

**功能说明：**
- 3 个独立 Tester LLM 实例并行运行（安全/性能/逻辑）
- 聚合后 LLM 判断是否需要循环修复（上限 3 轮）
- 自动修复执行失败的代码

**工作流程：**
```
                    ┌→ tester_security ─┐
                    │                  │
Engineer → map_testers ─→ tester_performance ─→ aggregator ─→ (engineer/executor)
                    │                  │
                    └→ tester_logic ───┘
                              ↓
                         3 并行 Send
```

**核心代码：**

```python
from langgraph.types import Send

def map_testers(state: AgentState):
    """Map 阶段：并行派发 3 个评审任务"""
    return [
        Send("tester_security", state),
        Send("tester_performance", state),
        Send("tester_logic", state)
    ]

def route_after_aggregate(state: AgentState):
    """Reduce 后路由：根据 LLM 决策判断是否需要循环修复"""
    loop = state.get("loop_count", 0)
    decision = state.get("aggregator_decision", "")

    if loop >= 3:
        return "executor"
    return "engineer" if decision == "不满足" else "executor"
```

**自定义 Reducer：**

```python
def reviews_reducer(old: List[str], new: List[str]) -> List[str]:
    """并行评审结果合并 + 循环清空"""
    if not new:
        return []  # engineer 返回 [] 时清空旧评审
    return old + new
```

**运行方式：**
```bash
python mapreduce.py
```

**输出示例：**
```
📍 [pm] [PM] PRD已生成
📍 [engineer] [Engineer] 代码已生成（第1轮）
📍 [tester_security] [Security Tester] 评审完成
📍 [tester_performance] [Performance Tester] 评审完成
📍 [tester_logic] [Logic Tester] 评审完成
📍 [aggregator] [Aggregator] 评审聚合完成，决策: 不满足

📍 [engineer] [Engineer] 代码已生成（第2轮）
📍 [aggregator] [Aggregator] 评审聚合完成，决策: 满足
📍 [executor] [Executor] 执行成功（第1次尝试）

✅ 自动修复测试完成
```

---

### 5. human_gate.py - Step 3：人工干预嫁接

**功能说明：**
- 在 aggregator 后插入 human_gate 节点
- 使用 `interrupt()` 暂停 Graph，等待外部 POST 恢复
- 支持 `Command(resume=decision)` 恢复执行
- 决策类型：通过 / 修改_xxx / 拒绝

**工作流程：**
```
Engineer → [3 并行 Tester] → Aggregator → HumanGate(interrupt)
                                                    ↓
                                              等待人工决策
                                                    ↓
                        ┌─────────────────────────────┴─────────────────────────────┐
                        ↓                                                           ↓
                   [通过]                                                       [修改_xxx]
                        ↓                                                           ↓
                   Executor                                                    Engineer
                        ↓                                                           ↓
                   [结束]                                                       [下一轮]
```

**核心代码：**

```python
from langgraph.types import interrupt, Command

def human_gate(state: AgentState):
    """人工审批节点：
    - 首次执行：调用 interrupt() 暂停 Graph
    - 恢复执行：接收 Command(resume=decision)，解析决策并更新状态
    """
    if state.get("human_approved") and state.get("human_decision"):
        return {"human_decision": state.get("human_decision"), ...}

    decision = interrupt({
        "stage": "post_aggregator",
        "aggregated_review": state.get("aggregated_review", ""),
        "loop_count": state.get("loop_count", 0),
        "prompt": "请审批聚合评审结果。输入：通过 / 修改_xxx / 拒绝"
    })

    is_approved = decision in ("通过", "approve", "yes", "y")
    is_modify = decision.startswith("修改") or decision.startswith("modify")

    return {
        "human_decision": decision,
        "human_approved": is_approved and not is_modify
    }

def route_human(state: AgentState):
    """human_gate 后的路由：根据人类决策分发"""
    decision = state.get("human_decision", "")
    loop = state.get("loop_count", 0)

    if decision in ("通过", "approve", "yes", "y"):
        return "executor"
    if decision.startswith("修改"):
        return "engineer" if loop < 3 else "executor"
    return END
```

**运行方式：**
```bash
python human_gate.py
```

**输出示例：**
```
============================================================
阶段 1：首次执行，预期在 human_gate 触发 interrupt
============================================================
📍 [pm] [PM] PRD已生成
📍 [engineer] [Engineer] 代码已生成（第1轮）
📍 [tester_security] [Tester-Security] 评审完成
📍 [tester_performance] [Tester-Performance] 评审完成
📍 [tester_logic] [Tester-Logic] 评审完成
📍 [aggregator] [Aggregator] 评审聚合完成，决策: 满足
⏸️ [human_gate] 触发中断 - 等待人工审批
   聚合评审摘要: 【安全评审】...

🔍 当前暂停点: ('human_gate',)
🔍 聚合评审: 【安全评审】...
🔍 当前轮次: 0

阶段 2：模拟人类输入 '修改_增加异常处理' 恢复执行
============================================================
📍 [engineer] [Engineer] 代码已生成（第2轮）

✅ 单机 interrupt 测试完成
```

## 多 LLM 实例注册表核心概念

### LLMRegistry 模式

```python
class LLMRegistry:
    def __init__(self, base_url: str, api_key: str, model: str):
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
        return (self._create(0.7), "系统提示词1")

    @property
    def engineer(self):
        return (self._create(0.2), "系统提示词2")
```

### 节点工厂模式

```python
def make_pm_node(llm_system: tuple):
    llm, system_prompt = llm_system

    def pm_node(state: AgentState):
        resp = llm.invoke([SystemMessage(content=system_prompt), ...])
        return {"prd": resp.content, ...}

    return pm_node
```

## Map-Reduce 核心概念

### Send 类并行派发

```python
from langgraph.types import Send

def map_testers(state: AgentState):
    return [
        Send("tester_security", {"code": state["code"]}),
        Send("tester_performance", {"code": state["code"]}),
        Send("tester_logic", {"code": state["code"]}),
    ]

builder.add_conditional_edges("engineer", map_testers, [
    "tester_security", "tester_performance", "tester_logic"
])
```

### 自定义 Reducer

```python
from typing import Annotated, List
import operator

class AgentState(TypedDict):
    reviews: Annotated[List[str], reviews_reducer]

def reviews_reducer(old: List[str], new: List[str]) -> List[str]:
    if not new:
        return []  # 清空列表
    return old + new  # 追加合并
```

## 人工干预核心概念

### interrupt() 暂停

```python
def human_gate(state: AgentState):
    decision = interrupt({
        "stage": "post_aggregator",
        "aggregated_review": state.get("aggregated_review", ""),
        "prompt": "请审批..."
    })
```

### Command(resume=) 恢复

```python
from langgraph.types import Command

for event in graph.stream(Command(resume="通过"), config):
    print(f"Event: {event}")
```

## 环境配置

### 安装依赖

```bash
pip install langgraph langgraph-checkpoint langchain-openai fastapi uvicorn sse-starlette
```

## 常见问题

### Q1: 多 LLM 注册表的优势是什么？

**A:** 不同节点可以使用不同的模型和参数，如 PM 用创意型模型（temperature=0.7），Engineer 用精确型模型（temperature=0.2）。

### Q2: Map-Reduce 和普通串行的区别？

**A:** 串行执行需要等待一个节点完成再执行下一个；Map-Reduce 通过 Send 并行执行多个节点，大幅提升效率。

### Q3: interrupt 和普通等待的区别？

**A:** 普通等待是同步的，会阻塞整个图；interrupt 会保存状态并退出，允许外部系统（如 API）恢复执行。

### Q4: 循环上限的作用是什么？

**A:** 防止无限循环，当 loop_count >= 3 时强制进入执行阶段，即使评审不通过也停止修复。

## 学习路径

1. **基础阶段**：运行 `multi_llm.py`，理解多 LLM 实例注册表
2. **并行阶段**：运行 `mapreduce.py`，理解 Map-Reduce 并行评审
3. **人工阶段**：运行 `human_gate.py`，理解人工干预嫁接
4. **实战阶段**：组合使用所有特性实现完整工作流

## 参考资料

- [LangGraph Multi-Agent 文档](https://langchain-ai.github.io/langgraph/concepts/multi_agent/)
- [LangGraph Send 文档](https://langchain-ai.github.io/langgraph/how-tos_branching/)
- [LangGraph Interrupt 文档](https://langchain-ai.github.io/langgraph/how-tos/interruption/)
