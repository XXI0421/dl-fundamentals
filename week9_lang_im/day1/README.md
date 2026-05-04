# LangGraph Multi-Agent 仿真教程（Day 1）

## 概述

本目录包含 LangGraph 多智能体系统的入门教学材料。通过模拟软件开发的完整流程，展示 LangGraph 的状态机机制、条件路由和人机协同功能。

**学习路径（递进关系）：**
```
state.py → human.py → simulate.py
  ↓           ↓           ↓
 基础      人机协同     状态持久化
```

## 核心概念

### 什么是 LangGraph 多智能体仿真？

LangGraph 多智能体仿真是通过状态机模拟多个 Agent 协作完成复杂任务的编程模式。每个 Agent 是一个节点，节点之间通过条件边动态路由，实现自动化的流程控制。

**核心优势：**
- 清晰的工作流程建模
- 支持循环和条件分支
- 自动状态管理
- 可视化的流程控制
- 人机协同支持

### LangGraph 核心组件

| 组件 | 说明 | 示例 |
|------|------|------|
| **State** | 所有节点共享的状态字典 | `{"messages": [], "prd": "", "code": ""}` |
| **Node** | 图中的节点，每个节点是一个函数 | `pm_node`, `engineer_node` |
| **Edge** | 节点之间的无条件连接 | `"pm" → "architect"` |
| **Conditional Edge** | 根据状态值动态决定下一个节点 | `tester` → `engineer` 或 `END` |
| **END** | 图的终止节点 | 结束图的执行 |

## 文件说明

### 1. state.py - 基础状态机 + 条件循环

**功能说明：**
- 演示 LangGraph 的基本概念和工作流程
- PM → Architect → Engineer → Tester 的完整链路
- 当 Tester 发现 Bug 时，自动返回 Engineer 修复，形成闭环

**工作流程：**
```
PM → Architect → Engineer → Tester
                              ↓
                    ┌────────┴────────┐
                    ↓                 ↓
               【有 Bug】          【通过】
                    ↓                 ↓
               Engineer            END
                    ↓
               Tester（再次测试）
```

**核心代码：**

```python
class AgentState(TypedDict):
    messages: Annotated[List[dict], operator.add]
    prd: str
    design: str
    code: str
    report: str
    loop_count: int

def should_continue(state: AgentState) -> str:
    loop = state.get("loop_count", 0)
    if loop >= 3:
        return END
    if "❌ Bug" in state.get("report", ""):
        return "engineer"
    return END

builder.add_conditional_edges("tester", should_continue, {
    "engineer": "engineer",
    END: END
})
```

**运行方式：**
```bash
python state.py
```

**输出示例：**
```
【PM】生成 PRD: [PRD] 基于需求'开发一个 Flappy Bird 游戏'...
【Architect】生成设计: [Design] 基于 PRD，技术栈：Python + Pygame...
【Engineer】生成代码: [Code] import pygame; class Bird...
【Tester】[Report] ❌ Bug 发现：碰撞检测逻辑错误...
【路由】发现 Bug，返回 Engineer 修复（第 1 轮）
【Engineer】生成代码: [Code] import pygame; class Bird...
【Tester】[Report] ✅ 所有测试通过。游戏可正常运行。
最终报告：✅ 所有测试通过。游戏可正常运行...
```

---

### 2. human.py - 人机协同基础示例

**功能说明：**
- 在 state.py 基础上增加 `human_gate` 节点
- 演示 LangGraph 的人机协同（Human-in-the-Loop）机制
- 支持在执行过程中暂停，等待人类审批后继续

**工作流程：**
```
PM → Architect → HumanGate
                     ↓
          ┌────────┴────────┐
          ↓                 ↓
     【已审批】           【未审批】
          ↓                 ↓
      Engineer            END（暂停）
          ↓
       Tester
```

**核心代码：**

```python
class AgentState(TypedDict):
    messages: Annotated[List[dict], operator.add]
    prd: str
    design: str
    code: str
    report: str
    loop_count: int
    human_approved: bool  # 审批标志

def human_gate(state: AgentState):
    if state.get("human_approved"):
        print("【HumanGate】✅ 已审批，放行")
        return {"messages": [{"role": "human_gate", "content": "审批通过"}]}

    print("【HumanGate】⏸ 暂停等待人类审批")
    return {}  # 不修改状态，直接让条件边判断

def route_human(state: AgentState):
    return "engineer" if state.get("human_approved") else END
```

**运行方式：**
```bash
python human.py
```

**输出示例：**
```
==================================================
【第 1 次运行】未审批状态
【PM】生成 PRD: [PRD] 开发 Flappy Bird
【Architect】生成设计: [Design] 基于PRD...
【HumanGate】⏸ 暂停等待人类审批

==================================================
【第 2 次运行】已审批
【HumanGate】✅ 已审批，放行
【Engineer】生成代码: [Code] 初始版...
【Tester】[Report] ✅ 通过
```

---

### 3. simulate.py - 完整的多阶段工作流

**功能说明：**
- 在 human.py 基础上增加状态持久化和人类意见整合
- 演示如何通过多次调用实现复杂的人机协同流程
- 支持在运行之间修改状态（如添加人类意见到 design）

**工作流程：**
```
第1次运行：PM → Architect → HumanGate → END（暂停）
              ↓ 保留 PRD 和 Design
第2次运行：PM（跳过）→ Architect（跳过）→ HumanGate → Engineer → Tester
```

**核心代码：**

```python
def pm_node(state: AgentState):
    if state.get("prd"):
        print("【PM】已有 PRD，跳过生成")
        return {"messages": [{"role": "pm", "content": state["prd"]}]}
    # ... 生成新 PRD

def engineer_node(state: AgentState):
    code = f"[Code] 初始版（基于设计：{design}）"
    if "人类意见" in design:
        code += " [已融入人类意见：双人模式]"
        print("【Engineer】✅ 检测到人类意见，已融入代码")
    return {"code": code, "loop_count": state.get("loop_count", 0) + 1, ...}
```

**运行方式：**
```bash
python simulate.py
```

**输出示例：**
```
==================================================
【第 1 次运行】未审批
【PM】生成 PRD: [PRD] Flappy Bird
【Architect】生成设计: [Design] 基于PRD：...
⏸ 暂停：请审批设计稿

--------
【第 2 次运行】已审批 + 人类意见
【PM】已有 PRD，跳过生成
【Architect】已有设计稿，跳过生成
【HumanGate】✅ 已审批，放行
【Engineer】✅ 检测到人类意见，已融入代码
【Engineer】生成代码: [Code] 初始版（基于设计：...）[已融入人类意见：双人模式]
【Tester】[Report] ✅ 通过

最终代码含'双人模式'：True
最终报告：[Report] ✅ 通过
```

## 环境配置

### 安装依赖

```bash
pip install langgraph
```

## 快速开始

### 1. 运行基础示例（state.py）

```bash
python state.py
```

**说明：**
- Tester 节点随机 50% 概率发现 Bug
- 发现 Bug 时自动触发修复循环
- 最多循环 3 次，防止无限循环

### 2. 运行人机协同基础示例（human.py）

```bash
python human.py
```

**说明：**
- 第 1 次运行会在 `human_gate` 处暂停
- 手动修改 `human_approved = True` 后重新运行即可继续

### 3. 运行完整工作流示例（simulate.py）

```bash
python simulate.py
```

**说明：**
- 第 1 次运行暂停后，第 2 次运行会保留之前的状态
- 可以在两次运行之间修改状态（如添加人类意见到 design）

## LangGraph 核心概念详解

### 状态管理

```python
class AgentState(TypedDict):
    messages: Annotated[List[dict], operator.add]
    prd: str
    design: str
    code: str
    report: str
    loop_count: int
    human_approved: bool
```

**关键点：**
- `TypedDict` 提供类型安全
- `Annotated[List[dict], operator.add]` 修饰器使 `messages` 字段自动追加新值
- 节点返回部分更新，LangGraph 自动合并到状态

### 节点定义

```python
def pm_node(state: AgentState):
    requirement = state["messages"][-1]["content"]
    prd = f"[PRD] 基于需求'{requirement}'，设计 Flappy Bird 变体游戏..."
    return {"prd": prd, "messages": [{"role": "pm", "content": prd}]}
```

### 条件路由

```python
def should_continue(state: AgentState) -> str:
    if state.get("loop_count", 0) >= 3:
        return END
    if "❌ Bug" in state.get("report", ""):
        return "engineer"
    return END

builder.add_conditional_edges(
    "tester",
    should_continue,
    {"engineer": "engineer", END: END}
)
```

### 图的构建流程

```python
builder = StateGraph(AgentState)

builder.add_node("pm", pm_node)
builder.add_node("architect", architect_node)
builder.add_node("engineer", engineer_node)
builder.add_node("tester", tester_node)

builder.set_entry_point("pm")
builder.add_edge("pm", "architect")
builder.add_edge("architect", "engineer")
builder.add_edge("engineer", "tester")
builder.add_conditional_edges("tester", should_continue, {
    "engineer": "engineer",
    END: END
})

graph = builder.compile()
```

## 人机协同模式

### 状态持久化

```python
def pm_node(state: AgentState):
    if state.get("prd"):
        print("【PM】已有 PRD，跳过生成")
        return {"messages": [{"role": "pm", "content": state["prd"]}]}
    # ... 生成新 PRD
```

### 人类意见整合

```python
def engineer_node(state: AgentState):
    code = f"[Code] 初始版（基于设计：{design}）"
    if "人类意见" in design:
        code += " [已融入人类意见：双人模式]"
    return {"code": code, ...}
```

### 多次调用实现暂停继续

```python
# 第 1 次运行
r1 = graph.invoke({
    "messages": [{"role": "user", "content": "Flappy Bird"}],
    "loop_count": 0,
    "human_approved": False
})

# 第 2 次运行：基于 r1 的状态继续
r2 = graph.invoke({
    "messages": r1["messages"],
    "prd": r1["prd"],
    "design": r1["design"] + " [人类意见：增加双人模式]",
    "loop_count": 0,
    "human_approved": True
})
```

## 常见问题

### Q1: state.py 和 human.py 的区别是什么？

**A:** state.py 展示基础的循环修复机制（tester 发现 bug → engineer 修复 → tester 重新测试）；human.py 在此基础上增加了 human_gate 节点，实现人机协同审批流程。

### Q2: simulate.py 相比 human.py 增加了什么？

**A:** simulate.py 增加了状态持久化能力，允许在多次调用之间保留和修改状态，实现更复杂的人机协同场景（如人类在审批时添加修改意见）。

### Q3: 如何防止无限循环？

**A:** 在条件路由函数中检查 `loop_count`：
```python
if state.get("loop_count", 0) >= 3:
    return END
```

### Q4: 为什么条件路由函数要返回字符串？

**A:** LangGraph 的条件边要求路由函数返回节点名称字符串（可哈希），而不是字典。状态更新应该在节点函数中完成。

## 学习路径

1. **基础阶段**：运行 `state.py`，理解 LangGraph 状态机和条件循环机制
2. **进阶阶段**：运行 `human.py`，理解人机协同审批流程
3. **完整阶段**：运行 `simulate.py`，理解状态持久化和多次调用
4. **实战阶段**：集成真实的 LLM API，实现智能代码生成和审批

## 扩展主题

### 添加新节点

```python
def reviewer_node(state: AgentState):
    code = state["code"]
    review = f"[Review] 代码审查：{code}"
    return {"review": review, "messages": [{"role": "reviewer", "content": review}]}

builder.add_node("reviewer", reviewer_node)
builder.add_edge("engineer", "reviewer")
builder.add_edge("reviewer", "tester")
```

### 多分支路由

```python
def route_after_test(state: AgentState):
    report = state.get("report", "")
    if "❌ Bug" in report:
        return "engineer"
    elif "warning" in report:
        return "reviewer"
    return END

builder.add_conditional_edges("tester", route_after_test, {
    "engineer": "engineer",
    "reviewer": "reviewer",
    END: END
})
```

### 可视化图结构

```python
graph.get_graph().draw_mermaid_png(output_file="agent_graph.png")
```

## 参考资料

- [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/)
- [LangGraph 核心概念](https://langchain-ai.github.io/langgraph/concepts/)
- [LangGraph 多智能体教程](https://langchain-ai.github.io/langgraph/tutorials/multi-agent/)
