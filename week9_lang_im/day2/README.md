# LangGraph Checkpointing 与时间旅行教程（Day 2）

## 概述

本目录包含 LangGraph 的高级特性教学材料：Checkpointer 状态持久化和 Time Travel 时间旅行。通过这些特性，可以实现复杂的人机协同工作流，支持在任意节点暂停、恢复和修改历史状态。

**学习路径（递进关系）：**
```
Day 1 (simulate.py)  →  Day 2 (checkpointing + time travel)
     ↓                          ↓
 手动状态传递              自动状态持久化
```

## 核心概念

### 什么是 Checkpointing？

Checkpointing 是 LangGraph 的状态持久化机制，允许将图执行过程中的状态保存到外部存储（SQLite）。配合 `configurable thread_id`，可以实现：

- 中断后恢复执行
- 多轮对话状态保持
- 并发执行不同分支

**核心优势：**
- 自动状态持久化
- 支持中断和恢复
- 线程级隔离
- 外部存储（不丢失）

### 什么是 Time Travel？

Time Travel 是 LangGraph 的历史回溯机制，允许：

- 查看完整的执行历史（所有快照）
- 回滚到任意历史状态
- 修改历史状态后重新执行
- 实现复杂的分支和合并场景

## 文件说明

### 1. checkpointer.py - Checkpoint 基础示例

**功能说明：**
- 演示 LangGraph Checkpointer 的基本用法
- 使用 SQLite 保存执行状态
- 支持通过 `thread_id` 恢复对话

**核心代码：**

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# 1. 创建 Checkpointer
memory = SqliteSaver.from_conn_string("checkpoints.sqlite")

# 2. 编译图时注入
graph = builder.compile(checkpointer=memory)

# 3. 运行（带 config）
config = {"configurable": {"thread_id": "job-001"}}
result = graph.invoke(initial_state, config=config)
```

**运行方式：**
```bash
python checkpointer.py
```

**输出示例：**
```
==================================================
【第 1 次运行】未审批状态
==================================================
【PM】生成 PRD: [PRD] Flappy Bird...
【Architect】生成设计: [Design] 基于PRD：...

第1次状态：code=无, design=[Design] 基于PRD：[PRD] Flappy Bird...

==================================================
【第 2 次运行】从 checkpoint 恢复 + 审批
==================================================
【PM】已有 PRD，跳过生成
【Architect】已有设计稿，跳过生成
【HumanGate】✅ 已审批，放行
【Engineer】✅ 检测到人类意见，已融入代码

最终代码：[Code] 初始版（基于设计：...）[已融入人类意见：双人模式]
最终报告：[Report] ✅ 通过

✅ Checkpoint 演示完成！
```

---

### 2. interrupt.py - Interrupt 中断示例

**功能说明：**
- 演示 `graph.interrupt()` 的使用方法
- 配合 Checkpointer 实现更精细的中断控制
- 在特定节点自动暂停，等待外部干预

**核心代码：**

```python
def human_gate(state: AgentState):
    if not state.get("human_approved"):
        print("⏸ 暂停：等待人类审批...")
        return {}  # 返回空状态，不修改任何字段

    # 检查是否被中断
    if graph.get_state(config).tasks:
        interrupted = graph.get_state(config).tasks[-1].interrupted
        if interrupted:
            print("⏸ 检测到中断，等待外部处理...")

    return {"messages": [{"role": "human_gate", "content": "审批通过"}]}
```

**运行方式：**
```bash
python interrupt.py
```

---

### 3. TimeTravel.py - 时间旅行示例

**功能说明：**
- 演示 LangGraph 的 Time Travel 特性
- 查看完整执行历史（所有快照）
- 回滚到任意历史状态并重新执行
- 修改历史状态后继续执行

**核心代码：**

```python
# 获取所有历史快照
history = []
for state in graph.get_state_history(config):
    history.append(state)

print(f"历史记录数: {len(history)}")

# 回滚到指定状态
target = history[5]  # 回滚到 architect 节点
new_values = dict(target.values)
new_values["design"] += " [回滚修改：增加道具系统]"
new_values["human_approved"] = False

# 更新状态并重新执行
graph.update_state(config, new_values, as_node="architect")

# 使用 Command(resume=...) 继续执行
for event in graph.stream(Command(resume="修改_增加道具系统"), config):
    print(f"回滚后 Event: {event}")
```

**运行方式：**
```bash
python TimeTravel.py
```

**输出示例：**
```
============================================================
【时间旅行】查看完整执行历史
============================================================
历史记录数: 9

--- 快照 [0] ---
  next: ()
  design: [Design] 基于PRD：[PRD] Flappy Bird [人类意见：增...
  code: 有
  human_approved: True

--- 快照 [5] ---
  next: ('human_gate',)
  design: [Design] 基于PRD：[PRD] Flappy Bird...
  code: 无
  human_approved: False

============================================================
【回滚】回到 ('architect',) 状态，修改设计后重新审批
============================================================
【Architect】生成设计: [Design] 基于PRD：[PRD] Flappy Bird [回滚修改：增加道具系统]...
```

## 环境配置

### 安装依赖

```bash
pip install langgraph langgraph-checkpoint
```

## 快速开始

### 1. 运行 Checkpoint 基础示例

```bash
python checkpointer.py
```

**说明：**
- 第 1 次运行在 human_gate 暂停，状态保存到 SQLite
- 第 2 次运行从 checkpoint 恢复，继续执行

### 2. 运行 Interrupt 示例

```bash
python interrupt.py
```

**说明：**
- 演示如何在特定节点中断执行
- 配合 Checkpointer 实现状态保持

### 3. 运行 Time Travel 示例

```bash
python TimeTravel.py
```

**说明：**
- 查看完整执行历史
- 选择任意快照回滚
- 修改历史状态后重新执行

## Checkpointer 核心概念

### SQLite Checkpoint

```python
from langgraph.checkpoint.sqlite import SqliteSaver

memory = SqliteSaver.from_conn_string("checkpoints.sqlite")
graph = builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "唯一标识符"}}
```

### Thread ID 的作用

```python
# 不同 thread_id 是完全独立的执行线程
config1 = {"configurable": {"thread_id": "job-001"}}
config2 = {"configurable": {"thread_id": "job-002"}}

# 同一个 thread_id 会从上次中断处继续
result = graph.invoke(state, config=config1)  # 第一次执行
result = graph.invoke(state, config=config1)  # 从 checkpoint 恢复
```

### 检查点配置

```python
# Memory Checkpoint（仅内存，断电丢失）
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# SQLite Checkpoint（持久化存储）
from langgraph.checkpoint.sqlite import SqliteSaver
memory = SqliteSaver.from_conn_string("checkpoints.sqlite")
graph = builder.compile(checkpointer=memory)
```

## Time Travel 核心概念

### 获取执行历史

```python
# 获取所有历史快照（从新到旧）
history = []
for state in graph.get_state_history(config):
    history.append(state)

# 快照包含：values（状态）、next（下一个节点）、config（配置）
for i, state in enumerate(history):
    print(f"快照 [{i}]: next={state.next}, values={state.values}")
```

### 回滚到指定状态

```python
# 选择要回滚的快照
target = history[5]  # 回滚到 architect 节点

# 提取状态值并修改
new_values = dict(target.values)
new_values["design"] += " [新功能]"

# 更新状态（as_node 指定从哪个节点重新执行）
graph.update_state(config, new_values, as_node="architect")
```

### 恢复执行

```python
# 使用 Command(resume=...) 继续执行
for event in graph.stream(Command(resume="修改内容"), config):
    print(f"Event: {event}")
```

## 常见问题

### Q1: Checkpointer 和手动状态传递的区别？

**A:** 手动状态传递（Day 1）需要每次调用时手动传入上次结果；Checkpointer 自动保存状态，只需传入 `thread_id` 即可自动恢复。

### Q2: 如何选择 Checkpoint 存储方式？

**A:** 开发调试用 `MemorySaver`（速度快）；生产环境用 `SqliteSaver`（持久化）。

### Q3: Time Travel 可以回滚多远？

**A:** 理论上可以回滚到任意历史快照，包括起点。但需要注意状态依赖关系。

### Q4: `as_node` 参数的作用是什么？

**A:** 指定从哪个节点重新开始执行。例如 `as_node="architect"` 表示从 architect 节点之后继续执行。

## 学习路径

1. **基础阶段**：运行 `checkpointer.py`，理解 Checkpoint 机制
2. **进阶阶段**：运行 `interrupt.py`，理解中断控制
3. **高级阶段**：运行 `TimeTravel.py`，理解时间旅行
4. **实战阶段**：组合使用 Checkpoint + Interrupt + Time Travel 实现复杂工作流

## 扩展主题

### 自定义 Checkpoint

```python
from langgraph.checkpoint.base import BaseCheckpointSaver

class CustomCheckpoint(BaseCheckpointSaver):
    def get(self, config):
        # 自定义获取逻辑
        pass

    def put(self, config, state):
        # 自定义保存逻辑
        pass
```

### 条件中断

```python
def should_interrupt(state: AgentState) -> bool:
    return state.get("loop_count", 0) >= 2

# 在图中使用条件中断
if should_interrupt(state):
    graph.interrupt()
```

### 可视化图结构

```python
try:
    png_bytes = graph.get_graph().draw_mermaid_png()
    with open("checkpointer_graph.png", "wb") as f:
        f.write(png_bytes)
    print("✅ 已生成可视化图")
except Exception as e:
    print(f"⚠️ 无法生成：{e}")
```

## 参考资料

- [LangGraph Checkpoint 文档](https://langchain-ai.github.io/langgraph/concepts/checkpointing/)
- [LangGraph Time Travel 文档](https://langchain-ai.github.io/langgraph/how-tos/time-travel/)
- [LangGraph Interrupt 文档](https://langchain-ai.github.io/langgraph/how-tos/interruption/)
