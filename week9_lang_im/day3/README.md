# LangGraph 高级特性实战教程（Day 3）

## 概述

本目录包含 LangGraph 的高级特性实战教学材料：Time Travel 时间旅行、FastAPI 异步 API 和 SSE 流式推送。通过这些特性，可以实现复杂的人机协同工作流，支持在任意节点暂停、恢复、修改历史状态以及提供实时流式反馈。

**学习路径（递进关系）：**
```
Day 1 (simulate.py)  →  Day 2 (checkpointing + time travel + parallel)
     ↓                          ↓
 手动状态传递              自动状态持久化 + 并行执行

Day 2 基础              Day 3 实战
     ↓                          ↓
 Time Travel            FastAPI API + SSE 流式推送 + Time Travel
```

## 核心概念

### 什么是 Time Travel？

Time Travel 是 LangGraph 的历史回溯机制，允许：

- 查看完整的执行历史（所有快照）
- 回滚到任意历史状态
- 修改历史状态后重新执行
- 实现复杂的分支和合并场景

### 什么是 SSE 流式推送？

Server-Sent Events (SSE) 是一种服务器推送技术，允许服务器向浏览器客户端发送实时更新。相比 WebSocket，SSE 更简单且支持自动重连。

### 什么是 FastAPI 异步 API？

FastAPI 是一个现代 Python Web 框架，支持异步操作，可以高效处理并发请求。

## 文件说明

### 1. timetravel.py - Time Travel 时间旅行示例

**功能说明：**
- 演示 LangGraph 的 Time Travel 特性
- 查看完整执行历史（所有快照）
- 回滚到任意历史状态并重新执行
- 修改历史状态后继续执行
- 使用 `MemorySaver` 替代 SQLite 实现内存持久化

**工作流程：**
```
PM → Architect → HumanGate(interrupt) → Engineer → Tester
                              ↑                    ↓
                              ←────── (Time Travel) ←
```

**核心代码：**

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

# 编译图时注入 checkpointer
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# 获取所有历史快照
history = []
for state in graph.get_state_history(config):
    history.append(state)

# 回滚到指定状态
target = history[5]  # 回滚到 architect 节点
new_values = dict(target.values)
new_values["design"] += " [回滚修改：增加道具系统]"

# 更新状态（as_node 指定从哪个节点重新执行）
graph.update_state(config, new_values, as_node="architect")

# 使用 Command(resume=...) 继续执行
for event in graph.stream(Command(resume="修改_增加道具系统"), config):
    print(f"回滚后 Event: {event}")
```

**运行方式：**
```bash
python timetravel.py
```

**输出示例：**
```
============================================================
【第1次】正常执行，将在 human_gate 暂停
============================================================
【PM】生成 PRD: [PRD] Flappy Bird...
【Architect】生成设计: [Design] 基于PRD：...
⏸ 捕获 interrupt: ...

============================================================
【时间旅行】查看执行历史
============================================================
历史快照数: 9

--- 快照 [5] ---
  next: ('human_gate',)
  design: [Design] 基于PRD：...
  human_approved: False

============================================================
【回滚】回到 architect 完成后的状态
============================================================
已修改 design: [Design] 基于PRD：... [回滚修改：增加道具系统]...

【干预】Command(resume='修改_增加道具系统')
Event: ['engineer']
  → Engineer code: [Code] 初始版（基于设计：... [回滚修改：增加道具系统]）...
```

---

### 2. async.py - FastAPI 异步 API 示例

**功能说明：**
- 演示 FastAPI 异步 API 的使用方法
- 使用 `AsyncSqliteSaver` 实现异步状态持久化
- 支持通过 `thread_id` 恢复对话
- 提供 `/chat` 和 `/state/{thread_id}` 两个接口

**核心代码：**

```python
from fastapi import FastAPI
from pydantic import BaseModel
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

app = FastAPI(title="Multi-Agent Software Team API")

class ChatRequest(BaseModel):
    query: str
    thread_id: str = "default"
    human_decision: str | None = None

@app.post("/chat")
async def chat(req: ChatRequest):
    config = {"configurable": {"thread_id": req.thread_id}}

    async with AsyncSqliteSaver.from_conn_string("api_checkpoints.sqlite") as memory:
        graph = builder.compile(checkpointer=memory)

        if req.human_decision:
            result = await graph.ainvoke(
                Command(resume=req.human_decision),
                config=config
            )
        else:
            result = await graph.ainvoke(
                {"messages": [{"role": "user", "content": req.query}], ...},
                config=config
            )

    return {"thread_id": req.thread_id, "answer": result.get("code"), ...}
```

**运行方式：**
```bash
python async.py
```

**API 接口：**

| 接口 | 方法 | 说明 |
|------|------|------|
| `/chat` | POST | 发送消息，返回执行结果 |
| `/state/{thread_id}` | GET | 查询线程当前状态 |

**请求示例：**
```bash
# 首次请求
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Flappy Bird", "thread_id": "job-001"}'

# 恢复执行（审批）
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"thread_id": "job-001", "human_decision": "通过"}'
```

---

### 3. see.py - SSE 流式推送 + HTML 前端演示

**功能说明：**
- 演示 SSE (Server-Sent Events) 流式推送
- 提供实时节点执行状态推送
- 内置 HTML 前端测试页面
- 支持中断恢复 (`Command(resume=...)`)

**核心代码：**

```python
from fastapi import Query
from sse_starlette.sse import EventSourceResponse

@app.get("/chat/stream")
async def chat_stream(
    query: str | None = Query(None, description="用户需求"),
    thread_id: str = Query("default", description="会话ID"),
    human_decision: str | None = Query(None, description="人类审批决策")
):
    async def event_generator():
        async with AsyncSqliteSaver.from_conn_string("api_checkpoints.sqlite") as memory:
            graph = builder.compile(checkpointer=memory)

            async for event in graph.astream(inputs, config):
                for node_name, node_output in event.items():
                    if node_name == "__interrupt__":
                        yield {"event": "interrupt", "data": json.dumps({...})}
                    else:
                        yield {"event": "node", "data": json.dumps({...})}

    return EventSourceResponse(event_generator())
```

**运行方式：**
```bash
python see.py
```

**访问方式：**
1. 打开浏览器访问 `http://localhost:8000/`
2. 在前端页面输入需求并点击「启动」
3. 等待中断后，点击「审批通过」或「修改_增加双人模式」

**前端页面功能：**
- 实时显示节点执行状态
- 支持中断恢复
- 展示聚合评审结果

---

## Time Travel 核心概念

### 获取执行历史

```python
# 获取所有历史快照（从新到旧）
history = list(graph.get_state_history(config))

# 快照包含：values（状态）、next（下一个节点）、config（配置）
for i, state in enumerate(history):
    print(f"快照 [{i}]: next={state.next}")
```

### 回滚到指定状态

```python
# 选择要回滚的快照
target = history[5]

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

## FastAPI + SSE 核心概念

### 异步 Checkpointer

```python
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

async with AsyncSqliteSaver.from_conn_string("checkpoints.sqlite") as memory:
    graph = builder.compile(checkpointer=memory)
    result = await graph.ainvoke(inputs, config=config)
```

### SSE 事件格式

```python
# 普通节点完成事件
yield {
    "event": "node",
    "data": json.dumps({"node": node_name, "status": "completed"})
}

# 中断事件
yield {
    "event": "interrupt",
    "data": json.dumps({"status": "waiting_approval", "design": "..."})
}

# 最终状态事件
yield {
    "event": "final",
    "data": json.dumps({"status": "completed", "code": "..."})
}
```

### 前端 EventSource 使用

```javascript
const es = new EventSource(url);

es.onmessage = (e) => {
    const data = JSON.parse(e.data);
    if (data.status === 'waiting_approval') {
        console.log('等待审批:', data.design);
    } else if (data.node) {
        console.log('节点完成:', data.node);
    }
};

es.onerror = () => {
    es.close();
};
```

## 常见问题

### Q1: Time Travel 和普通恢复执行的区别？

**A:** 普通恢复执行只能从上次中断点继续；Time Travel 可以回滚到任意历史状态，修改后再继续，实现真正的「时间旅行」。

### Q2: AsyncSqliteSaver 和 SqliteSaver 的区别？

**A:** AsyncSqliteSaver 支持异步操作，适合 FastAPI 等异步框架；SqliteSaver 是同步版本，适合同步环境。

### Q3: SSE 和 WebSocket 的区别？

**A:** SSE 是单向的（服务器→客户端），简单且支持自动重连；WebSocket 是双向的，更复杂。实时状态推送用 SSE 足够。

### Q4: 如何选择 Checkpoint 存储方式？

**A:** 开发调试用 `MemorySaver`（速度快）；生产环境用 `SqliteSaver`（持久化）；FastAPI 异步环境用 `AsyncSqliteSaver`。

## 环境配置

### 安装依赖

```bash
pip install langgraph langgraph-checkpoint fastapi uvicorn sse-starlette
```

## 学习路径

1. **基础阶段**：运行 `timetravel.py`，理解 Time Travel 机制
2. **API 阶段**：运行 `async.py`，理解 FastAPI 异步 API
3. **流式阶段**：运行 `see.py`，理解 SSE 流式推送
4. **实战阶段**：组合使用所有特性实现复杂工作流

## 参考资料

- [LangGraph Checkpoint 文档](https://langchain-ai.github.io/langgraph/concepts/checkpointing/)
- [LangGraph Time Travel 文档](https://langchain-ai.github.io/langgraph/how-tos/time-travel/)
- [FastAPI 文档](https://fastapi.tiangolo.com/)
- [SSE 文档](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)
