# step2.py
import json
import os
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from sse_starlette.sse import EventSourceResponse

from timetravel import builder

app = FastAPI(title="Multi-Agent SSE API")

# 全局编译一次（不要每次请求重新编译）
# 注意：AsyncSqliteSaver 不能全局保持连接，需要在请求内创建
_graph_builder = builder


@app.get("/chat/stream")
async def chat_stream(
    query: str | None = Query(None, description="用户需求，首次运行必填，resume时无需填写"),
    thread_id: str = Query("default", description="会话ID"),
    human_decision: str | None = Query(None, description="人类审批决策，用于resume")
):
    config = {"configurable": {"thread_id": thread_id}}

    async def event_generator():
        async with AsyncSqliteSaver.from_conn_string("api_checkpoints.sqlite") as memory:
            graph = _graph_builder.compile(checkpointer=memory)

            if human_decision:
                # Resume 模式：不需要 query
                inputs = Command(resume=human_decision)
            elif query:
                # 首次运行：必须有 query
                inputs = {
                    "messages": [{"role": "user", "content": query}],
                    "loop_count": 0,
                    "human_approved": False,
                }
            else:
                # 既没有 query 也没有 human_decision
                yield {
                    "event": "error",
                    "data": json.dumps({"error": "请提供 query（首次运行）或 human_decision（resume）"}),
                }
                return

            async for event in graph.astream(inputs, config):
                for node_name, node_output in event.items():
                    if node_name == "__interrupt__":
                        interrupt_data = node_output[0].value if isinstance(node_output, tuple) else node_output
                        yield {
                            "event": "interrupt",
                            "data": json.dumps({
                                "node": "human_gate",
                                "status": "waiting_approval",
                                "design": interrupt_data.get("design", "")[:200],
                                "options": interrupt_data.get("options", []),
                            }, ensure_ascii=False),
                        }
                    else:
                        yield {
                            "event": "node",
                            "data": json.dumps({
                                "node": node_name,
                                "status": "completed",
                                "keys": list(node_output.keys()),
                            }, ensure_ascii=False),
                        }

            snapshot = await graph.aget_state(config)
            yield {
                "event": "final",
                "data": json.dumps({
                    "next": str(snapshot.next),
                    "code": snapshot.values.get("code", "")[:100],
                    "report": snapshot.values.get("report", "")[:100],
                }, ensure_ascii=False),
            }

    return EventSourceResponse(event_generator())


@app.get("/")
async def index():
    """前端测试页面：使用原生 EventSource（GET）"""
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
    <title>Multi-Agent SSE Demo</title>
    <style>
        body { font-family: monospace; padding: 20px; }
        #log { border: 1px solid #ccc; padding: 10px; height: 400px; overflow-y: auto; background: #f5f5f5; }
        .event-node { color: blue; }
        .event-interrupt { color: orange; font-weight: bold; }
        .event-final { color: green; font-weight: bold; }
        button { padding: 10px 20px; margin: 5px; }
    </style>
</head>
<body>
    <h2>Multi-Agent 实时流式演示</h2>
    <input id="query" value="Flappy Bird" placeholder="需求描述" style="width:300px;">
    <input id="thread" value="job-sse-001" placeholder="thread_id">
    <br><br>
    <button onclick="start()">1. 启动（首次运行）</button>
    <button onclick="approve()">2. 审批通过</button>
    <button onclick="modify()">3. 修改_增加双人模式</button>
    <div id="log"></div>

    <script>
        const log = document.getElementById('log');
        let currentThread = document.getElementById('thread').value;

        function append(text, cls) {
            const div = document.createElement('div');
            div.className = cls || '';
            div.textContent = new Date().toLocaleTimeString() + ' | ' + text;
            log.appendChild(div);
            log.scrollTop = log.scrollHeight;
        }

        function connect(url, threadId) {
            currentThread = threadId;
            const es = new EventSource(url);
            
            es.onmessage = (e) => {
                const data = JSON.parse(e.data);
                if (data.status === 'waiting_approval') {
                    append('⏸ INTERRUPT: ' + data.design, 'event-interrupt');
                    append('选项: ' + data.options.join(', '), 'event-interrupt');
                } else if (data.node) {
                    append('✓ Node: ' + data.node + ' | keys: ' + data.keys?.join(','), 'event-node');
                } else {
                    append('🏁 FINAL: ' + JSON.stringify(data), 'event-final');
                }
            };
            
            es.onerror = (e) => {
                append('Connection closed or error', 'event-final');
                es.close();
            };
        }

        function start() {
            const query = encodeURIComponent(document.getElementById('query').value);
            const thread = encodeURIComponent(document.getElementById('thread').value);
            log.innerHTML = '';
            append('Connecting... ' + thread);
            connect(`/chat/stream?query=${query}&thread_id=${thread}`, thread);
        }

        function approve() {
            const thread = encodeURIComponent(currentThread);
            append('Resuming with: 通过');
            connect(`/chat/stream?thread_id=${thread}&human_decision=${encodeURIComponent('通过')}`, currentThread);
        }

        function modify() {
            const thread = encodeURIComponent(currentThread);
            append('Resuming with: 修改_增加双人模式');
            connect(`/chat/stream?thread_id=${thread}&human_decision=${encodeURIComponent('修改_增加双人模式')}`, currentThread);
        }
    </script>
</body>
</html>
    """)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
