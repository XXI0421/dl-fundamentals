import os
from fastapi import FastAPI
from pydantic import BaseModel
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command


# 复用 Day 2 的 builder
from timetravel import builder

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
                {
                    "messages": [{"role": "user", "content": req.query}],
                    "loop_count": 0,
                    "human_approved": False
                },
                config=config
            )
        
        # 修复：使用异步方法 aget_state 替代同步 get_state
        snapshot = await graph.aget_state(config)
    
    return {
        "thread_id": req.thread_id,
        "answer": result.get("code") or result.get("report") or "无输出",
        "design": result.get("design", ""),
        "next": snapshot.next
    }

@app.get("/state/{thread_id}")
async def get_state(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    
    async with AsyncSqliteSaver.from_conn_string("api_checkpoints.sqlite") as memory:
        graph = builder.compile(checkpointer=memory)
        snapshot = await graph.aget_state(config)
    
    return {
        "thread_id": thread_id,
        "next": snapshot.next,
        "design": snapshot.values.get("design", "")[:100],
        "code": snapshot.values.get("code", "")[:100]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
