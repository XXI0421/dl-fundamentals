# checkpointer.py
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
# pip install langgraph-checkpoint-sqlite
from langgraph.checkpoint.sqlite import SqliteSaver
import operator

class AgentState(TypedDict):
    messages: Annotated[List[dict], operator.add]
    prd: str
    design: str
    code: str
    report: str
    loop_count: int
    human_approved: bool

def pm_node(state: AgentState):
    req = state["messages"][-1]["content"] if state["messages"] else "开发游戏"
    prd = f"[PRD] {req}"
    print(f"【PM】生成 PRD: {prd[:50]}...")
    return {"prd": prd, "messages": [{"role": "pm", "content": prd}]}

def architect_node(state: AgentState):
    prd = state.get("prd", "")
    design = f"[Design] 基于PRD：{prd}"
    print(f"【Architect】生成设计: {design[:50]}...")
    return {"design": design, "messages": [{"role": "architect", "content": design}]}

def human_gate(state: AgentState):
    if state.get("human_approved"):
        print("【HumanGate】✅ 已审批，放行")
        return {"messages": [{"role": "human_gate", "content": "审批通过"}]}
    print("⏸ 暂停：请审批设计稿")
    print(f"设计稿：{state['design']}")
    return {}

def engineer_node(state: AgentState):
    report = state.get("report", "")
    design = state.get("design", "")
    if "❌ Bug" in report:
        code = f"[Code] 修复版（针对：{report[:30]}...）"
    else:
        code = f"[Code] 初始版（基于设计：{design[:30]}...）"
    if "人类意见" in design:
        code += " [已融入人类意见：双人模式]"
        print("【Engineer】✅ 检测到人类意见，已融入代码")
    return {"code": code, "loop_count": state.get("loop_count", 0) + 1,
            "messages": [{"role": "engineer", "content": code}]}

def tester_node(state: AgentState):
    loop = state.get("loop_count", 0)
    if loop == 1:
        report = "[Report] ❌ Bug：碰撞检测错误，需修复"
    else:
        report = "[Report] ✅ 通过"
    return {"report": report, "messages": [{"role": "tester", "content": report}]}

def route_human(state: AgentState):
    return "engineer" if state.get("human_approved") else END

def route_tester(state: AgentState):
    loop = state.get("loop_count", 0)
    report = state.get("report", "")
    if loop >= 3:
        return END
    return "engineer" if "❌ Bug" in report else END

builder = StateGraph(AgentState)
builder.add_node("pm", pm_node)
builder.add_node("architect", architect_node)
builder.add_node("human_gate", human_gate)
builder.add_node("engineer", engineer_node)
builder.add_node("tester", tester_node)

builder.add_edge(START, "pm")
builder.add_edge("pm", "architect")
builder.add_edge("architect", "human_gate")
builder.add_conditional_edges("human_gate", route_human, {"engineer": "engineer", END: END})
builder.add_edge("engineer", "tester")
builder.add_conditional_edges("tester", route_tester, {"engineer": "engineer", END: END})

# SqliteSaver 必须使用上下文管理器
with SqliteSaver.from_conn_string("checkpoints.sqlite") as memory:
    graph = builder.compile(checkpointer=memory)

    try:
        png_bytes = graph.get_graph().draw_mermaid_png()
        with open("checkpointer_graph.png", "wb") as f:
            f.write(png_bytes)
        print("✅ 已生成可视化图：checkpointer_graph.png")
    except Exception as e:
        print(f"⚠️ 无法生成可视化图：{e}")

    config = {"configurable": {"thread_id": "job-001"}}

    # Step 1: 首次运行
    print("【Step 1】首次运行...")
    graph.invoke({
        "messages": [{"role": "user", "content": "Flappy Bird"}],
        "loop_count": 0,
        "human_approved": False
    }, config=config)
    
    snapshot = graph.get_state(config)
    print(f"停在：{snapshot.next}") 
    # 预期输出: () 
    # () 表示图在 human_gate 处认为"已结束"（因为 route_human 返回了 END）

    # Step 2: 人类审批
    print("\n【Step 2】人类修改 design 并审批...")
    graph.update_state(config, {
        "design": snapshot.values["design"] + " [人类意见：增加双人模式]",
        "human_approved": True
    })
    # 但 update_state 修改 human_approved=True 后，
    # stream(None) 会重新评估条件边，
    # 发现现在可以走 engineer，于是继续执行。 

    # Step 3: 从断点恢复
    print("\n【Step 3】从断点恢复执行...")
    for event in graph.stream(None, config):  # ← None 是关键
        print(event)
