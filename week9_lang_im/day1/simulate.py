# simulate.py
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
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
    """PM：如果已有 PRD（第2次运行），直接透传"""
    if state.get("prd"):
        print("【PM】已有 PRD，跳过生成")
        return {"messages": [{"role": "pm", "content": state["prd"]}]}
    
    req = state["messages"][-1]["content"] if state["messages"] else "开发游戏"
    prd = f"[PRD] {req}"
    print(f"【PM】生成 PRD: {prd[:50]}...")
    return {"prd": prd, "messages": [{"role": "pm", "content": prd}]}

def architect_node(state: AgentState):
    """Architect：如果已有 design（第2次运行），直接透传"""
    if state.get("design"):
        print(f"【Architect】已有设计稿，跳过生成: {state['design'][:50]}...")
        return {"messages": [{"role": "architect", "content": state["design"]}]}
    
    prd = state.get("prd", "")
    design = f"[Design] 基于PRD：{prd}"
    print(f"【Architect】生成设计: {design[:50]}...")
    return {"design": design, "messages": [{"role": "architect", "content": design}]}

def human_gate(state: AgentState):
    if state.get("human_approved"):
        print("【HumanGate】✅ 已审批，放行")
        return {"messages": [{"role": "human_gate", "content": "审批通过"}]}
    
    print("\n⏸ 暂停：请审批设计稿")
    print(f"设计稿：{state['design']}")
    return {}

def engineer_node(state: AgentState):
    report = state.get("report", "")
    design = state.get("design", "")
    
    if "❌ Bug" in report:
        code = f"[Code] 修复版（基于设计：{design}，针对：{report}）"
    else:
        code = f"[Code] 初始版（基于设计：{design}）"
    
    if "人类意见" in design:
        code += " [已融入人类意见：双人模式]"
        print("【Engineer】✅ 检测到人类意见，已融入代码")
    
    return {
        "code": code, 
        "loop_count": state.get("loop_count", 0) + 1,
        "messages": [{"role": "engineer", "content": code}]
    }

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

# 建图
builder = StateGraph(AgentState)
builder.add_node("pm", pm_node)
builder.add_node("architect", architect_node)
builder.add_node("human_gate", human_gate)
builder.add_node("engineer", engineer_node)
builder.add_node("tester", tester_node)

builder.set_entry_point("pm")
builder.add_edge("pm", "architect")
builder.add_edge("architect", "human_gate")
builder.add_conditional_edges("human_gate", route_human, {"engineer": "engineer", END: END})
builder.add_edge("engineer", "tester")
builder.add_conditional_edges("tester", route_tester, {"engineer": "engineer", END: END})

graph = builder.compile()

try:
    png_bytes = graph.get_graph().draw_mermaid_png()
    with open("simulate_graph.png", "wb") as f:
        f.write(png_bytes)
    print("✅ 已生成可视化图：simulate_graph.png")
except Exception as e:
    print(f"⚠️ 无法生成可视化图：{e}")

if __name__ == "__main__":
    # 第 1 次：未审批
    r1 = graph.invoke({
        "messages": [{"role": "user", "content": "Flappy Bird"}], 
        "loop_count": 0, 
        "human_approved": False
    })
    print(f"\n暂停状态：code={'有' if r1.get('code') else '无'}")
    
    print("--------")
    # 第 2 次：审批 + 人类修改 design
    r2 = graph.invoke({
        "messages": r1["messages"],
        "prd": r1["prd"],
        "design": r1["design"] + " [人类意见：增加双人模式]",
        "loop_count": 0,
        "human_approved": True
    })
    
    print(f"\n最终代码：{r2['code']}")
    print(f"含'双人模式'：{'双人模式' in r2['code']}")
    print(f"含'人类意见'：{'人类意见' in r2['code']}")
    print(f"最终报告：{r2['report']}")
