# human.py
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
    human_approved: bool  # ← 新增：审批标志

# ========== 节点（复用之前的，略改）==========
def pm_node(state: AgentState):
    req = state["messages"][-1]["content"] if state["messages"] else "开发游戏"
    return {"prd": f"[PRD] {req}", "messages": [{"role": "pm", "content": f"[PRD] {req}"}]}

def architect_node(state: AgentState):
    return {"design": f"[Design] 基于PRD：{state['prd'][:30]}...", 
            "messages": [{"role": "architect", "content": f"[Design] 基于PRD"}]}

def human_gate(state: AgentState):
    """人类审批节点"""
    if state.get("human_approved"):
        print("【HumanGate】✅ 已审批，放行")
        return {"messages": [{"role": "human_gate", "content": "审批通过"}]}
    
    print("\n" + "="*50)
    print("【HumanGate】⏸ 暂停等待人类审批")
    print(f"设计稿：{state['design']}")
    print("请在外部修改 state['human_approved'] = True 后重新运行")
    print("="*50 + "\n")
    return {}  # 不修改任何状态，直接让条件边判断

def engineer_node(state: AgentState):
    report = state.get("report", "")
    if "❌ Bug" in report:
        code = f"[Code] 修复版（针对：{report[:30]}...）"
    else:
        code = f"[Code] 初始版，基于设计：{state['design'][:30]}"
    return {"code": code, "loop_count": state.get("loop_count", 0) + 1,
            "messages": [{"role": "engineer", "content": code}]}

def tester_node(state: AgentState):
    # 确定性：第1轮有Bug，第2轮修复
    loop = state.get("loop_count", 0)
    if loop == 1:
        report = "[Report] ❌ Bug：碰撞检测错误，需修复"
    else:
        report = "[Report] ✅ 通过"
    return {"report": report, "messages": [{"role": "tester", "content": report}]}

# ========== 条件边 ==========
def route_human(state: AgentState):
    if state.get("human_approved"):
        return "engineer"
    return END  # 未审批 → 暂停

def route_tester(state: AgentState):
    report = state.get("report", "")
    loop = state.get("loop_count", 0)
    if loop >= 3:
        return END
    if "❌ Bug" in report:
        return "engineer"
    return END

# ========== 建图 ==========
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
    with open("human_graph.png", "wb") as f:
        f.write(png_bytes)
    print("✅ 已生成可视化图：human_graph.png")
except Exception as e:
    print(f"⚠️ 无法生成可视化图：{e}")

# ========== 测试：人机协同 ==========
if __name__ == "__main__":
    # 第 1 次运行：未审批，会在 human_gate 暂停
    print("="*50)
    print("【第 1 次运行】未审批状态")
    result = graph.invoke({
        "messages": [{"role": "user", "content": "开发 Flappy Bird"}],
        "loop_count": 0,
        "human_approved": False  # ← 初始未审批
    })
    print(f"状态：{result.get('next_agent', 'END')}")
    print(f"设计稿：{result['design']}")
    
    # 模拟人类审批：修改 state 后重新运行
    print("\n" + "="*50)
    print("【模拟人类审批】修改 human_approved = True")
    print("="*50)
    
    # 第 2 次运行：已审批，完整执行
    result2 = graph.invoke({
        "messages": result["messages"],  # 保留历史
        "prd": result["prd"],
        "design": result["design"],
        "loop_count": 0,
        "human_approved": True  # ← 人类已审批
    })
    print(f"\n最终代码：{result2['code'][:50]}...")
    print(f"最终报告：{result2['report']}")
