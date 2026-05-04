# TimeTravel.py
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
import operator
from langgraph.types import interrupt, Command

# ========== 1. State 定义 ==========
class AgentState(TypedDict):
    messages: Annotated[List[dict], operator.add]
    prd: str
    design: str
    code: str
    report: str
    loop_count: int
    human_approved: bool

# ========== 2. 节点实现 ==========
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
    """
    人机协同节点：使用 interrupt 原生暂停。
    外部通过 Command(resume=决策) 恢复，interrupt() 返回决策内容。
    """
    # 如果已审批（如从 checkpoint 恢复后），直接放行
    if state.get("human_approved"):
        print("【HumanGate】✅ 已审批，放行")
        return {"messages": [{"role": "human_gate", "content": "审批通过"}]}
    
    print("【HumanGate】⏸ 触发 interrupt，等待人类审批...")
    
    # interrupt 会暂停图执行，把 payload 抛给外部
    # 当外部调用 Command(resume=决策) 时，interrupt() 返回该决策
    human_response = interrupt({
        "stage": "design_review",
        "design": state.get("design", ""),
        "question": "请审批设计稿",
        "options": ["通过", "修改_增加双人模式", "拒绝"]
    })
    
    print(f"【HumanGate】收到人类决策: {human_response}")
    
    if human_response == "通过":
        return {
            "human_approved": True,
            "messages": [{"role": "human", "content": "审批通过"}]
        }
    elif human_response == "修改_增加双人模式":
        new_design = state.get("design", "") + " [人类意见：增加双人模式]"
        return {
            "human_approved": True,
            "design": new_design,
            "messages": [{"role": "human", "content": "审批通过并修改设计"}]
        }
    else:
        return {
            "human_approved": False,
            "messages": [{"role": "human", "content": "审批拒绝"}]
        }

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
    print(f"【Tester】{report[:50]}...")
    return {"report": report, "messages": [{"role": "tester", "content": report}]}

# ========== 3. 条件边 ==========
def route_human(state: AgentState):
    return "engineer" if state.get("human_approved") else END

def route_tester(state: AgentState):
    loop = state.get("loop_count", 0)
    report = state.get("report", "")
    if loop >= 3:
        print("【路由】达到最大修复次数，强制结束")
        return END
    return "engineer" if "❌ Bug" in report else END

# ========== 4. 建图 + Checkpointer ==========
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

# ========== 5. 运行：interrupt + resume 演示 ==========
if __name__ == "__main__":
    with SqliteSaver.from_conn_string("checkpoints.sqlite") as memory:
        graph = builder.compile(checkpointer=memory)
        config = {"configurable": {"thread_id": "job-interrupt-001"}}
        
        print("\n" + "="*60)
        print("【时间旅行】查看完整执行历史")
        print("="*60)

        history = list(graph.get_state_history(config))
        print(f"历史记录数: {len(history)}")

        for i, snap in enumerate(history):
            print(f"\n--- 快照 [{i}] ---")
            print(f"  next: {snap.next}")
            print(f"  design: {snap.values.get('design','无')[:40]}...")
            print(f"  code: {'有' if snap.values.get('code') else '无'}")
            print(f"  human_approved: {snap.values.get('human_approved')}")

        # 演示：回滚到 architect 完成、human_gate 之前
        if len(history) >= 3:
            target = history[-3]  # 找到 architect 节点后的快照
            print(f"\n{'='*60}")
            print(f"【回滚】回到 {target.next} 状态，修改设计后重新审批")
            print(f"{'='*60}")
            
            # 修改历史 design
            new_values = dict(target.values)
            # 如果 design 不存在（architect 还未执行），先初始化
            if "design" not in new_values:
                new_values["design"] = "[Design] 基于PRD：[PRD] Flappy Bird"
            new_values["design"] += " [回滚修改：增加道具系统]"
            new_values["human_approved"] = False  # 重新触发审批
            
            # 用 update_state 回滚到指定节点
            graph.update_state(config, new_values, as_node="architect")
            
            # 重新 stream，从 architect 之后继续
            for event in graph.stream(Command(resume="通过"), config):
                print(f"回滚后 Event: {event}")