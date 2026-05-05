# TimeTravel.py
import os
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command
import operator

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
    """人机协同节点：支持通过/修改_xxx/拒绝"""
    if state.get("human_approved"):
        print("【HumanGate】✅ 已审批，放行")
        return {"messages": [{"role": "human_gate", "content": "审批通过"}]}

    print("【HumanGate】⏸ 触发 interrupt，等待人类审批...")

    human_response = interrupt({
        "stage": "design_review",
        "design": state.get("design", ""),
        "question": "请审批设计稿",
        "options": ["通过", "修改_增加双人模式", "修改_增加道具系统", "拒绝"]
    })

    print(f"【HumanGate】收到人类决策: {human_response}")

    if human_response == "通过":
        return {
            "human_approved": True,
            "messages": [{"role": "human", "content": "审批通过"}]
        }
    elif human_response.startswith("修改_"):
        # 提取修改内容，如 "修改_增加道具系统" → "增加道具系统"
        modification = human_response[3:]
        new_design = state.get("design", "") + f" [人类意见：{modification}]"
        return {
            "human_approved": True,
            "design": new_design,
            "messages": [{"role": "human", "content": f"审批通过并修改：{modification}"}]
        }
    else:
        return {
            "human_approved": False,
            "messages": [{"role": "human", "content": "审批拒绝"}]
        }


def engineer_node(state: AgentState):
    """代码生成：使用完整 design，不截断"""
    report = state.get("report", "")
    design = state.get("design", "")  

    if "❌ Bug" in report:
        code = f"[Code] 修复版（针对：{report}，基于设计：{design}）"
    else:
        code = f"[Code] 初始版（基于设计：{design}）"

    if "人类意见" in design:
        code += " [已融入人类意见]"
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


# ========== 5. 主流程：时间旅行演示 ==========
if __name__ == "__main__":
    with SqliteSaver.from_conn_string("checkpoints.sqlite") as memory:
        graph = builder.compile(checkpointer=memory)
        config = {"configurable": {"thread_id": "job-timetravel-001"}}

        # ===== 第1次：正常执行到 interrupt =====
        print("=" * 60)
        print("【第1次】正常执行，将在 human_gate 暂停")
        print("=" * 60)

        for event in graph.stream(
            {"messages": [{"role": "user", "content": "Flappy Bird"}], "loop_count": 0, "human_approved": False},
            config=config
        ):
            if "__interrupt__" in event:
                print(f"⏸ 捕获 interrupt: {event['__interrupt__'][0].value.get('design', '')[:40]}...")
            else:
                print(f"Event: {list(event.keys())}")

        # ===== 时间旅行：查看历史 =====
        print("\n" + "=" * 60)
        print("【时间旅行】查看执行历史")
        print("=" * 60)
        history = list(graph.get_state_history(config))
        print(f"历史快照数: {len(history)}")

        for i, snap in enumerate(history):
            print(f"\n--- 快照 [{i}] ---")
            print(f"  next: {snap.next}")
            print(f"  design: {snap.values.get('design', '无')[:40]}...")
            print(f"  code: {'有' if snap.values.get('code') else '无'}")
            print(f"  human_approved: {snap.values.get('human_approved')}")

        # ===== 回滚到 architect 之后 =====
        target = None
        for snap in history:
            if snap.next == ('human_gate',) and snap.values.get('design'):
                target = snap
                break

        if target:
            print(f"\n{'='*60}")
            print(f"【回滚】回到 architect 完成后的状态 (next={target.next})")
            print(f"{'='*60}")

            # 修改 design，追加道具系统
            new_values = dict(target.values)
            new_design = new_values["design"] + " [回滚修改：增加道具系统]"
            new_values["design"] = new_design
            # 重置审批状态，重新走 interrupt 流程
            new_values["human_approved"] = False

            # 关键：as_node="architect" 让框架知道处于 architect 执行完的状态
            graph.update_state(config, new_values, as_node="architect")
            print(f"已修改 design: {new_design[:60]}...")

            # ===== 恢复执行：stream(None) 走到 interrupt =====
            print(f"\n{'='*60}")
            print("【恢复】stream(None) 从 human_gate 继续，触发 interrupt...")
            print(f"{'='*60}")

            for event in graph.stream(None, config):
                if "__interrupt__" in event:
                    print(f"⏸ 再次捕获 interrupt，当前 design: {event['__interrupt__'][0].value.get('design', '')[:50]}...")

            # ===== 人类干预：Command(resume) =====
            print(f"\n{'='*60}")
            print("【干预】Command(resume='修改_增加道具系统')")
            print(f"{'='*60}")

            for event in graph.stream(Command(resume="修改_增加道具系统"), config):
                print(f"Event: {list(event.keys())}")
                if "engineer" in event:
                    print(f"  → Engineer code: {event['engineer']['code'][:80]}...")
                if "tester" in event:
                    print(f"  → Tester report: {event['tester']['report'][:60]}...")

            # ===== 最终验证 =====
            final = graph.get_state(config)
            print(f"\n{'='*60}")
            print("【最终交付】")
            print(f"最终代码: {final.values.get('code', '无')}")
            print(f"含'道具系统': {'道具系统' in final.values.get('code', '')}")
            print(f"含'人类意见': {'人类意见' in final.values.get('code', '')}")
            print(f"{'='*60}")
