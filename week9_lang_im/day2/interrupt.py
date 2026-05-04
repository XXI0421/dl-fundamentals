# interrupt.py
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
        
        print("=" * 60)
        print("【Step 1】启动图执行，将在 human_gate 触发 interrupt")
        print("=" * 60)
        
        # 第 1 次 stream：运行到 interrupt 处自动暂停
        for event in graph.stream(
            {
                "messages": [{"role": "user", "content": "Flappy Bird"}],
                "loop_count": 0,
                "human_approved": False
            },
            config=config
        ):
            print(f"Event: {event}")
        
        # 检查中断状态
        snapshot = graph.get_state(config)
        print(f"\n当前 next: {snapshot.next}")
        print(f"当前 design: {snapshot.values.get('design', '无')[:50]}...")
        
        # 打印 interrupt 信息
        if snapshot.tasks:
            for task in snapshot.tasks:
                if getattr(task, "interrupts", None):
                    print(f"\n⏸ 检测到 interrupt 数据:")
                    for idx, intr in enumerate(task.interrupts):
                        print(f"   [{idx}] {intr.value}")
        
        print("\n" + "=" * 60)
        print("【Step 2】人类决策：Command(resume='修改_增加双人模式')")
        print("=" * 60)
        
        # 模拟人类审批：传回决策，图从 interrupt 处恢复
        human_decision = "修改_增加双人模式"
        for event in graph.stream(Command(resume=human_decision), config):
            print(f"Event: {event}")
        
        # 最终结果
        final = graph.get_state(config)
        print(f"\n{'='*60}")
        print("【最终交付】")
        print(f"代码: {final.values.get('code', '无')[:80]}...")
        print(f"报告: {final.values.get('report', '无')}")
        print(f"含'双人模式': {'双人模式' in final.values.get('code', '')}")
        print(f"含'人类意见': {'人类意见' in final.values.get('code', '')}")
        print(f"{'='*60}")
