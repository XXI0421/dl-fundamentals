# state.py
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
import operator

# ========== 1. State 定义 ==========
class AgentState(TypedDict):
    messages: Annotated[List[dict], operator.add]  # 消息历史，自动追加
    prd: str           # PM 输出
    design: str        # Architect 输出  
    code: str          # Engineer 输出
    report: str        # Tester 输出
    next_agent: str    # 下一个执行节点
    loop_count: int    # 循环计数（防止死循环）

# ========== 2. 节点实现（模拟 Week 7 的 Agent 行为）==========
def pm_node(state: AgentState):
    """PM：根据需求生成 PRD"""
    requirement = state["messages"][-1]["content"] if state["messages"] else "开发一个简单游戏"
    prd = f"[PRD] 基于需求'{requirement}'，设计 Flappy Bird 变体游戏，包含：1.玩家控制小鸟飞行 2.管道障碍物 3.计分系统"
    print(f"【PM】生成 PRD: {prd[:50]}...")
    return {"prd": prd, "messages": [{"role": "pm", "content": prd}]}

def architect_node(state: AgentState):
    """Architect：根据 PRD 生成技术设计"""
    prd = state["prd"]
    design = f"[Design] 基于 PRD，技术栈：Python + Pygame。架构：GameLoop + SpriteManager + CollisionDetector"
    print(f"【Architect】生成设计: {design[:50]}...")
    return {"design": design, "messages": [{"role": "architect", "content": design}]}

def engineer_node(state: AgentState):
    """Engineer：根据设计生成代码"""
    design = state["design"]
    code = f"[Code] import pygame; class Bird(pygame.sprite.Sprite): ..."
    loop_count = state.get("loop_count", 0) + 1
    print(f"【Engineer】生成代码: {code[:50]}...")
    return {"code": code, "loop_count": loop_count, "messages": [{"role": "engineer", "content": code}]}

def tester_node(state: AgentState):
    """Tester：测试代码，生成报告"""
    code = state["code"]
    # 模拟：50% 概率发现 Bug（为了演示循环）
    import random
    has_bug = random.choice([True, False])
    if has_bug:
        report = "[Report] ❌ Bug 发现：碰撞检测逻辑错误，小鸟穿过管道未触发游戏结束。需修复。"
    else:
        report = "[Report] ✅ 所有测试通过。游戏可正常运行。"
    print(f"【Tester】{report[:50]}...")
    return {"report": report, "messages": [{"role": "tester", "content": report}]}

# ========== 3. 条件边：是否继续修复 ==========
def should_continue(state: AgentState) -> str:
    """判断是否有 Bug 需要修复，或达到最大循环次数"""
    report = state.get("report", "")
    loop = state.get("loop_count", 0)
    
    if loop >= 3:
        print("【路由】达到最大修复次数，强制结束")
        return END
    
    if "❌ Bug" in report:
        print(f"【路由】发现 Bug，返回 Engineer 修复（第 {loop} 轮）")
        return "engineer"

    
    return END

# ========== 4. 建图 ==========
builder = StateGraph(AgentState)

builder.add_node("pm", pm_node)
builder.add_node("architect", architect_node)
builder.add_node("engineer", engineer_node)
builder.add_node("tester", tester_node)

builder.set_entry_point("pm")
builder.add_edge("pm", "architect")
builder.add_edge("architect", "engineer")
builder.add_edge("engineer", "tester")
builder.add_conditional_edges("tester", should_continue, {"engineer": "engineer", END: END})

# 循环：tester → engineer（修复后重新测试）
# 注意：engineer 执行后回到 tester，形成闭环

graph = builder.compile()

try:
    png_bytes = graph.get_graph().draw_mermaid_png()
    with open("state_graph.png", "wb") as f:
        f.write(png_bytes)
    print("✅ 已生成可视化图：state_graph.png")
except Exception as e:
    print(f"⚠️ 无法生成可视化图：{e}")

# ========== 5. 测试 ==========
if __name__ == "__main__":
    result = graph.invoke({
        "messages": [{"role": "user", "content": "开发一个 Flappy Bird 游戏"}],
        "loop_count": 0
    })
    print(f"\n最终报告：{result['report'][:100]}...")
    print(f"总消息数：{len(result['messages'])}")
