from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
import operator

class ParallelState(TypedDict):
    code: str
    reviews: Annotated[List[str], operator.add]  # 关键：operator.add 让并行节点追加不覆盖
    final_report: str

# ========== 1. 代码生成节点（上游）==========
def engineer_node(state: ParallelState):
    code = "[Code] import pygame; class Bird: pass  # 双人模式支持"
    print(f"【Engineer】生成代码: {code[:50]}...")
    return {"code": code}

# ========== 2. 并行评审节点（3 个）==========
def tester_security(state: ParallelState):
    review = f"[安全评审] 代码 {state['code'][:30]}... 检查：无 SQL 注入，无路径遍历"
    print(f"【Security】{review[:60]}...")
    return {"reviews": [review]}

def tester_performance(state: ParallelState):
    review = f"[性能评审] 代码 {state['code'][:30]}... 检查：FPS 60，内存 < 100MB"
    print(f"【Performance】{review[:60]}...")
    return {"reviews": [review]}

def tester_function(state: ParallelState):
    review = f"[功能评审] 代码 {state['code'][:30]}... 检查：双人模式正常，计分正确"
    print(f"【Function】{review[:60]}...")
    return {"reviews": [review]}

def _safe(state: ParallelState):
    review = f"[安全评审] 代码 {state['code'][:30]}... 检查：无需安全评审"
    print(f"【Safe】{review[:60]}...")
    return {"reviews": [review]}

# ========== 3. 条件边函数：返回 Send 列表实现 Map ==========
def dispatch_reviewers(state: ParallelState):
    """Map 阶段：一个输入，同时发给 3 个节点"""
    import random
    is_safe = random.choice([True, False])
    if not is_safe:
        print(f"\n【Dispatch】派发 3 个并行评审任务...")
        return [
            Send("tester_security", {"code": state["code"]}),
            Send("tester_performance", {"code": state["code"]}),
            Send("tester_function", {"code": state["code"]}),
        ]
    else:
        print("无需安全评审")
        return [Send("safe", {"code": state["code"], "reviews": state["reviews"], "final_report": state["final_report"]})]

# ========== 4. 聚合节点：Reduce 阶段 ==========
def aggregate_reviews(state: ParallelState):
    """Reduce 阶段：合并所有评审意见"""
    print(f"\n【Aggregate】收到 {len(state['reviews'])} 条评审:")
    for r in state["reviews"]:
        print(f"  - {r[:60]}...")
    
    final = "=== 综合评审报告 ===\n" + "\n".join(state["reviews"])
    return {"final_report": final}

# ========== 5. 建图 ==========
builder = StateGraph(ParallelState)

builder.add_node("engineer", engineer_node)
builder.add_node("tester_security", tester_security)
builder.add_node("tester_performance", tester_performance)
builder.add_node("tester_function", tester_function)
builder.add_node("aggregate", aggregate_reviews)
builder.add_node("safe", _safe)

builder.add_edge(START, "engineer")
builder.add_conditional_edges("engineer", dispatch_reviewers, {
    "tester_security": "tester_security",
    "tester_performance": "tester_performance",
    "tester_function": "tester_function",
    "safe": "safe",
})

# 3 个并行节点都汇聚到 aggregate
builder.add_edge("tester_security", "aggregate")
builder.add_edge("tester_performance", "aggregate")
builder.add_edge("tester_function", "aggregate")
builder.add_edge("safe", "aggregate")

builder.add_edge("aggregate", END)

graph = builder.compile()
try:
    png_bytes = graph.get_graph().draw_mermaid_png()
    with open("parallel_graph.png", "wb") as f:
        f.write(png_bytes)
    print("✅ 已生成可视化图：parallel_graph.png")
except Exception as e:
    print(f"⚠️ 无法生成可视化图：{e}")


# ========== 6. 测试 ==========
if __name__ == "__main__":
    result = graph.invoke({"code": "", "reviews": [], "final_report": ""})
    print(f"\n{'='*60}")
    print("【最终报告】")
    print(result["final_report"])
    print(f"{'='*60}")
    print(f"评审总数: {len(result['reviews'])}")
