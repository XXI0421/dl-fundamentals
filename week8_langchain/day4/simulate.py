# simulate.py
import os
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END

# ========== 1. 定义 State 结构 ==========
class GraphState(TypedDict):
    """所有节点共享的状态"""
    question: str          # 用户问题
    context: str          # 检索到的上下文
    answer: str            # 生成的答案
    grade: Literal["yes", "no"]  # 评估结果：是否足够好
    loop_count: int          # 循环计数器

# ========== 2. 定义节点函数 ==========
# 每个节点：接收 state，返回对 state 的更新（字典）
def retrieve(state: GraphState):
    print(f"【节点: retrieve】问题: {state['question']}")
    # 模拟检索
    return {"context": f"关于 {state['question']} 的知识..."}

def generate(state: GraphState):
    print(f"【节点: generate】上下文: {state['context'][:30]}...")
    return {"answer": f"{state['question']} 很重要"}

def grade_answer(state: GraphState):
    """评估答案质量"""
    print(f"【节点: grade】评估答案: {state['answer'][:30]}...")
    # 模拟评估：如果问题包含"为什么"，认为需要改进
    if "为什么" in state["question"]:
        return {"grade": "no"}
    return {"grade": "yes"}

def rewrite_query(state: GraphState):
    """重写用户问题"""
    print(f"【节点: rewrite】重写问题: {state['question']}")
    return {
        "question": f"基于已有回答，请更深入说明：{state['question']} 的核心机制",
        "loop_count": state.get("loop_count", 0) + 1
    }

# ========== 3. 构建图 ==========
builder = StateGraph(GraphState)

# 添加节点
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
builder.add_node("grade", grade_answer)
builder.add_node("rewrite", rewrite_query)

# 添加边（无条件）
builder.set_entry_point("retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", "grade")

# 添加条件边（核心！）
def decide_next(state: GraphState) -> Literal["rewrite", "__end__"]:
    if state["grade"] == "yes":
        return "__end__"  # 够好，结束
    # 循环计数器超过 3 次，也结束
    if state.get("loop_count", 0) > 3:
        return "__end__"
    return "rewrite"  # 不够好，重写


builder.add_conditional_edges(
    "grade",
    decide_next,
    {"rewrite": "rewrite", "__end__": END}
)

# 重写后重新检索，形成循环
builder.add_edge("rewrite", "retrieve")

# ========== 4. 编译并运行 ==========
graph = builder.compile()

# ========== 5. 测试 ==========
if __name__ == "__main__":
    # 测试 A：简单问题（应该直接结束）
    # print("=" * 40)
    # print("【测试 A】简单问题")
    # result = graph.invoke({"question": "LCEL 是什么"})
    # print(f"最终状态: {result}")
    
    # 测试 B：复杂问题（应该触发循环：rewrite → retrieve → generate → grade）
    print("\n" + "=" * 40)
    print("【测试 B】复杂问题（触发循环）")
    result = graph.invoke({"question": "为什么需要 LangGraph"})
    print(f"最终状态: {result}")
