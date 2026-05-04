# langgraph_rag.py
import os
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# ========== 1. State ==========
class RAGState(TypedDict):
    question: str
    context: str
    answer: str
    grade: Literal["sufficient", "insufficient"]
    loop_count: int

# ========== 2. 组件初始化（复用 Day 2）==========
embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-zh-v1.5",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True}
)

# 加载已有向量库（不要重新 Embedding）
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

llm = ChatOpenAI(
    model="moonshot-v1-128k",
    api_key=os.getenv("KIMI_API_KEY") or "your-key",
    base_url="https://api.moonshot.cn/v1",
    temperature=0
)

# ========== 3. 节点实现 ==========

def retrieve(state: RAGState):
    """检索节点"""
    print(f"\n【检索】{state['question'][:50]}...")
    docs = retriever.invoke(state["question"])
    context = "\n\n".join(f"[{i+1}] {d.page_content[:150]}" for i, d in enumerate(docs))
    return {"context": context}

def generate(state: RAGState):
    """生成节点"""
    prompt = ChatPromptTemplate.from_template("""基于以下上下文回答问题。如果上下文不足以回答，请明确说"信息不足"。

上下文：
{context}

问题：{question}
请给出简洁准确的回答：""")
    
    chain = prompt | llm
    answer = chain.invoke({
        "context": state["context"], 
        "question": state["question"]
    }).content
    print(f"【生成】{answer[:80]}...")
    return {"answer": answer}

def grade_answer(state: RAGState):
    """评估节点：硬编码兜底 + LLM 精排"""
    answer = state.get("answer", "")
    
    # 第一层：硬编码
    insufficient_signals = ["信息不足", "不知道", "无法回答", "没有相关", "未找到"]
    if any(signal in answer for signal in insufficient_signals):
        print("【评估】insufficient (硬编码命中)")
        return {"grade": "insufficient"}
    
    # 第二层：LLM 评估
    prompt = f"""严格评估以下回答是否充分回答了问题。

问题：{state['question']}
回答：{answer}
上下文：{state['context'][:200]}...

如果回答充分且准确，输出 sufficient。
如果回答不充分、模糊或偏离问题，输出 insufficient。
只输出一个词，不要解释。"""

    result = llm.invoke(prompt).content.strip().lower()
    grade = "sufficient" if "sufficient" in result else "insufficient"
    print(f"【评估】{grade} (LLM判断)")
    return {"grade": grade}


def rewrite_query(state: RAGState):
    """改写节点：LLM 基于上一轮失败原因，生成新的检索 Query"""
    prompt = f"""原问题：{state['question']}
上一轮答案被认为：{state['grade']}
参考上下文：{state['context'][:200]}...

任务：生成一个更具体、更适合向量检索的新问题。
要求：
- 不要重复原问题的措辞
- 加入更具体的关键词
- 只输出新问题，不要解释

新问题："""

    new_question = llm.invoke(prompt).content.strip()
    print(f"【改写】{state['question'][:40]}... → {new_question[:40]}...")
    
    return {
        "question": new_question,
        "loop_count": state.get("loop_count", 0) + 1
    }

# ========== 4. 建图 ==========
builder = StateGraph(RAGState)

builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
builder.add_node("grade", grade_answer)
builder.add_node("rewrite", rewrite_query)

builder.set_entry_point("retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", "grade")

def route(state: RAGState):
    """条件路由"""
    if state.get("grade") == "sufficient":
        return END
    if state.get("loop_count", 0) >= 2:  # 最多重写 2 次（共 3 轮检索）
        print("【终止】达到最大迭代次数，返回当前答案")
        return END
    return "rewrite"

builder.add_conditional_edges("grade", route, {"rewrite": "rewrite", END: END})
builder.add_edge("rewrite", "retrieve")

graph = builder.compile()

try:
    png_bytes = graph.get_graph().draw_mermaid_png()
    with open("raggraph.png", "wb") as f:
        f.write(png_bytes)
    print("✅ 已生成可视化图：raggraph.png")
except Exception as e:
    print(f"⚠️ 无法生成可视化图：{e}")

# ========== 5. 测试入口 ==========
if __name__ == "__main__":
    test_queries = [
        # "LCEL 是什么",  # 文档里有，预期 1 轮 sufficient
        "LangGraph 和 CrewAI 有什么区别",  # 文档里可能没有，预期触发 rewrite
    ]
    
    for q in test_queries:
        print(f"\n{'='*50}")
        print(f"【问题】{q}")
        result = graph.invoke({"question": q, "loop_count": 0})
        print(f"\n【最终答案】{result['answer']}")
        print(f"【迭代次数】{result.get('loop_count', 0)}")
