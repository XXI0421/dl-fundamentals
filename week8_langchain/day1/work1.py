# 目标：模拟 Week 5 RAG 的 "检索 + 格式化" 阶段
# 输入: {"query": "某问题"}
# 输出: "格式化后的上下文字符串"

# 要求：
# 1. retriever 用 RunnableLambda 模拟，接收 {"query": ...}，返回 str
# 2. format_docs 用 RunnableLambda 包装你的手搓格式化函数
# 3. 用 | 连接：retriever | format_docs
# 4. 测试 chain.invoke({"query": "什么是LLM"})

from langchain_core.runnables import RunnableLambda, RunnablePassthrough

retriever = RunnableLambda(lambda x: x["query"]) # 输入 dict → 输出 str
format_docs = RunnableLambda(lambda x: {"formatted": x["docs"]}) # 输入 dict → 输出 dict
chain = ({"query": retriever, "docs": RunnablePassthrough()} | format_docs)

print(chain.invoke({"query": "什么是LLM"}))