# rag.py - LangChain RAG (Retrieval-Augmented Generation) 示例
# 该文件演示了如何使用 LCEL 构建一个简单的检索增强生成链

# 从 langchain_core.runnables 模块导入两个核心组件：
# RunnableLambda: 允许将任意函数包装为 Runnable，使其可以参与 LCEL 管道
# RunnablePassthrough: 透传输入数据，不做任何修改直接传递给下一个组件
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# ==================== 创建模拟检索器 ====================

# 使用 RunnableLambda 包装一个 lambda 函数，模拟 Week 5 中的文档检索器
# 输入: 一个字典，包含 'question' 键
# 输出: 返回格式化的检索结果字符串
retriever = RunnableLambda(lambda inputs: f"[检索结果] 关于 {inputs['question']} 的文档内容...")

# ==================== 构建 RAG 链 ====================

# LCEL 支持字典形式的并行处理
# 当链遇到字典时，会并行执行每个值对应的组件
# {"context": retriever, "question": RunnablePassthrough()} 的含义：
#   - "context" 键：将输入传递给 retriever 获取检索上下文
#   - "question" 键：使用 RunnablePassthrough() 直接透传原始输入

# 然后通过 | 操作符连接到下一个组件，将检索结果和问题格式化为最终输出
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | RunnableLambda(lambda x: f"上下文: {x['context']}; 问题: {x['question']['question']}")
)

# ==================== 运行链 ====================

# 调用链的 invoke 方法，传入问题
# 输入: {"question": "什么是LCEL"}
# 执行流程:
# 1. 字典并行处理阶段：
#    - retriever 收到 {"question": "什么是LCEL"}，返回 "[检索结果] 关于 什么是LCEL 的文档内容..."
#    - RunnablePassthrough 直接返回 {"question": "什么是LCEL"}
# 2. 合并结果为 {"context": "...", "question": {"question": "什么是LCEL"}}
# 3. 最后一个 RunnableLambda 格式化输出
print(chain.invoke({"question": "什么是LCEL"}))