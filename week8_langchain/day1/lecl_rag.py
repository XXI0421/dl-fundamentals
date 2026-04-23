# lcel_rag.py - 使用 LCEL (LangChain Expression Language) 实现 RAG
# 本文件演示了使用 LCEL 管道语法实现检索增强生成，相比传统手搓方式更简洁

# 从 langchain_openai 导入 ChatOpenAI，用于调用 Kimi API
from langchain_openai import ChatOpenAI

# 从 langchain_community 导入 Chroma 向量数据库
from langchain_community.vectorstores import Chroma

# 从 langchain_core.prompts 导入 ChatPromptTemplate，用于创建提示词模板
from langchain_core.prompts import ChatPromptTemplate

# 从 langchain_core.output_parsers 导入 StrOutputParser，用于解析输出
from langchain_core.output_parsers import StrOutputParser

# 从 langchain_core.runnables 导入 RunnablePassthrough，用于透传输入
from langchain_core.runnables import RunnablePassthrough

# 从 langchain_core.documents 导入 Document 类，用于表示文档对象
from langchain_core.documents import Document

# 从 langchain_huggingface 导入 HuggingFaceEmbeddings，用于加载本地嵌入模型
from langchain_huggingface import HuggingFaceEmbeddings

# 导入 os 模块，用于读取环境变量
import os

# ==================== 配置 Kimi API ====================
# 在 Moonshot AI 平台获取 API Key: https://platform.moonshot.cn/
KIMI_API_KEY = os.getenv("KIMI_API_KEY") or "your-kimi-api-key"

# ==================== 配置本地嵌入模型 ====================
# 使用 HuggingFaceEmbeddings 加载 bge-zh 中文嵌入模型
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-zh-v1.5",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True}
)

# ==================== 1. 初始化向量库 ====================
# 创建模拟文档，包含关于 LCEL 的知识
docs = [Document(page_content="LCEL是LangChain的管道语法，用于以声明式方式组合不同的组件")]

# 使用 Chroma 从文档创建向量库，使用本地嵌入模型
vectorstore = Chroma.from_documents(docs, embedding_model)

# 将向量库转换为检索器
retriever = vectorstore.as_retriever()

# ==================== 2. 定义提示词模板 ====================
# 定义提示词模板，包含上下文和问题两个占位符
template = """基于以下上下文回答问题：
{context}

问题：{question}"""

# 使用 ChatPromptTemplate 从模板创建提示词对象
prompt = ChatPromptTemplate.from_template(template)

# ==================== 3. LCEL 管道（核心）====================
# 使用 LCEL 管道语法组合各个组件，这是 LCEL 的核心优势
# 数据流向：输入 -> 字典并行处理 -> prompt -> llm -> parser -> 输出
chain = (
    # 字典并行处理：同时执行检索和透传问题
    {
        # "context" 键：检索器获取相关文档后，用 lambda 函数格式化
        "context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
        # "question" 键：使用 RunnablePassthrough 直接透传输入问题
        "question": RunnablePassthrough()
    }
    # | 操作符：将上一步的字典输出传递给提示词模板
    | prompt
    # | 操作符：将格式化后的提示词传递给 Kimi LLM
    | ChatOpenAI(
        model="moonshot-v1-128k",
        api_key=KIMI_API_KEY,
        base_url="https://api.moonshot.cn/v1",
        temperature=0.7
    )
    # | 操作符：将 LLM 输出解析为字符串
    | StrOutputParser()
)

# ==================== 4. 运行（支持多种调用模式）====================
# LCEL 支持三种调用模式：
# - .invoke(): 同步调用，获取完整结果
# - .stream(): 流式调用，逐块返回结果
# - .batch(): 批量调用，处理多个输入

if __name__ == "__main__":
    if KIMI_API_KEY == "your-kimi-api-key":
        print("请先设置 KIMI_API_KEY 环境变量")
        print("获取地址：https://platform.moonshot.cn/")
    else:
        # 使用 invoke 模式运行
        print("问题：什么是LCEL")
        print("回答：", chain.invoke("什么是LCEL"))