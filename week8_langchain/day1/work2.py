# 目标：实现一个可运行的 LCEL RAG Chain

# 步骤：
# 1. 准备 3 个 Document，内容随意（如 "Python是一种编程语言", "LangChain是框架", "LCEL是管道语法"）
# 2. 用 HuggingFaceEmbeddings 和 Chroma 构建 retriever
# 3. 用 ChatPromptTemplate 定义 prompt
# 4. 用 LCEL 组装 chain
# 5. 测试 chain.invoke("什么是LCEL") 和 chain.stream("什么是LCEL")

# 提示：ChatOpenAI 在 langchain_openai
# 提示：Chroma 在 langchain_community.vectorstores
# 提示：HuggingFaceEmbeddings 在 langchain_huggingface
# 提示：ChatPromptTemplate 在 langchain_core.prompts
# 提示：Document 在 langchain_core.documents

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
KIMI_API_KEY = os.getenv("KIMI_API_KEY") or "your-kimi-api-key"

# 1.构建向量库和文档
embedding_model = HuggingFaceEmbeddings(
    model_name = "BAAI/bge-base-zh-v1.5", 
    model_kwargs = {"device": "cuda"},
    encode_kwargs = {"normalize_embeddings": True}
    )

docs = [
    Document(page_content="Python是一种编程语言"),
    Document(page_content="LangChain是框架"),
    Document(page_content="LCEL是管道语法")
]

vectorstore = Chroma.from_documents(docs, embedding_model)
retriever = vectorstore.as_retriever()

# 2.定义 prompt
template = """
基于以下上下文回答问题:
{context}
问题：{question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 3.定义LLM
llm = ChatOpenAI(
    model = "moonshot-v1-128k",
    api_key = KIMI_API_KEY,
    base_url="https://api.moonshot.cn/v1",
    temperature = 0.7,
)

# 4.组装 chain
chain = (
    {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), 
    "question": RunnablePassthrough()} 
    | prompt 
    | llm 
    | StrOutputParser()
    )

if __name__ == "__main__":
    if KIMI_API_KEY == "your-kimi-api-key":
        print("请先设置 KIMI_API_KEY 环境变量")
        print("获取地址：https://platform.moonshot.cn/")
    else:
        # 使用 invoke 模式运行
        # print("问题：什么是LCEL")
        # print("回答：", chain.invoke("什么是LCEL"))

        # 使用 stream 模式运行
        print("问题：什么是LCEL")
        for chunk in chain.stream("什么是LCEL"):
            print(chunk, end="")