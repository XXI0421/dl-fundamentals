# week5_style_rag.py - Week 5 风格的 RAG 实现
# 本文件演示了传统的手动编排方式实现 RAG 检索增强生成

# 从 langchain_openai 导入 ChatOpenAI，用于调用 Kimi API
from langchain_openai import ChatOpenAI

# 从 langchain_community 导入 Chroma 向量数据库
from langchain_community.vectorstores import Chroma

# 从 langchain_core 导入 Document 类，用于表示文档对象
from langchain_core.documents import Document

# 从 langchain_huggingface 导入 HuggingFaceEmbeddings，用于加载本地嵌入模型
from langchain_huggingface import HuggingFaceEmbeddings

# 导入 os 模块，用于读取环境变量
import os

# ==================== 配置 Kimi API ====================
# 从环境变量获取 Kimi API Key，如果没有则使用默认值
# 在 Moonshot AI 平台获取 API Key: https://platform.moonshot.cn/
KIMI_API_KEY = os.getenv("KIMI_API_KEY") or "your-kimi-api-key"

# ==================== 配置本地嵌入模型 ====================
# 使用 HuggingFaceEmbeddings 加载 bge-zh 中文嵌入模型
# 模型会自动从 HuggingFace Hub 下载
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-zh-v1.5",  # 指定模型名称，这里使用中型模型
    model_kwargs={"device": "cuda"},      # 设置运行设备，cuda 表示使用 GPU
    encode_kwargs={"normalize_embeddings": True}  # 对嵌入向量进行归一化处理
)

# ==================== 1. 初始化向量库 =====================
# 创建模拟文档，包含关于 LCEL 的知识
docs = [Document(page_content="LCEL是LangChain的管道语法，用于组合不同的组件")]

# 使用 Chroma 从文档创建向量库，使用本地嵌入模型
vectorstore = Chroma.from_documents(docs, embedding_model)

# 将向量库转换为检索器，用于后续检索相关文档
retriever = vectorstore.as_retriever()

# ==================== 2. 手搓函数 ====================

# 检索函数：接收查询，从向量库中获取相关文档
def retrieve(query):
    return retriever.invoke(query)  # 使用 invoke 方法调用检索器

# 文档格式化函数：将文档列表转换为字符串格式
def format_docs(docs):
    # 遍历文档列表，提取每个文档的 page_content，用两个换行符连接
    return "\n\n".join(d.page_content for d in docs)

# 提示词构建函数：将上下文和问题组合成完整的提示词
def build_prompt(context, question):
    return f"基于以下上下文回答问题：\n\n{context}\n\n问题：{question}"

# 生成函数：调用 Kimi LLM 生成回答
def generate(prompt):
    # 创建 ChatOpenAI 实例，配置为使用 Kimi API
    llm = ChatOpenAI(
        model="moonshot-v1-128k",  # 使用 Kimi 128K 上下文窗口模型
        api_key=KIMI_API_KEY,      # 设置 API Key
        base_url="https://api.moonshot.cn/v1",  # Kimi API 端点
        temperature=0.7             # 设置温度参数，控制回答随机性
    )
    response = llm.invoke(prompt)  # 调用模型生成回答
    return response.content        # 提取回答内容

# ==================== 3. 手搓编排 ====================
# 手动编排 RAG 流程，依次调用各个函数

def rag_pipeline(question):
    docs = retrieve(question)       # 步骤1：检索相关文档
    context = format_docs(docs)     # 步骤2：格式化文档为上下文
    prompt = build_prompt(context, question)  # 步骤3：构建提示词
    return generate(prompt)         # 步骤4：调用LLM生成回答

# ==================== 运行测试 ====================
if __name__ == "__main__":
    # 检查 API Key 是否正确设置
    if KIMI_API_KEY == "your-kimi-api-key":
        print("请先设置 KIMI_API_KEY 环境变量")
        print("获取地址：https://platform.moonshot.cn/")
    else:
        # 运行 RAG 管道
        print("问题：什么是LCEL")
        print("回答：", rag_pipeline("什么是LCEL"))