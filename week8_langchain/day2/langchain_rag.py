# langchain_rag.py - LangChain 检索增强生成示例
# 该文件演示了如何使用 LangChain 实现 RAG 模型，包括文档加载、向量检索、内容压缩和最终生成。

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# ========== 配置 ==========
KIMI_API_KEY = os.getenv("KIMI_API_KEY") or "your-kimi-api-key"


# ========== 1. 文档加载与分割 ==========
# 从 agent_data 提取的领域概念作为文档
raw_docs = [
    Document(page_content="ReAct是一种将推理Reasoning和行动Action交替进行的LLM Agent范式，通过Thought-Action-Observation循环实现智能决策。ReAct框架让大语言模型在推理过程中执行工具调用，结合链式思考与行动执行构建智能体架构。"),
    Document(page_content="Chain-of-Thought CoT思维链提示技术通过逐步推理展示中间思考过程，帮助大模型解决复杂问题。思维链CoT通过分解多步逻辑的中间思考过程，让大语言模型显式展示推理路径。"),
    Document(page_content="Tool Use工具调用是大模型调用外部API的能力，通过Function Calling接口扩展LLM的能力边界。大模型工具使用能力指根据用户请求选择合适工具，并通过结构化输出参数调用函数。"),
    Document(page_content="RAG检索增强生成技术结合外部知识库和大模型生成能力，先查文档再回答问题。RAG通过向量检索与大模型生成结合，解决大语言模型的幻觉问题，注入准确知识。"),
    Document(page_content="Agent Planning智能体任务规划是多步骤任务分解与执行策略，将复杂目标分解为子任务。智能体规划能力指目标导向的子任务调度，通过动态调整计划实现任务执行。"),
    Document(page_content="Reflection智能体自我反思机制让Agent根据环境反馈调整策略，实现自我纠错。自我反思能力使智能体能够总结失败经验并更新策略，具备元认知能力的自我评估。"),
    Document(page_content="Multi-Agent多智能体协作系统让多个Agent分工协作，通过角色扮演驱动对话模拟。多智能体系统通过Agent间的通信与协调机制，实现复杂任务的分布式处理。"),
    Document(page_content="System Prompt系统提示词用于定义智能体角色和行为规范，是控制大模型行为的顶层指令。系统提示词作为持久化记忆的上下文设定，约束输出格式并定义Agent的人格与行为。"),
    Document(page_content="Zero-shot零样本学习指智能体无需示例直接执行任务，基于指令遵循实现自主决策。零样本能力让大模型在没有训练样本的情况下，直接理解任务意图并执行推理。"),
    Document(page_content="Few-shot Prompting少样本提示通过提供少量示例引导模型，实现In-context Learning。少样本提示技术基于上下文学习示例演示，让模型通过类比学习输出格式和任务模式。"),
    Document(page_content="Prompt Injection提示注入攻击是通过恶意指令注入绕过系统提示的安全机制。提示注入攻击检测用户输入中的隐藏指令，防止提示词越狱和恶意操控。"),
    Document(page_content="Memory记忆机制是智能体的核心组件，包含短期记忆和长期记忆的存储与检索。智能体记忆系统通过向量数据库存储历史信息，支持对话上下文的维护。"),
    Document(page_content="LCEL是LangChain Expression Language的缩写，使用管道符号连接组件。LCEL提供声明式的方式组合链组件，支持invoke、stream、batch三种调用模式。"),
    Document(page_content="Attention Mechanism注意力机制通过Query-Key-Value计算模式动态分配权重。注意力机制是Transformer的核心组件，通过自注意力捕捉序列中的长距离依赖。"),
    Document(page_content="Backpropagation反向传播算法是神经网络训练的核心，通过链式法则计算损失对参数的梯度。反向传播算法通过误差反向传递更新权重，从输出层向输入层传播误差信号。"),
]

print(f"加载了 {len(raw_docs)} 个文档")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    separators=["\n\n", "\n", "。", "，", " ", ""]
)
chunks = splitter.split_documents(raw_docs)
print(f"分割后共 {len(chunks)} 块")


# ========== 2. 向量库构建 ==========
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-zh-v1.5",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True}
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    collection_name="week8_rag"
)


# ========== 3. 高级检索器组装 ==========
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 3.1 LLM 初始化（用于 MultiQuery 和 Compression）
llm = ChatOpenAI(
    model="moonshot-v1-128k",
    api_key=KIMI_API_KEY,
    base_url="https://api.moonshot.cn/v1",
    temperature=0.7
)

# 3.2 MultiQuery（LLM自动生成同义查询）
multi_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm,
    include_original=True
)

# 3.3 Ensemble（BM25 + 向量混合检索）
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 5
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, base_retriever],
    weights=[0.3, 0.7]  # 向量权重更高，因为语义查询多
)

# 3.4 Compression（召回后内容压缩，套在 Ensemble 外面）
compressor = FlashrankRerank()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=ensemble_retriever
)


# ========== 4. LCEL RAG Chain ==========
# 选择检索器：base_retriever / multi_retriever / ensemble_retriever / compression_retriever
SELECTED_RETRIEVER = compression_retriever

template = """基于以下上下文回答问题。如果上下文不足以回答，请说"我不知道"。

上下文：
{context}

问题：{question}
"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))

def log_retrieval(docs):
    print("\n【检索结果】")
    for i, doc in enumerate(docs):
        print(f"{i+1}. {doc.page_content[:50]}...")
    return docs

chain = (
    {"context": SELECTED_RETRIEVER | log_retrieval | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# ========== 5. 运行测试 ==========
def run_test(query):
    print(f"\n【问题】{query}")
    print(f"【回答】{chain.invoke(query)}")


if __name__ == "__main__":
    if KIMI_API_KEY == "your-kimi-api-key":
        print("请先设置 KIMI_API_KEY 环境变量")
        print("获取地址：https://platform.moonshot.cn/")
    else:
        # 测试问题集合
        test_queries = [
            "什么是 LCEL",
            # "ReAct 是什么",
            # "思维链的作用是什么",
            # "RAG 如何解决幻觉问题",
            # "智能体有哪些核心能力",
            # "什么是提示注入攻击",
            # "多智能体协作的优势"
        ]
        
        for query in test_queries:
            run_test(query)