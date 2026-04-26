# improved_rag.py - 优化的 RAG 检索器实现
# 包含 MultiQuery、Ensemble（BM25+向量）、ContextualCompression 三种优化策略

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

# ==================== 1. 配置 ====================
KIMI_API_KEY = os.getenv("KIMI_API_KEY") or "your-kimi-api-key"

# ==================== 2. 准备文档（模拟 Week 5 的加载结果）====================
raw_docs = [
    Document(page_content="LangChain 是一个框架，用于构建 LLM 应用。核心概念包括链、代理和记忆。"),
    Document(page_content="LCEL 是 LangChain Expression Language 的缩写，使用管道符号连接组件。"),
    Document(page_content="向量数据库用于存储文本的 Embedding 向量，支持相似度检索。Chroma 和 FAISS 是常用选择。"),
    Document(page_content="文本分割是 RAG 的关键步骤。RecursiveCharacterTextSplitter 按语义层级递归切分文档。"),
]

# ==================== 3. 分割（使用递归分割）====================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    separators=["\n\n", "\n", "。", "，", " ", ""]
)
chunks = splitter.split_documents(raw_docs)
print(f"分割后共 {len(chunks)} 块")

# ==================== 4. 初始化 Embedding（BGE 中文）====================
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-zh-v1.5",
    model_kwargs={"device": "cuda"},  
    encode_kwargs={"normalize_embeddings": True}
)

# ==================== 5. 构建向量库（Chroma 内存模式）====================
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    collection_name="week8_improved_rag"
)

# ==================== 6. 初始化 LLM ====================
llm = ChatOpenAI(
    model="moonshot-v1-128k",
    api_key=KIMI_API_KEY,
    base_url="https://api.moonshot.cn/v1",
    temperature=0.7
)

# ==================== 7. 基础检索器 ====================
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ==================== 8. 优化检索器 ====================

# 8.1 MultiQuery：LLM 自动生成 3 个同义 Query 并行检索
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm,
    include_original=True  # 保留原始 Query
)

# 8.2 Ensemble：BM25 + 向量 混合检索
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 3

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, base_retriever],
    weights=[0.4, 0.6]  # BM25 权重 40%，向量 60%
)

# 8.3 ContextualCompression：召回后重排序压缩
try:
    compressor = FlashrankRerank()
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    flashrank_available = True
except Exception as e:
    print(f"Flashrank 不可用，跳过压缩检索器: {e}")
    flashrank_available = False

# ==================== 9. 测试不同检索器 ====================
def test_retrievers(query):
    print(f"\n【测试查询】{query}")
    
    # 基础检索
    print("\n--- 基础向量检索 ---")
    results = base_retriever.invoke(query)
    for i, doc in enumerate(results):
        print(f"{i+1}. {doc.page_content}")
    
    # MultiQuery 检索
    print("\n--- MultiQuery 检索 ---")
    results = multi_query_retriever.invoke(query)
    for i, doc in enumerate(results):
        print(f"{i+1}. {doc.page_content}")
    
    # Ensemble 检索
    print("\n--- Ensemble 检索 (BM25+向量) ---")
    results = ensemble_retriever.invoke(query)
    for i, doc in enumerate(results):
        print(f"{i+1}. {doc.page_content}")
    
    # ContextualCompression 检索
    if flashrank_available:
        print("\n--- ContextualCompression 检索 ---")
        results = compression_retriever.invoke(query)
        for i, doc in enumerate(results):
            print(f"{i+1}. {doc.page_content}")

if __name__ == "__main__":
    if KIMI_API_KEY == "your-kimi-api-key":
        print("请先设置 KIMI_API_KEY 环境变量")
        print("获取地址：https://platform.moonshot.cn/")
    else:
        test_retrievers("什么是 LCEL")
        test_retrievers("LangChain 有哪些核心概念")