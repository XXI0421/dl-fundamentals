# rag.py - LangChain 基础检索示例
# 该文件演示了如何使用 LangChain 实现向量检索
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. 准备文档
raw_docs = [
    Document(page_content="LangChain 是一个框架，用于构建 LLM 应用。核心概念包括链、代理和记忆。"),
    Document(page_content="LCEL 是 LangChain Expression Language 的缩写，使用管道符号连接组件。"),
    Document(page_content="向量数据库用于存储文本的 Embedding 向量，支持相似度检索。Chroma 和 FAISS 是常用选择。"),
    Document(page_content="文本分割是 RAG 的关键步骤。RecursiveCharacterTextSplitter 按语义层级递归切分文档。"),
]

# 2. 分割（使用递归分割）
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    separators=["\n\n", "\n", "。", "，", " ", ""]
)
chunks = splitter.split_documents(raw_docs)
print(f"分割后共 {len(chunks)} 块")

# 3. 初始化 Embedding（BGE 中文）
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-zh-v1.5",
    model_kwargs={"device": "cuda"},  
    encode_kwargs={"normalize_embeddings": True}
)

# 4. 构建向量库（Chroma 内存模式）
# 注意：from_documents 会自动调用 embedding_model.embed_documents()
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    collection_name="week8_rag"
)

# 5. 基础检索
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
results = retriever.invoke("什么是 LCEL")
print("\n【基础检索 Top-3】")
for i, doc in enumerate(results):
    print(f"{i+1}. {doc.page_content[:60]}...")
