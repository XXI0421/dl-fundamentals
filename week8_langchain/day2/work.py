# 场景：你有一个 data/ 目录，里面有混合格式的技术文档（.txt, .md, .pdf）。
# 要求构建一个 RAG 系统，支持运行时通过命令行参数切换检索策略，所有策略共用同一套 LCEL 管道。
# 目标：代码量控制在 60 行以内（不含 import），但覆盖 Day 2 全部核心知识点。

import os
import argparse
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_community.document_loaders import PyPDFLoader

# 步骤 A：文档加载与分割
# 加载 data/ 目录下所有 .txt, .md, .pdf 文件
# 用 RecursiveCharacterTextSplitter 分割，但后处理合并碎片：
# 如果某个 chunk 长度  < 30  字，把它和下一个 chunk 拼接（避免单独的小标题 chunk 浪费 Embedding 和检索名额）
# 打印最终 chunk 数量和平均长度
def load_and_split(data_dir="data"):
    docs = []
    for f in os.listdir(data_dir):
        path = os.path.join(data_dir, f)
        try:
            if f.endswith(".txt"):
                with open(path, "r", encoding="utf-8") as file:
                    content = file.read()
                    docs.append(Document(page_content=content, metadata={"source": path}))
            elif f.endswith(".md"):
                with open(path, "r", encoding="utf-8") as file:
                    content = file.read()
                    docs.append(Document(page_content=content, metadata={"source": path}))
            elif f.endswith(".pdf"):
                docs.extend(PyPDFLoader(path).load())
        except Exception as e:
            print(f"⚠ 加载 {f} 失败: {e}")
    # 分割
    splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30, separators=["\n\n", "\n", "。", "，", " ", ""])
    chunks = splitter.split_documents(docs) if docs else [Document(page_content="LCEL是LangChain管道语法")]
    # 合并短 chunk
    merged = []
    i = 0
    while i < len(chunks):
        curr = chunks[i]
        while i + 1 < len(chunks) and len(curr.page_content) < 30:
            curr = Document(page_content=curr.page_content + chunks[i+1].page_content)
            i += 1
        merged.append(curr)
        i += 1
    avg_len = sum(len(c.page_content) for c in merged) // len(merged)
    print(f"分割完成：{len(merged)} 块，平均长度 {avg_len}")
    return merged

# 步骤 B：向量库构建与持久化
# 用 BAAI/bge-base-zh-v1.5 + Chroma 
# 持久化到 ./chroma_db_week8 
# 如果目录已存在，直接加载，不重新 Embedding
# 打印 "新建向量库" 或 "加载已有向量库"
def build_vectorstore(chunks):
    embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-base-zh-v1.5", model_kwargs={"device": "cuda"})
    db_path = "./chroma_db_week8"
    if os.path.exists(db_path):
        vectorstore = Chroma(persist_directory=db_path, embedding_function=embedding)
        print("加载已有向量库")
    else:
        vectorstore = Chroma.from_documents(chunks, embedding, persist_directory=db_path)
        vectorstore.persist()
        print("新建向量库")
    return vectorstore

# 步骤 C：策略工厂（核心工程模式）
# 实现一个函数 get_retriever(strategy: str, vectorstore, llm, chunks):
# strategy	实现	额外依赖	
# base	vectorstore.as_retriever(k=4)	无	
# multi		MultiQueryRetriever.from_llm(...)   需要 LLM	
# ensemble	EnsembleRetriever([bm25, vector], weights=[0.3, 0.7])   需要 BM25Retriever.from_documents(chunks)
# compress	ContextualCompressionRetriever 套在 ensemble 外面	需要 compressor = FlashrankRerank()
def get_retriever(strategy, vectorstore, llm, chunks):
    base = vectorstore.as_retriever(search_kwargs={"k": 4})
    if strategy == "base":
        return base
    elif strategy == "multi":
        return MultiQueryRetriever.from_llm(retriever=base, llm=llm)
    elif strategy == "ensemble":
        bm25 = BM25Retriever.from_documents(chunks)
        bm25.k = 4
        return EnsembleRetriever(retrievers=[bm25, base], weights=[0.3, 0.7])
    elif strategy == "compress":
        bm25 = BM25Retriever.from_documents(chunks)
        bm25.k = 4
        ensemble = EnsembleRetriever(retrievers=[bm25, base], weights=[0.3, 0.7])
        return ContextualCompressionRetriever(base_compressor=FlashrankRerank(), base_retriever=ensemble)

# 步骤 D：LCEL 组装与观测
# 统一 Chain:{"context": retriever | log_retrieval | format_docs, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser() 
# log_retrieval 必须打印：召回 chunk 数量、每个 chunk 的前 30 字、策略名称
# 支持 chain.invoke() 和 chain.stream() 两种调用
def build_chain(retriever, llm, strategy):
    def log_retrieval(docs):
        print(f"\n【策略】{strategy} | 召回 {len(docs)} 条")
        for i, d in enumerate(docs):
            source = d.metadata.get('source', 'unknown')
            source_name = os.path.basename(source) if source != 'unknown' else 'unknown'
            print(f"  [{i+1}] {d.page_content[:30]}... [{source_name}]")
        return docs

    def format_docs(docs):
        formatted = []
        for d in docs:
            source = d.metadata.get('source', 'unknown')
            source_name = os.path.basename(source) if source != 'unknown' else 'unknown'
            formatted.append(f"{d.page_content} [{source_name}]")
        return "\n".join(formatted)

    prompt = ChatPromptTemplate.from_template("基于上下文回答：\n{context}\n问题：{question}")
    return (
        {"context": retriever | log_retrieval | format_docs, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )

# 步骤 E：对比实验
# 用以下 3 个 Query 在 4 种策略上测试，记录观察：
# queries = [
#     "LCEL 是什么",           # 直接关键词
#     "怎么把组件串起来",       # 语义描述，无关键词
#     "RAG 和 Agent 的关系"    # 需要综合多段知识
#     "链条传动原理"          # 测试噪声过滤
# ]

# 要求：
# 命令行支持  python work.py --strategy base （或  multi / ensemble / compress ）
# 加载  data/  目录，自动处理  .txt / .md / .pdf 
# 持久化向量库，二次启动秒加载
# 4 种策略运行时切换，共用同一套 LCEL Chain
# 每个 Query 打印策略名、召回 chunk 数、延迟、回答
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="base", choices=["base", "multi", "ensemble", "compress"])
    args = parser.parse_args()

    llm = ChatOpenAI(model="moonshot-v1-128k", api_key=os.getenv("KIMI_API_KEY") or "your-key", base_url="https://api.moonshot.cn/v1")
    chunks = load_and_split()
    vectorstore = build_vectorstore(chunks)
    retriever = get_retriever(args.strategy, vectorstore, llm, chunks)
    chain = build_chain(retriever, llm, args.strategy)

    queries = [ # "LCEL 是什么", 
                "怎么把组件串起来", 
                # "RAG 和 Agent 的关系", 
                # "链条传动原理"
                ]
    for q in queries:
        print(f"\n【问题】{q}")
        print(f"【回答】{chain.invoke(q)}")