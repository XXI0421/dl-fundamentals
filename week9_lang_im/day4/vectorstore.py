# 本文件用于创建向量数据库，参考详见 day2/work.py
import os
import argparse
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_community.document_loaders import PyPDFLoader
import time

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

def build_vectorstore(chunks):
    embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-base-zh-v1.5", model_kwargs={"device": "cuda"})
    db_path = "./chroma_db"
    if os.path.exists(db_path):
        vectorstore = Chroma(persist_directory=db_path, embedding_function=embedding)
        print("加载已有向量库")
    else:
        vectorstore = Chroma.from_documents(chunks, embedding, persist_directory=db_path)
        vectorstore.persist()
        print("新建向量库")
    return vectorstore

def get_retriever(strategy, vectorstore, chunks):
    base = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = ChatOpenAI(
        model="moonshot-v1-128k", 
        api_key=os.getenv("KIMI_API_KEY") or "your-key", 
        base_url="https://api.moonshot.cn/v1"
        )
    
    if strategy == "base":
        return base

    elif strategy == "multi":
        return MultiQueryRetriever.from_llm(retriever=base, llm=llm)

    elif strategy == "ensemble":
        bm25 = BM25Retriever.from_documents(chunks)
        bm25.k = 4
        return EnsembleRetriever(retrievers=[bm25, base], weights=[0.3, 0.7])

    elif strategy == "compress":
        # 召回长文档（如整页 PDF，k=3，每篇 1000+ 字）
        base = vectorstore.as_retriever(search_kwargs={"k": 3})
        compressor = LLMChainExtractor.from_llm(llm)  # LLM 提取相关句子
        return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base)

    elif strategy == "rerank":
        # 先多召回（海选 10 个），再精排（取 4 个）
        base = vectorstore.as_retriever(search_kwargs={"k": 10})
        bm25 = BM25Retriever.from_documents(chunks)
        bm25.k = 10
        ensemble = EnsembleRetriever(retrievers=[bm25, base], weights=[0.3, 0.7])
        # Flashrank 精排：从 10 个里选最相关的 4 个
        compressor = FlashrankRerank(top_n=4)
        return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=ensemble)

if __name__ == "__main__":
    print("=" * 50)
    print("向量库工具模块")
    print("=" * 50)
    print("\n用法：")
    print("  from vectorstore import load_and_split, build_vectorstore, get_retriever")
    print("")
    print("  chunks = load_and_split()")
    print("  vectorstore = build_vectorstore(chunks)")
    print("  retriever = get_retriever('base', vectorstore, chunks)")
    print("=" * 50)