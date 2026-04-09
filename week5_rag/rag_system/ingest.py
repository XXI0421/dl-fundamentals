import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
import re

# 尝试导入 PDF 解析器
try:
    import pypdf
    PDF_SUPPORT = True
except:
    PDF_SUPPORT = False
    print("Warning: pypdf not installed, PDF support disabled")

class DocumentIngestor:
    def __init__(self, embedding_model='BAAI/bge-base-zh'):
        self.encoder = SentenceTransformer(embedding_model)
        self.dim = self.encoder.get_sentence_embedding_dimension()
        self.chunks = []      # 存储文本块
        self.sources = []     # 存储来源（文件名+页码/位置）
        
    def parse_pdf(self, filepath: str) -> List[Tuple[str, str]]:
        """解析 PDF，返回 (文本片段, 位置标识) 列表"""
        if not PDF_SUPPORT:
            raise RuntimeError("pypdf not installed")
        
        documents = []
        with open(filepath, 'rb') as f:
            reader = pypdf.PdfReader(f)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():
                    documents.append((text, f"{os.path.basename(filepath)}:Page{i+1}"))
        return documents
    
    def parse_markdown(self, filepath: str) -> List[Tuple[str, str]]:
        """解析 Markdown，按段落返回"""
        documents = []
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 按标题分块（简单实现）
        sections = re.split(r'\n##?\s+', content)
        for i, section in enumerate(sections):
            if section.strip():
                documents.append((section.strip(), f"{os.path.basename(filepath)}:Section{i}"))
        return documents
    
    def chunk_text(self, text: str, source: str, 
                   chunk_size: int = 512, overlap: int = 128) -> List[Tuple[str, str]]:
        """
        滑动窗口分块
        关键：重叠区域避免关键信息被切分在边界
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            if len(chunk_words) < 50:  # 过滤太短的无意义块
                continue
            
            chunk_text = " ".join(chunk_words)
            chunk_id = f"{source}:Chunk{i//(chunk_size-overlap)}"
            chunks.append((chunk_text, chunk_id))
        
        return chunks
    
    def ingest_file(self, filepath: str) -> int:
        """处理单个文件，返回生成的块数"""
        ext = os.path.splitext(filepath)[1].lower()
        
        # 解析原始文档
        if ext == '.pdf':
            raw_docs = self.parse_pdf(filepath)
        elif ext in ['.md', '.markdown', '.txt']:
            raw_docs = self.parse_markdown(filepath)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        
        # 分块
        total_chunks = 0
        for text, source in raw_docs:
            chunks = self.chunk_text(text, source)
            for chunk_text, chunk_id in chunks:
                self.chunks.append(chunk_text)
                self.sources.append(chunk_id)
                total_chunks += 1
        
        print(f"Ingested {filepath}: {total_chunks} chunks")
        return total_chunks
    
    def build_index(self, index_path: str = "./faiss_index"):
        """构建 FAISS HNSW 索引并保存"""
        if not self.chunks:
            raise ValueError("No documents ingested")
        
        print(f"Encoding {len(self.chunks)} chunks...")
        embeddings = self.encoder.encode(
            self.chunks, 
            show_progress_bar=True,
            convert_to_numpy=True
        ).astype('float32')
        
        # 归一化
        faiss.normalize_L2(embeddings)
        
        # HNSW 索引
        index = faiss.IndexHNSWFlat(self.dim, 16)
        index.hnsw.efConstruction = 200
        index.add(embeddings)
        
        # 保存
        os.makedirs(index_path, exist_ok=True)
        faiss.write_index(index, f"{index_path}/docs.index")
        
        # 保存元数据（ chunks 和 sources ）
        with open(f"{index_path}/metadata.json", 'w', encoding='utf-8') as f:
            json.dump({
                "chunks": self.chunks,
                "sources": self.sources,
                "embedding_model": self.encoder.get_sentence_embedding_dimension()
            }, f, ensure_ascii=False)
        
        print(f"Index saved to {index_path}")
        return index

if __name__ == "__main__":
    # 测试：索引当前目录下的 test.md 或 test.pdf
    ingestor = DocumentIngestor()
    
    # 示例：假设你有 test.md
    if os.path.exists("test.md"):
        ingestor.ingest_file("test.md")
        ingestor.build_index()
    else:
        print("Create test.md to test ingestion")
