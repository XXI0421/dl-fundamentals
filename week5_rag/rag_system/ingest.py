import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
import re
from config import EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, HNSW_M, HNSW_EF_CONSTRUCTION

# 尝试导入 PDF 解析器
try:
    import pypdf
    PDF_SUPPORT = True
except:
    PDF_SUPPORT = False
    print("Warning: pypdf not installed, PDF support disabled")

class ModelManager:
    """模型单例管理器，避免重复加载模型"""
    _instance = None
    _models = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_encoder(cls, model_name: str) -> SentenceTransformer:
        """获取或创建编码器模型"""
        if model_name not in cls._models:
            print(f"Loading encoder model: {model_name}")
            cls._models[model_name] = SentenceTransformer(model_name)
        return cls._models[model_name]
    
    @classmethod
    def clear_cache(cls):
        """清空模型缓存（用于测试或释放内存）"""
        cls._models.clear()

class DocumentIngestor:
    def __init__(self, embedding_model: str = None):
        model_name = embedding_model or EMBEDDING_MODEL
        self.encoder = ModelManager.get_encoder(model_name)
        self.dim = self.encoder.get_sentence_embedding_dimension()
        self.chunks = []
        self.sources = []
        self.chunk_size = CHUNK_SIZE
        self.overlap = CHUNK_OVERLAP
        
    def parse_pdf(self, filepath: str) -> List[Tuple[str, str]]:
        """解析 PDF，返回 (文本片段, 位置标识) 列表"""
        if not PDF_SUPPORT:
            raise RuntimeError("pypdf not installed")
        
        try:
            documents = []
            with open(filepath, 'rb') as f:
                reader = pypdf.PdfReader(f)
                for i, page in enumerate(reader.pages):
                    try:
                        text = page.extract_text()
                        if text and text.strip():
                            documents.append((text.strip(), f"{os.path.basename(filepath)}:Page{i+1}"))
                    except Exception as e:
                        print(f"Warning: Failed to extract text from page {i+1}: {str(e)}")
                        continue
            return documents
        except Exception as e:
            print(f"Error parsing PDF file {filepath}: {str(e)}")
            raise
    
    def parse_markdown(self, filepath: str) -> List[Tuple[str, str]]:
        """解析 Markdown，按段落返回"""
        try:
            documents = []
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 按标题分块（简单实现）
            sections = re.split(r'\n##?\s+', content)
            for i, section in enumerate(sections):
                if section and section.strip():
                    documents.append((section.strip(), f"{os.path.basename(filepath)}:Section{i}"))
            return documents
        except UnicodeDecodeError:
            # 尝试其他编码
            try:
                documents = []
                with open(filepath, 'r', encoding='gbk') as f:
                    content = f.read()
                
                sections = re.split(r'\n##?\s+', content)
                for i, section in enumerate(sections):
                    if section and section.strip():
                        documents.append((section.strip(), f"{os.path.basename(filepath)}:Section{i}"))
                return documents
            except Exception as e:
                print(f"Error parsing Markdown file {filepath}: {str(e)}")
                raise
        except Exception as e:
            print(f"Error parsing Markdown file {filepath}: {str(e)}")
            raise
    
    def _is_chinese(self, text: str) -> bool:
        """检测文本是否主要为中文"""
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        return chinese_chars > len(text) * 0.3
    
    def get_optimal_chunk_size(self, text: str, file_type: str) -> Tuple[int, int]:
        """根据文件类型和内容动态计算最优分块大小"""
        # 基础分块大小
        base_chunk_size = 150
        base_overlap = 30
        
        # 根据文件类型调整
        if file_type == '.pdf':
            # PDF 通常包含更多格式化内容，使用稍大的分块
            base_chunk_size = 200
            base_overlap = 40
        elif file_type in ['.md', '.markdown']:
            # Markdown 有明确的结构，使用中等分块
            base_chunk_size = 180
            base_overlap = 35
        elif file_type == '.txt':
            # 纯文本，使用标准分块
            base_chunk_size = 150
            base_overlap = 30
        
        # 根据文本长度调整
        text_length = len(text)
        if text_length > 10000:
            # 长文本使用更大的分块
            base_chunk_size = min(base_chunk_size * 1.5, 300)
        elif text_length < 1000:
            # 短文本使用更小的分块
            base_chunk_size = max(base_chunk_size * 0.7, 100)
        
        # 根据语言调整
        if self._is_chinese(text):
            # 中文每个字符更短，使用稍大的分块
            base_chunk_size = int(base_chunk_size * 1.2)
        
        return int(base_chunk_size), int(base_overlap * (base_chunk_size / 150))
    
    def _chunk_chinese(self, text: str, source: str, chunk_size: int, overlap: int) -> List[Tuple[str, str]]:
        """中文文本分块：按字符分块，保留语义完整性"""
        chunks = []
        
        for i in range(0, len(text), chunk_size - overlap):
            chunk_text = text[i:i + chunk_size]
            if len(chunk_text) < 50:
                continue
            
            chunk_id = f"{source}:Chunk{i//(chunk_size-overlap)}"
            chunks.append((chunk_text, chunk_id))
        
        return chunks
    
    def _chunk_english(self, text: str, source: str, chunk_size: int, overlap: int) -> List[Tuple[str, str]]:
        """英文文本分块：按单词分块"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            if len(chunk_words) < 50:
                continue
            
            chunk_text = " ".join(chunk_words)
            chunk_id = f"{source}:Chunk{i//(chunk_size-overlap)}"
            chunks.append((chunk_text, chunk_id))
        
        return chunks
    
    def chunk_text(self, text: str, source: str, file_type: str = '.txt') -> List[Tuple[str, str]]:
        """
        智能分块：根据文件类型和内容动态调整分块策略
        关键：重叠区域避免关键信息被切分在边界
        """
        # 动态计算最优分块大小
        chunk_size, overlap = self.get_optimal_chunk_size(text, file_type)
        print(f"Using chunk size: {chunk_size}, overlap: {overlap} for {file_type}")
        
        if self._is_chinese(text):
            return self._chunk_chinese(text, source, chunk_size, overlap)
        else:
            return self._chunk_english(text, source, chunk_size, overlap)
    
    def ingest_file(self, filepath: str) -> int:
        """处理单个文件，返回生成的块数"""
        try:
            ext = os.path.splitext(filepath)[1].lower()
            
            # 解析原始文档
            if ext == '.pdf':
                if not PDF_SUPPORT:
                    raise RuntimeError("pypdf not installed, cannot parse PDF files")
                raw_docs = self.parse_pdf(filepath)
            elif ext in ['.md', '.markdown', '.txt']:
                raw_docs = self.parse_markdown(filepath)
            else:
                raise ValueError(f"Unsupported file type: {ext}")
            
            # 分块
            total_chunks = 0
            for text, source in raw_docs:
                chunks = self.chunk_text(text, source, file_type=ext)
                for chunk_text, chunk_id in chunks:
                    self.chunks.append(chunk_text)
                    self.sources.append(chunk_id)
                    total_chunks += 1
            
            print(f"Ingested {filepath}: {total_chunks} chunks")
            return total_chunks
        except Exception as e:
            print(f"Error ingesting file {filepath}: {str(e)}")
            return 0
    
    def build_index(self, index_path: str = "./faiss_index"):
        """构建 FAISS HNSW 索引并保存"""
        try:
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
            index = faiss.IndexHNSWFlat(self.dim, HNSW_M)
            index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
            index.add(embeddings)
            
            # 保存
            os.makedirs(index_path, exist_ok=True)
            faiss.write_index(index, f"{index_path}/docs.index")
            
            # 优化元数据存储：使用 numpy 数组存储大对象
            np.save(f"{index_path}/chunks.npy", np.array(self.chunks, dtype=object))
            np.save(f"{index_path}/sources.npy", np.array(self.sources, dtype=object))
            
            # 只保存轻量级元数据到 JSON
            with open(f"{index_path}/metadata.json", 'w', encoding='utf-8') as f:
                json.dump({
                    "num_chunks": len(self.chunks),
                    "embedding_dim": self.dim,
                    "embedding_model": EMBEDDING_MODEL,
                    "chunk_size": self.chunk_size,
                    "overlap": self.overlap,
                    "hnsw_m": HNSW_M,
                    "hnsw_ef_construction": HNSW_EF_CONSTRUCTION
                }, f, ensure_ascii=False)
            
            print(f"Index saved to {index_path}")
            return index
        except Exception as e:
            print(f"Error building index: {str(e)}")
            raise

if __name__ == "__main__":
    # 测试：索引当前目录下的 test.md 或 test.pdf
    ingestor = DocumentIngestor()
    
    # 示例：假设你有 test.md
    if os.path.exists("test.md"):
        ingestor.ingest_file("test.md")
        ingestor.build_index()
    else:
        print("Create test.md to test ingestion")
