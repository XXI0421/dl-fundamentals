import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Tuple
from config import EMBEDDING_MODEL, RERANK_MODEL, HNSW_EF_SEARCH

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
    def get_reranker(cls, model_name: str) -> CrossEncoder:
        """获取或创建重排序模型"""
        key = f"reranker_{model_name}"
        if key not in cls._models:
            print(f"Loading reranker model: {model_name}")
            cls._models[key] = CrossEncoder(model_name)
        return cls._models[key]
    
    @classmethod
    def clear_cache(cls):
        """清空模型缓存（用于测试或释放内存）"""
        cls._models.clear()

class AdvancedRetriever:
    def __init__(self, index_path: str = "./faiss_index"):
        try:
            # 加载索引
            self.index = faiss.read_index(f"{index_path}/docs.index")
            
            # 加载元数据（兼容新旧两种格式）
            with open(f"{index_path}/metadata.json", 'r', encoding='utf-8') as f:
                meta = json.load(f)
                
                # 检查是否为新格式（使用 numpy 数组）
                if "chunks" in meta:
                    # 旧格式：chunks 和 sources 直接在 JSON 中
                    self.chunks = meta["chunks"]
                    self.sources = meta["sources"]
                else:
                    # 新格式：从 numpy 文件加载
                    self.chunks = list(np.load(f"{index_path}/chunks.npy", allow_pickle=True))
                    self.sources = list(np.load(f"{index_path}/sources.npy", allow_pickle=True))
            
            # 使用模型管理器获取模型（避免重复加载）
            self.bi_encoder = ModelManager.get_encoder(EMBEDDING_MODEL)
            self.cross_encoder = ModelManager.get_reranker(RERANK_MODEL)
            
            # HNSW 搜索参数（从配置读取）
            self.index.hnsw.efSearch = HNSW_EF_SEARCH
        except Exception as e:
            print(f"Error initializing retriever: {str(e)}")
            raise
    
    def hyde_augment(self, query: str) -> str:
        """简单规则版 HyDE（与 Day 4 相同）"""
        expansions = {
            "react": "reasoning acting agent LLM",
            "agent": "autonomous planning tool use",
            "rag": "retrieval augmented generation",
            "fine-tune": "adaptation training",
            "embedding": "vector representation semantic"
        }
        
        query_lower = query.lower()
        extra_terms = []
        for key, val in expansions.items():
            if key in query_lower:
                extra_terms.append(val)
        
        if extra_terms:
            return f"{query} {' '.join(extra_terms)}"
        return query
    
    def retrieve(self, query: str, 
                 use_hyde: bool = True, 
                 use_rerank: bool = True,
                 top_k: int = 5) -> List[dict]:
        """
        完整检索流程
        返回: [{"text": "...", "source": "...", "score": 0.95}, ...]
        """
        try:
            # 1. HyDE 增强
            search_query = self.hyde_augment(query) if use_hyde else query
            
            # 2. Bi-Encoder 召回
            query_emb = self.bi_encoder.encode(search_query, convert_to_numpy=True)
            faiss.normalize_L2(query_emb.reshape(1, -1))
            
            distances, indices = self.index.search(query_emb.reshape(1, -1), 50)
            candidates = [{"text": self.chunks[i], 
                          "source": self.sources[i], 
                          "score": float(distances[0][j])} 
                         for j, i in enumerate(indices[0])]
            
            # 3. Cross-Encoder 重排序
            if use_rerank:
                try:
                    pairs = [[query, c["text"]] for c in candidates]
                    rerank_scores = self.cross_encoder.predict(pairs, show_progress_bar=False)
                    
                    for i, score in enumerate(rerank_scores):
                        candidates[i]["rerank_score"] = float(score)
                    
                    # 按重排分数排序
                    candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
                except Exception as e:
                    print(f"Warning: Reranking failed, using vector search results: {str(e)}")
            
            # 返回 Top-K
            return candidates[:top_k]
        except Exception as e:
            print(f"Error during retrieval: {str(e)}")
            return []

if __name__ == "__main__":
    # 测试（假设已构建索引）
    retriever = AdvancedRetriever()
    results = retriever.retrieve("what is ReAct agent", use_hyde=True, use_rerank=True)
    for r in results:
        print(f"[{r['source']}] {r['text'][:100]}... (score: {r.get('rerank_score', r['score']):.3f})")
