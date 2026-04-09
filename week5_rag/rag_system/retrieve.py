import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Tuple

class AdvancedRetriever:
    def __init__(self, index_path: str = "./faiss_index"):
        # 加载索引
        self.index = faiss.read_index(f"{index_path}/docs.index")
        
        # 加载元数据
        with open(f"{index_path}/metadata.json", 'r', encoding='utf-8') as f:
            meta = json.load(f)
            self.chunks = meta["chunks"]
            self.sources = meta["sources"]
        
        # 加载模型
        self.bi_encoder = SentenceTransformer('BAAI/bge-base-en-v1.5')
        self.cross_encoder = CrossEncoder('BAAI/bge-reranker-base')
        
        # HNSW 搜索参数
        self.index.hnsw.efSearch = 100
    
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
            pairs = [[query, c["text"]] for c in candidates]
            rerank_scores = self.cross_encoder.predict(pairs, show_progress_bar=False)
            
            for i, score in enumerate(rerank_scores):
                candidates[i]["rerank_score"] = float(score)
            
            # 按重排分数排序
            candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        
        # 返回 Top-K
        return candidates[:top_k]

if __name__ == "__main__":
    # 测试（假设已构建索引）
    retriever = AdvancedRetriever()
    results = retriever.retrieve("what is ReAct agent", use_hyde=True, use_rerank=True)
    for r in results:
        print(f"[{r['source']}] {r['text'][:100]}... (score: {r.get('rerank_score', r['score']):.3f})")
