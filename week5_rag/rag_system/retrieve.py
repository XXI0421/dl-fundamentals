import faiss
import json
import numpy as np
import os
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Tuple
from config import EMBEDDING_MODEL, RERANK_MODEL, HNSW_EF_SEARCH, KIMI_API_KEY, KIMI_API_URL, KIMI_MODEL, ENABLE_KIMI_API

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
    
    def _call_kimi_api(self, prompt: str) -> str:
        """调用 Kimi API 生成文本"""
        try:
            # 动态读取 API key
            api_key = os.getenv('KIMI_API_KEY', KIMI_API_KEY)
            api_url = os.getenv('KIMI_API_URL', KIMI_API_URL)
            model = os.getenv('KIMI_MODEL', KIMI_MODEL)
            
            # 确保 API key 是字符串且不包含中文
            api_key = str(api_key)
            if any('\u4e00' <= c <= '\u9fff' for c in api_key):
                print("Warning: API key contains Chinese characters, which may cause issues")
            
            headers = {
                "Content-Type": "application/json; charset=utf-8",
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json"
            }
            
            data = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 500
            }
            
            # 手动编码 JSON 以确保 UTF-8
            import json
            json_data = json.dumps(data, ensure_ascii=False).encode('utf-8')
            
            # 确保使用 UTF-8 编码
            response = requests.post(
                api_url, 
                headers=headers, 
                data=json_data, 
                timeout=30, 
                verify=False  # 可选：如果遇到 SSL 问题
            )
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            print(f"Error calling Kimi API: {str(e)}")
            return ""
    
    def hyde_augment(self, query: str) -> str:
        """HyDE 增强：使用 LLM 生成虚拟文档"""
        # 动态读取配置
        enable_kimi = os.getenv('ENABLE_KIMI_API', 'False').lower() == 'true'
        kimi_api_key = os.getenv('KIMI_API_KEY', '')
        
        if enable_kimi and kimi_api_key and kimi_api_key != 'your_kimi_api_key_here':
            # 使用 Kimi API 生成虚拟文档
            prompt = f"请针对以下问题生成一个详细的回答文档，假设你是这方面的专家：\n{query}"
            virtual_document = self._call_kimi_api(prompt)
            
            if virtual_document:
                print("Using Kimi API for HyDE augmentation")
                return virtual_document
            else:
                # 如果 API 调用失败，回退到规则版
                print("Kimi API failed, falling back to rule-based augmentation")
        
        # 规则版 HyDE（作为回退）
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
    
    def generate_answer(self, query: str, retrieved_docs: List[dict]) -> str:
        """
        基于检索到的文档生成回答
        使用 Kimi API 进行总结
        """
        # 动态读取配置
        enable_kimi = os.getenv('ENABLE_KIMI_API', 'False').lower() == 'true'
        kimi_api_key = os.getenv('KIMI_API_KEY', '')
        
        if enable_kimi and kimi_api_key and kimi_api_key != 'your_kimi_api_key_here':
            # 使用 Kimi API 生成回答
            context = "\n\n".join([f"来源: {doc['source']}\n内容: {doc['text']}" for doc in retrieved_docs[:3]])
            prompt = f"请基于以下检索到的信息，对用户的问题进行详细回答。\n\n用户问题: {query}\n\n检索到的信息:\n{context}\n\n回答要求:\n1. 直接回答问题，不要有任何引言或开场白\n2. 基于检索到的信息，不要添加外部知识\n3. 回答要详细、准确、有条理\n4. 引用信息时要注明来源"
            
            answer = self._call_kimi_api(prompt)
            if answer:
                print("Using Kimi API for answer generation")
                return answer
        
        # 回退到简单总结
        print("Falling back to simple summary")
        if retrieved_docs:
            return f"基于检索到的信息，关于 '{query}' 的相关内容来自以下文档: {', '.join([doc['source'] for doc in retrieved_docs[:3]])}"
        else:
            return "未检索到相关信息，无法生成回答。"

if __name__ == "__main__":
    # 测试（假设已构建索引）
    retriever = AdvancedRetriever()
    results = retriever.retrieve("what is ReAct agent", use_hyde=True, use_rerank=True)
    for r in results:
        print(f"[{r['source']}] {r['text'][:100]}... (score: {r.get('rerank_score', r['score']):.3f})")
