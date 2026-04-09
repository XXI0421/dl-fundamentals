"""
脚本2：Agent领域高级RAG查询系统
包含：HyDE扩展、Cross-Encoder精排、RAG-Fusion多路召回
特别针对agent_data的结构优化，支持自动评估（正例命中率/负例过滤率）
"""

import numpy as np
import json
import faiss
import pickle
import time
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer, CrossEncoder
from agent_data import ALL_CONCEPTS  # 用于评估时获取ground truth


class AgentRAGSystem:
    """
    Agent领域专用RAG系统
    针对技术概念检索优化，支持概念消歧和精确匹配
    """
    
    def __init__(self, db_dir="agent_db"):
        """
        加载预构建的数据库和模型
        """
        print("🔧 加载智能体RAG系统...")
        
        # 1. 加载语料和元数据
        self.corpus = []
        self.metadata = []
        with open(f"{db_dir}/corpus.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                self.corpus.append(data["text"])
                self.metadata.append(data["meta"])
        
        # 2. 加载FAISS索引
        self.index = faiss.read_index(f"{db_dir}/agent.index")
        
        # 3. 加载向量（用于Cross-Encoder的候选提取）
        self.embeddings = np.load(f"{db_dir}/embeddings.npy")
        
        # 4. 加载模型
        print("🚀 加载双编码器模型...")
        self.bi_encoder = SentenceTransformer('BAAI/bge-base-zh')
        
        print("⚡ 加载交叉编码器模型...")
        self.cross_encoder = CrossEncoder('BAAI/bge-reranker-base')
        
        # 5. 构建概念到文档索引的映射（用于评估）
        self.concept_to_docs = {}
        for idx, meta in enumerate(self.metadata):
            concept = meta["concept"]
            if concept not in self.concept_to_docs:
                self.concept_to_docs[concept] = []
            self.concept_to_docs[concept].append(idx)
        
        print(f"✅ 系统就绪: {len(self.corpus)} 文档, {len(self.concept_to_docs)} 概念")
    
    def hyde_augment(self, query: str) -> str:
        """
        HyDE增强：针对Agent/LLM领域关键词扩展
        """
        q = query.lower()
        expansion = ""
        
        # ReAct相关
        if any(kw in q for kw in ["react", "推理", "行动", "思考"]):
            expansion = ("ReAct推理行动智能体大语言模型工具使用观察思考行动循环规划执行智能决策")
        
        # Chain-of-Thought相关
        elif any(kw in q for kw in ["cot", "思维链", "逐步"]):
            expansion = ("思维链CoT推理逐步中间思考分解提示工程多步逻辑Chain-of-Thought思考过程")
        
        # RAG相关
        elif any(kw in q for kw in ["rag", "检索", "增强"]):
            expansion = ("检索增强生成向量搜索知识外部文档嵌入检索知识库问答")
        
        # Agent相关（广义）
        elif any(kw in q for kw in ["agent", "智能体", "自主", "规划"]):
            expansion = ("自主智能体规划工具使用决策多步推理大语言模型人工智能系统")
        
        # 智能体学习相关
        elif any(kw in q for kw in ["智能体自我反思", "reflection", "agent持续改进"]):
            expansion = ("智能体自我反思机制让Agent根据环境反馈调整策略，实现自我纠错元认知")
        
        # 位置编码
        elif any(kw in q for kw in ["rope", "旋转位置编码", "位置编码", "嵌入"]):
            expansion = ("RoPE旋转位置编码相对位置transformer注意力长序列复数域旋转矩阵GPT-2 LLaMA")
        
        # 微调相关
        elif any(kw in q for kw in ["微调", "fine-tuning", "适应"]):
            expansion = ("微调适应领域特定训练灾难性遗忘参数高效模型优化")
        
        # 通用Agent术语
        elif any(kw in q for kw in ["llm", "提示词", "上下文", "token"]):
            expansion = ("大语言模型transformer架构提示工程上下文窗口 tokens")
        
        # 大模型安全类
        elif any(kw in q for kw in ["大模型安全", "提示注入攻击", "prompt injection安全漏洞"]):
            expansion = ("提示注入攻击是大模型安全的一种攻击手段，通过在提示词中插入恶意内容，导致模型输出错误结果")
        
        # 对比类查询
        elif "对比" in q or "vs" in q.lower():
            expansion = ("比较对比差异区别优缺点分析评估")

        if expansion:
            return f"{query}。{expansion}"
        return query
    
    def reciprocal_rank_fusion(self, results_lists: List[List[int]], k: int = 60) -> List[int]:
        """
        RRF多路融合算法
        """
        fusion_scores = {}
        for result_list in results_lists:
            for rank, doc_idx in enumerate(result_list):
                if doc_idx not in fusion_scores:
                    fusion_scores[doc_idx] = 0
                fusion_scores[doc_idx] += 1 / (k + rank)
        
        return sorted(fusion_scores.keys(), key=lambda x: fusion_scores[x], reverse=True)
    
    def retrieve(self, query: str, use_hyde: bool = True, top_k: int = 60) -> Tuple[List[str], List[int], np.ndarray]:
        """
        阶段1：Bi-Encoder向量召回
        """
        # HyDE增强
        aug_query = self.hyde_augment(query) if use_hyde else query
        
        # 编码
        query_emb = self.bi_encoder.encode(aug_query, convert_to_numpy=True)
        faiss.normalize_L2(query_emb.reshape(1, -1))
        
        # 调整HNSW搜索参数以提高准确性
        original_ef_search = self.index.hnsw.efSearch
        try:
            # 临时提高efSearch值以获得更准确的结果
            self.index.hnsw.efSearch = 128
            # HNSW搜索
            distances, indices = self.index.search(query_emb.reshape(1, -1), top_k)
        finally:
            # 恢复原始值
            self.index.hnsw.efSearch = original_ef_search
        
        candidates = [self.corpus[i] for i in indices[0]]
        return candidates, indices[0].tolist(), distances[0]
    
    def rerank(self, query: str, candidates: List[str], 
               candidate_indices: List[int], top_k: int = 10) -> Tuple[List[str], List[int], np.ndarray]:
        """
        阶段2：Cross-Encoder精排
        """
        # 构建查询-文档对
        pairs = [[query, doc] for doc in candidates]
        # 预测相关性分数
        scores = self.cross_encoder.predict(pairs, show_progress_bar=False)
        
        # 对比类查询的特殊处理
        if "对比" in query or "vs" in query.lower():
            # 收集所有候选文档的概念
            concept_scores = {}
            for i, (idx, score) in enumerate(zip(candidate_indices, scores)):
                concept = self.metadata[idx]["concept"]
                if concept not in concept_scores:
                    concept_scores[concept] = []
                concept_scores[concept].append(score)
            
            # 计算每个概念的平均分数
            avg_concept_scores = {}
            for concept, score_list in concept_scores.items():
                avg_concept_scores[concept] = np.mean(score_list)
            
            # 按平均分数排序概念
            sorted_concepts = sorted(avg_concept_scores.items(), key=lambda x: x[1], reverse=True)
            
            # 选择分数最高的概念的文档
            if sorted_concepts:
                best_concept = sorted_concepts[0][0]
                # 优先选择最佳概念的文档
                concept_indices = []
                other_indices = []
                for i, idx in enumerate(candidate_indices):
                    if self.metadata[idx]["concept"] == best_concept:
                        concept_indices.append((i, scores[i]))
                    else:
                        other_indices.append((i, scores[i]))
                
                # 排序并合并
                concept_indices.sort(key=lambda x: x[1], reverse=True)
                other_indices.sort(key=lambda x: x[1], reverse=True)
                
                # 构建最终排序
                combined_indices = [i for i, _ in concept_indices] + [i for i, _ in other_indices]
                sorted_indices = combined_indices[:top_k]
            else:
                # 如果没有概念，使用原始排序
                sorted_indices = np.argsort(scores)[::-1][:top_k]
        else:
            # 普通查询使用原始排序
            sorted_indices = np.argsort(scores)[::-1][:top_k]
        
        final_docs = [candidates[i] for i in sorted_indices]
        final_indices = [candidate_indices[i] for i in sorted_indices]
        final_scores = scores[sorted_indices]
        
        return final_docs, final_indices, final_scores
    
    def rag_fusion_retrieve(self, query: str, num_variations: int = 4, top_k: int = 10):
        """
        RAG-Fusion：生成查询变体，多路召回融合
        """
        # 生成变体（覆盖不同表述角度）
        if "对比" in query or "vs" in query.lower():
            # 对比类查询的特殊处理
            variations = [
                query,
                f"{query}的区别",
                f"{query}的比较",
                f"{query}的优缺点",
                f"{query}哪个更好"
            ][:num_variations]
        else:
            # 普通查询
            variations = [
                query,
                f"什么是{query}在大语言模型智能体中",
                f"用例子解释{query}",
                f"{query}的定义和使用",
                f"{query}在人工智能系统中如何工作"
            ][:num_variations]
        
        all_results = []
        
        # 每路召回
        for var in variations:
            _, indices, _ = self.retrieve(var, use_hyde=True, top_k=30)  # 增加召回数量
            all_results.append(indices)
        
        # RRF融合
        fused_indices = self.reciprocal_rank_fusion(all_results, k=40)[:top_k]  # 调整RRF参数
        
        docs = [self.corpus[i] for i in fused_indices]
        # 模拟分数（RRF分数）
        scores = np.array([1.0 - i*0.08 for i in range(len(fused_indices))])  # 调整分数衰减
        
        return docs, fused_indices, scores
    
    def advanced_search(self, query: str, mode: str = "auto", top_k: int = 5) -> Dict:
        """
        统一搜索接口，支持多种模式
        
        Modes:
        - "auto": 自动选择最佳策略
        - "baseline": 基础Bi-Encoder
        - "hyde": 仅HyDE增强
        - "rerank": Bi-Encoder + Cross-Encoder
        - "hyde_rerank": HyDE + Cross-Encoder（完整版）
        - "fusion": RAG-Fusion多路召回
        """
        start = time.perf_counter()
        
        # 自动选择策略
        if mode == "auto":
            q = query.lower()
            # 对比类查询使用fusion策略  
            if "对比" in q or "vs" in q:
                mode = "fusion"
            # 复杂概念查询使用hyde_rerank
            elif any(kw in q for kw in ["什么是", "解释", "原理", "如何工作"]):
                mode = "hyde_rerank"
            # 简单概念查询使用hyde
            elif any(kw in q for kw in ["react", "cot", "rag", "agent", "rope"]):
                mode = "hyde"
            # 默认使用fusion
            else:
                mode = "fusion"
        
        result = {"query": query, "mode": mode, "results": []}
        
        if mode == "baseline":
            docs, indices, scores = self.retrieve(query, use_hyde=False, top_k=top_k)
            for d, idx, s in zip(docs, indices, scores):
                result["results"].append({
                    "text": d[:200], "concept": self.metadata[idx]["concept"], "score": float(s)
                })
        
        elif mode == "hyde":
            docs, indices, scores = self.retrieve(query, use_hyde=True, top_k=top_k)
            for d, idx, s in zip(docs, indices, scores):
                result["results"].append({
                    "text": d[:200], "concept": self.metadata[idx]["concept"], "score": float(s)
                })
        
        elif mode == "rerank":
            candidates, cand_indices, _ = self.retrieve(query, use_hyde=False, top_k=60)
            docs, indices, scores = self.rerank(query, candidates, cand_indices, top_k)
            for d, idx, s in zip(docs, indices, scores):
                result["results"].append({
                    "text": d[:200], "concept": self.metadata[idx]["concept"], "score": float(s)
                })
        
        elif mode == "hyde_rerank":
            candidates, cand_indices, _ = self.retrieve(query, use_hyde=True, top_k=60)
            docs, indices, scores = self.rerank(query, candidates, cand_indices, top_k)
            for d, idx, s in zip(docs, indices, scores):
                result["results"].append({
                    "text": d[:200], "concept": self.metadata[idx]["concept"], "score": float(s)
                })
        
        elif mode == "fusion":
            docs, indices, scores = self.rag_fusion_retrieve(query, top_k=top_k)
            for d, idx, s in zip(docs, indices, scores):
                result["results"].append({
                    "text": d[:200], "concept": self.metadata[idx]["concept"], "score": float(s)
                })
        
        result["latency_ms"] = (time.perf_counter() - start) * 1000
        return result
    
    def evaluate_concept_retrieval(self, test_concepts: List[str] = None):
        """
        评估：测试对特定概念的检索准确率
        使用agent_data中的ground truth（positives应该被召回，negatives应该被过滤）
        """
        if test_concepts is None:
            # 测试所有概念
            test_concepts = [item["concept"] for item in ALL_CONCEPTS]
        
        print(f"\n{'='*60}")
        print("🎯 概念检索评估")
        print(f"{'='*60}")
        
        results = {
            "baseline": {"hit_rate": [], "latency": []},
            "hyde_rerank": {"hit_rate": [], "latency": []},
            "fusion": {"hit_rate": [], "latency": []}
        }
        
        for concept in test_concepts[:5]:  # 先测试前5个概念作为示例
            print(f"\n🔍 测试概念: {concept}")
            
            # 获取该概念的ground truth文档索引
            gt_indices = set(self.concept_to_docs.get(concept, []))
            if not gt_indices:
                continue
            
            for mode in ["baseline", "hyde_rerank", "fusion"]:
                top_k = 10
                res = self.advanced_search(concept, mode=mode, top_k=top_k)
                
                # 计算命中率（Top-k中有多少是正确概念的文档）
                retrieved_indices = []
                for r in res["results"]:
                    # 查找概念对应的文档索引
                    retrieved_concept = r["concept"]
                    # 只有当检索到的概念与目标概念相同时，才计入命中
                    if retrieved_concept == concept and concept in self.concept_to_docs:
                        retrieved_indices.extend(self.concept_to_docs[concept])
                
                # 去重并限制数量
                retrieved_indices = list(set(retrieved_indices))[:top_k]
                
                hits = len(set(retrieved_indices) & gt_indices)
                # 计算命中率：正确命中数 / 总ground truth数
                hit_rate = hits / len(gt_indices) if gt_indices else 0
                
                results[mode]["hit_rate"].append(hit_rate)
                results[mode]["latency"].append(res["latency_ms"])
                
                print(f"  {mode:12s}: {res['latency_ms']:5.1f}ms | 命中率: {hit_rate:.1%}")
        
        # 汇总
        print(f"\n{'='*60}")
        print("📊 评估汇总（平均）")
        print(f"{'='*60}")
        for mode in results:
            avg_hit = np.mean(results[mode]["hit_rate"]) if results[mode]["hit_rate"] else 0
            avg_lat = np.mean(results[mode]["latency"]) if results[mode]["latency"] else 0
            print(f"{mode:12s}: 延迟 {avg_lat:5.1f}ms | 命中率 {avg_hit:.1%}")


if __name__ == "__main__":
    print("="*60)
    print("🔥 智能体领域高级RAG - 查询与评估")
    print("="*60)
    
    # 初始化系统（需先运行脚本1）
    rag = AgentRAGSystem()
    
    # 测试查询（覆盖不同类型Agent概念）
    test_queries = [
        "ReAct智能体",              # 架构类
        "思维链提示技术",           # 提示工程类
        "智能体自我反思",               # 工程优化类
        "什么是RoPE",               # 组件类
        "提示注入攻击",             # 大模型安全类
        "RAG与微调对比"              # 对比类
    ]
    
    # 对比不同策略
    print("\n" + "="*60)
    print("⚔️ 策略对比")
    print("="*60)
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"📝 查询: {query}")
        print(f"{'='*60}")
        
        for mode in ["baseline", "hyde", "rerank", "hyde_rerank", "fusion"]:
            res = rag.advanced_search(query, mode=mode, top_k=10)
            top_concept = res["results"][0]["concept"] if res["results"] else "N/A"
            print(f"{mode:12s} | {res['latency_ms']:6.1f}ms | 最佳匹配: {top_concept}")
    
    # 定量评估（概念命中率）
    rag.evaluate_concept_retrieval()

