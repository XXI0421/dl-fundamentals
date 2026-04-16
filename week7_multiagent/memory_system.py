"""
LongTermMemory 完整实现 - 第二课
基于Week 6真实代码重构，支持：
1. FAISS向量索引（真实语义检索）
2. 单例模式（强制所有Agent共享）
3. 磁盘持久化
"""
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import time


class LongTermMemory:
    """
    长期记忆系统（基于FAISS的向量检索）
    
    核心特性：
    1. 语义检索：基于FAISS的相似度搜索
    2. 持久化：facts.json + FAISS向量索引
    3. 单例模式：全局唯一实例强制共享
    
    存储结构：
    ./long_term_memory/
        ├── facts.json      # 结构化事实列表
        └── facts.index     # FAISS向量索引
    """
    
    _instance: Optional['LongTermMemory'] = None
    _initialized: bool = False
    
    def __new__(cls, storage_path: str = "./long_term_memory"):
        """强制单例：无论new多少次，返回同一对象"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, storage_path: str = "./long_term_memory"):
        # 防止重复初始化（单例模式关键）
        if LongTermMemory._initialized:
            return
            
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.facts_file = self.storage_path / "facts.json"
        self.index_file = self.storage_path / "facts.index"
        
        # 向量维度（bge-base-zh）
        self.dim = 768
        
        # 初始化FAISS索引
        self._init_index()
        
        # 加载编码器（复用Week 5模型）
        self.encoder = SentenceTransformer("BAAI/bge-base-zh")
        
        # 元数据存储
        self.facts: Dict[str, Dict] = {}  # id -> {text, category, timestamp, importance, index}
        self._load_metadata()
        
        LongTermMemory._initialized = True
        print(f"[LTM] 长期记忆单例初始化 @ {storage_path}")
        print(f"[LTM] 已加载 {len(self.facts)} 条历史事实")
    
    def _init_index(self):
        """初始化FAISS索引"""
        if self.index_file.exists():
            self.index = faiss.read_index(str(self.index_file))
            print(f"[LTM] 加载FAISS索引，包含 {self.index.ntotal} 条向量")
        else:
            # HNSW索引，适合小规模高维向量（<10万条）
            self.index = faiss.IndexHNSWFlat(self.dim, 16)
            self.index.hnsw.efConstruction = 40
            self.index.hnsw.efSearch = 16
            print(f"[LTM] 创建新的FAISS HNSW索引")
    
    def _load_metadata(self):
        """从磁盘加载历史事实"""
        if self.facts_file.exists():
            with open(self.facts_file, 'r', encoding='utf-8') as f:
                self.facts = json.load(f)
    
    def _save_metadata(self):
        """持久化到磁盘（即时写入，防丢失）"""
        with open(self.facts_file, 'w', encoding='utf-8') as f:
            json.dump(self.facts, f, ensure_ascii=False, indent=2)
        faiss.write_index(self.index, str(self.index_file))
    
    def add_fact(self, text: str, category: str = "general", 
                 importance: float = 0.5, agent_id: str = "unknown") -> str:
        """
        添加事实到长期记忆（关键接口）
        
        Args:
            text: 事实内容（自然语言）
            category: 分类（sop_progress, user_profile, tech_decision等）
            importance: 重要度 0-1（用于冲突解决时排序）
            agent_id: 写入者标识（用于溯源）
        
        Returns:
            fact_id: 事实ID
        """
        # 去重检查（语义相似度 > 0.95 则更新）
        existing_id = self._check_duplicate(text)
        if existing_id:
            self.facts[existing_id]["importance"] = max(
                self.facts[existing_id]["importance"], 
                importance
            )
            self.facts[existing_id]["timestamp"] = datetime.now().isoformat()
            self._save_metadata()
            print(f"[LTM] 更新已有事实 [{category}]: {text[:40]}...")
            return existing_id
        
        # 编码并添加到索引
        embedding = self.encoder.encode(text, convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(embedding.reshape(1, -1))
        
        fact_id = hashlib.md5(text.encode()).hexdigest()[:12]
        self.index.add(embedding.reshape(1, -1))
        
        self.facts[fact_id] = {
            "id": fact_id,
            "text": text,
            "category": category,
            "importance": importance,
            "agent_id": agent_id,
            "timestamp": datetime.now().isoformat(),
            "index": self.index.ntotal - 1  # 在FAISS中的位置
        }
        
        self._save_metadata()
        print(f"[LTM] Agent({agent_id}) 写入事实 [{category}]: {text[:40]}...")
        return fact_id
    
    def _check_duplicate(self, text: str, threshold: float = 0.95) -> Optional[str]:
        """检查是否已存在相似事实"""
        if self.index.ntotal == 0:
            return None
        
        emb = self.encoder.encode(text, convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(emb.reshape(1, -1))
        
        D, I = self.index.search(emb.reshape(1, -1), 1)
        # L2距离越小越相似，转换为相似度：sim = 1 / (1 + distance)
        similarity = 1 / (1 + D[0][0])
        if similarity > threshold:
            for fid, meta in self.facts.items():
                if meta.get("index") == I[0][0]:
                    return fid
        return None
    
    def retrieve(self, query: str, top_k: int = 3, 
                 category_filter: Optional[str] = None) -> List[Dict]:
        """
        语义检索（真实FAISS向量搜索）
        
        Args:
            query: 查询文本
            top_k: 返回数量
            category_filter: 类别过滤
        
        Returns:
            按相似度排序的事实列表
        """
        if self.index.ntotal == 0:
            return []
        
        emb = self.encoder.encode(query, convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(emb.reshape(1, -1))
        
        D, I = self.index.search(emb.reshape(1, -1), top_k * 2)  # 多取一些用于过滤
        
        results = []
        for dist, idx in zip(D[0], I[0]):
            # 找到对应事实
            for fid, meta in self.facts.items():
                if meta.get("index") == idx:
                    # 类别过滤
                    if category_filter and meta["category"] != category_filter:
                        continue
                    
                    # 重要性过滤
                    if meta["importance"] < 0.3:
                        continue
                    
                    # L2距离转换为相似度
                    similarity = 1 / (1 + dist)
                    
                    results.append({
                        "text": meta["text"],
                        "category": meta["category"],
                        "score": float(similarity),
                        "agent_id": meta.get("agent_id", "unknown"),
                        "timestamp": meta["timestamp"]
                    })
                    break
        
        # 按相似度排序
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results[:top_k]
    
    def get_user_profile(self) -> Dict:
        """获取用户画像（特殊便捷方法）"""
        prefs = [f for f in self.facts.values() if f["category"] == "user_profile"]
        return {
            "preferences": [p["text"] for p in prefs],
            "fact_count": len(self.facts)
        }
    
    def get_all_facts(self) -> List[Dict]:
        """调试接口：查看所有记忆"""
        return list(self.facts.values())
    
    def get_fact_count(self) -> int:
        """获取事实数量（用于调试）"""
        return len(self.facts)
    
    def delete_fact(self, fact_id: str):
        """删除事实（标记重要性为0）"""
        if fact_id in self.facts:
            self.facts[fact_id]["importance"] = 0
            self._save_metadata()
            print(f"[LTM] 删除事实: {fact_id}")


def get_long_term_memory() -> LongTermMemory:
    """
    全局单例获取函数（强制所有Agent共享同一记忆）
    
    这是Week 7架构的核心约束：无论何处调用，返回同一LTM实例
    """
    return LongTermMemory()


# 短期记忆（保持独立，每个Agent实例化时创建）
class ConversationSummaryMemory:
    """
    短期记忆：基于滑动窗口的对话摘要
    每个Agent独立拥有，不共享
    """
    
    def __init__(self, k: int = 3):
        self.k = k  # 滑动窗口大小
        self.messages: List[Dict] = []
        self.summaries: List[str] = []  # 整轮总结
    
    def add_message(self, role: str, content: str):
        """添加消息（来自MessageBus的接收）"""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })
        
        # 触发整轮总结（简化模拟：每2条消息总结一次）
        if len(self.messages) % 2 == 0:
            self._summarize_round()
    
    def _summarize_round(self):
        """模拟整轮总结（实际应为LLM调用）"""
        last_msgs = self.messages[-2:]
        summary = f"[摘要] {last_msgs[0]['role']}与{last_msgs[1]['role']}讨论了{last_msgs[0]['content'][:20]}..."
        self.summaries.append(summary)
        # 保持窗口大小
        if len(self.summaries) > self.k:
            self.summaries.pop(0)
    
    def get_context(self) -> str:
        """获取当前短期记忆上下文（用于Prompt注入）"""
        recent = "\n".join([f"{m['role']}: {m['content'][:50]}..." for m in self.messages[-4:]])
        summary = "\n".join(self.summaries)
        return f"## 近期对话摘要\n{summary}\n\n## 最近消息\n{recent}"
    
    def add_system_context(self, context: str):
        """SOP引擎使用：添加上游Agent产出到短期记忆"""
        self.messages.append({
            "role": "system",
            "content": f"[上游产出] {context}",
            "timestamp": time.time()
        })
    
    def clear(self):
        """清空短期记忆"""
        self.messages = []
        self.summaries = []



