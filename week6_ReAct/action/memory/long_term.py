import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class LongTermMemory:
    """会话级跨会话持久记忆（基于 FAISS）
    
    支持会话隔离：每个会话有独立的记忆存储
    """
    
    def __init__(self, 
                 session_id: Optional[str] = None,
                 index_path: str = "./long_term_memory",
                 embedding_model: str = "BAAI/bge-base-zh"):
        """
        Args:
            session_id: 会话ID，为None时创建新会话
            index_path: 索引存储根目录
            embedding_model: 嵌入模型名称
        """
        self.session_id = session_id or self._generate_session_id()
        self.index_path = Path(index_path) / self.session_id
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # 加载或创建索引
        self.dim = 768  # bge-base-zh 维度
        self._init_index()
        
        # 编码器（复用 Week 5 模型）
        self.encoder = SentenceTransformer(embedding_model)
        
        # 元数据存储（事实文本 + 时间戳 + 重要性）
        self.facts: Dict[str, Dict] = {}  # id -> {text, category, timestamp, importance}
        self._load_metadata()
        
        print(f"[LTM] 会话 {self.session_id} 初始化完成")
    
    def _generate_session_id(self) -> str:
        """生成唯一会话ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"session_{timestamp}_{hashlib.md5(timestamp.encode()).hexdigest()[:8]}"
    
    def _init_index(self):
        """初始化 FAISS 索引"""
        index_file = self.index_path / "facts.index"
        if index_file.exists():
            self.index = faiss.read_index(str(index_file))
            print(f"[LTM] 加载会话 {self.session_id} 的长期记忆索引，包含 {self.index.ntotal} 条事实")
        else:
            # HNSW 索引，适合小规模高维向量（<10万条）
            self.index = faiss.IndexHNSWFlat(self.dim, 16)
            self.index.hnsw.efConstruction = 40
            self.index.hnsw.efSearch = 16
            print(f"[LTM] 为会话 {self.session_id} 创建新的长期记忆索引")
    
    def get_fact_count(self):
        """获取事实数量（用于调试）"""
        return self.index.ntotal if hasattr(self, 'index') else 0
    
    def _load_metadata(self):
        """加载事实元数据"""
        meta_file = self.index_path / "facts.json"
        if meta_file.exists():
            with open(meta_file, 'r', encoding='utf-8') as f:
                self.facts = json.load(f)
    
    def _save_metadata(self):
        """持久化元数据"""
        with open(self.index_path / "facts.json", 'w', encoding='utf-8') as f:
            json.dump(self.facts, f, ensure_ascii=False, indent=2)
        faiss.write_index(self.index, str(self.index_path / "facts.index"))
    
    def add_fact(self, 
                 text: str, 
                 category: str = "general", 
                 importance: float = 1.0) -> str:
        """
        添加长期记忆事实
        
        Args:
            text: 事实描述（如"用户出生于1996年"）
            category: 类别（preference/fact/relationship）
            importance: 重要性（0-1，用于后续筛选）
        """
        # 去重检查（语义相似度 > 0.95 则更新）
        existing_id = self._check_duplicate(text)
        if existing_id:
            self.facts[existing_id]["importance"] = max(
                self.facts[existing_id]["importance"], 
                importance
            )
            self.facts[existing_id]["timestamp"] = datetime.now().isoformat()
            print(f"[LTM] 更新已有事实: {text[:30]}...")
            return existing_id
        
        # 编码并添加
        embedding = self.encoder.encode(text, convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(embedding.reshape(1, -1))
        
        fact_id = hashlib.md5(text.encode()).hexdigest()[:12]
        self.index.add(embedding.reshape(1, -1))
        
        self.facts[fact_id] = {
            "text": text,
            "category": category,
            "timestamp": datetime.now().isoformat(),
            "importance": importance,
            "index": self.index.ntotal - 1  # 在 faiss 中的位置
        }
        
        self._save_metadata()
        print(f"[LTM] 新增事实 [{category}]: {text[:40]}...")
        return fact_id
    
    def _check_duplicate(self, text: str, threshold: float = 0.95) -> Optional[str]:
        """检查是否已存在相似事实"""
        if self.index.ntotal == 0:
            return None
        
        emb = self.encoder.encode(text, convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(emb.reshape(1, -1))
        
        D, I = self.index.search(emb.reshape(1, -1), 1)
        # L2距离越小越相似，需要转换为相似度：sim = 1 / (1 + distance)
        similarity = 1 / (1 + D[0][0])
        if similarity > threshold:
            for fid, meta in self.facts.items():
                if meta.get("index") == I[0][0]:
                    return fid
        return None
    
    def retrieve(self, 
                 query: str, 
                 top_k: int = 3,
                 category_filter: Optional[str] = None) -> List[Dict]:
        """
        检索相关长期记忆
        
        Returns:
            [{"text": "...", "category": "...", "score": 0.92}, ...]
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
                    
                    # 重要性过滤（低于0.3的不返回）
                    if meta["importance"] < 0.3:
                        continue
                    
                    # L2距离转换为相似度：sim = 1 / (1 + distance)
                    similarity = 1 / (1 + dist)
                    
                    results.append({
                        "text": meta["text"],
                        "category": meta["category"],
                        "score": float(similarity),
                        "timestamp": meta["timestamp"]
                    })
                    break
        
        # 按相似度排序
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results[:top_k]
    
    def get_user_profile(self) -> Dict[str, List[str]]:
        """获取用户画像（按类别聚合）"""
        profile = {"preference": [], "fact": [], "relationship": [], "identity": []}
        for meta in self.facts.values():
            cat = meta["category"]
            if cat in profile:
                profile[cat].append(meta["text"])
            else:
                # 未知类别添加到 fact 中
                profile["fact"].append(meta["text"])
        return profile

    def delete_fact(self, fact_id: str):
        """删除事实（软删除，标记重要性为0）"""
        if fact_id in self.facts:
            self.facts[fact_id]["importance"] = 0
            self._save_metadata()
            print(f"[LTM] 删除事实: {fact_id}")
    
    def clear_all(self):
        """清空当前会话的所有长期记忆"""
        self.index.reset()
        self.facts = {}
        self._save_metadata()
        print(f"[LTM] 已清空会话 {self.session_id} 的所有记忆")

# 会话级存储（支持多会话）
_sessions: Dict[str, LongTermMemory] = {}

def get_long_term_memory(session_id: Optional[str] = None) -> LongTermMemory:
    """
    获取或创建长期记忆实例
    
    Args:
        session_id: 会话ID，为None时创建新会话
    
    Returns:
        LongTermMemory实例
    """
    if session_id is None:
        # 创建新会话
        ltm = LongTermMemory()
        _sessions[ltm.session_id] = ltm
        return ltm
    
    # 尝试加载已有会话
    if session_id in _sessions:
        return _sessions[session_id]
    
    # 检查磁盘上是否存在该会话
    ltm = LongTermMemory(session_id=session_id)
    _sessions[session_id] = ltm
    return ltm

def list_sessions(root_path: str = "./long_term_memory") -> List[str]:
    """列出所有已保存的会话"""
    path = Path(root_path)
    if not path.exists():
        return []
    return [p.name for p in path.iterdir() if p.is_dir() and p.name.startswith("session_")]
