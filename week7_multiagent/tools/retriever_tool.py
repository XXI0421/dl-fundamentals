import json
import sys
import os
from typing import List, Dict, Optional, Any
from pathlib import Path

# 将 Week 5 路径加入 sys.path（根据你的实际目录结构调整）
week5_path = Path(__file__).parent.parent.parent.parent / "week5_rag" / "rag_system"
if str(week5_path) not in sys.path:
    sys.path.insert(0, str(week5_path))

# 默认索引路径（指向 week5_rag/rag_system/faiss_index）
DEFAULT_INDEX_PATH = str(week5_path / "faiss_index")

try:
    from retrieve import AdvancedRetriever
    from config import EMBEDDING_MODEL, RERANK_MODEL
    WEEK5_AVAILABLE = True
except ImportError as e:
    print(f"Warning: RAG system modules not found: {e}")
    WEEK5_AVAILABLE = False
    AdvancedRetriever = None

from .base import tool

# 单例模式：避免重复加载索引和模型
class RetrieverSingleton:
    _instance = None
    
    @classmethod
    def get_retriever(cls, index_path: str = DEFAULT_INDEX_PATH) -> Optional[AdvancedRetriever]:
        if not WEEK5_AVAILABLE:
            return None
        
        if cls._instance is None:
            try:
                print(f"[RetrieverTool] 正在加载 RAG system 索引: {index_path}")
                cls._instance = AdvancedRetriever(index_path=index_path)
                print(f"[RetrieverTool] 索引加载完成，维度: {cls._instance.dim}")
            except Exception as e:
                print(f"[RetrieverTool] 索引加载失败: {e}")
                return None
        return cls._instance

@tool
def retriever_tool(
    query: str, 
    top_k: int = 5, 
    use_hyde: bool = True, 
    use_rerank: bool = True,
    index_path: str = DEFAULT_INDEX_PATH
) -> str:
    """
    从私有知识库（FAISS 向量索引）中检索相关文档。
    适用于查询专业术语、历史文档内容、技术概念解释等。
    
    参数:
        query: 检索查询语句（自然语言）
        top_k: 返回结果数量（默认5条，最多10条）
        use_hyde: 是否使用 HyDE 查询增强（默认开启，提升语义召回）
        use_rerank: 是否使用 Cross-Encoder 重排序（默认开启，提升精度）
        index_path: 索引目录路径（默认 ./faiss_index）
    
    返回:
        格式化的检索结果，包含内容片段、来源引用 [^N^]、相关度分数
        如果未找到结果，返回提示信息建议调整查询词
    """
    retriever = RetrieverSingleton.get_retriever(index_path)
    
    if retriever is None:
        # Mock 模式（用于测试 Week 6 逻辑，不依赖 Week 5 索引）
        mock_results = [
            {"text": f"这是关于 '{query}' 的模拟检索结果。在实际环境中，请确保 Week 5 索引已构建。", 
             "source": "mock_data.md", 
             "score": 0.95}
        ]
        return _format_results(mock_results)
    
    try:
        # 调用 Week 5 AdvancedRetriever
        results = retriever.retrieve(
            query=query,
            use_hyde=use_hyde,
            use_rerank=use_rerank,
            top_k=min(top_k, 10)  # 限制最大数量
        )
        
        if not results:
            return ("未检索到相关文档。建议：\n"
                    "1. 尝试使用更通用的关键词（如用 'AI Agent' 代替 'ReAct Agent'）\n"
                    "2. 检查知识库是否已索引相关文档（运行 RAG system ingest.py）\n"
                    "3. 关闭 HyDE 增强再试（use_hyde=false）")
        
        return _format_results(results)
        
    except Exception as e:
        return f"检索系统错误：{str(e)}。请检查索引文件是否存在且未损坏。"

def _format_results(docs: List[Dict]) -> str:
    """格式化检索结果为 Agent 友好的字符串（带引用标记）"""
    formatted = []
    
    for i, doc in enumerate(docs, 1):
        text = doc.get('text', '').strip()
        source = doc.get('source', '未知来源')
        
        # 获取分数（优先使用重排序分数，其次是向量分数）
        score = doc.get('rerank_score', doc.get('score', 0.0))
        
        # 截断过长文本（保留前 300 字符，避免撑爆上下文）
        if len(text) > 300:
            text = text[:300] + "..."
        
        # 格式：[^1^] 相关度: 0.95 | 来源: xxx.md | 内容: ...
        entry = (f"[^{i}^] 相关度: {score:.3f} | 来源: {source}\n"
                 f"    {text}")
        formatted.append(entry)
    
    # 添加引用说明
    footer = "\n（请使用 [^N^] 格式引用上述信息）"
    
    return "\n\n".join(formatted) + footer

@tool
def knowledge_base_status(index_path: str = DEFAULT_INDEX_PATH) -> str:
    """
    检查知识库状态（文档数量、索引健康度）。
    用于诊断检索工具是否可用。
    """
    if not WEEK5_AVAILABLE:
        return "RAG system 模块未加载，无法检查知识库状态"  
    
    try:
        retriever = RetrieverSingleton.get_retriever(index_path)
        if retriever is None:
            return "知识库索引加载失败，请检查路径和文件完整性"
        
        num_docs = len(retriever.chunks) if hasattr(retriever, 'chunks') else '未知'
        return (f"知识库状态正常：\n"
                f"- 索引路径: {index_path}\n"
                f"- 文档块数: {num_docs}\n"
                f"- 嵌入模型: {EMBEDDING_MODEL}\n"
                f"- 重排模型: {RERANK_MODEL}")
    except Exception as e:
        return f"状态检查失败: {e}"
