import streamlit as st
import os
import sys
from datetime import datetime

# 将父目录加入路径以导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ingest import DocumentIngestor
from retrieve import AdvancedRetriever

# ============= 页面配置 =============
st.set_page_config(
    page_title="Week 5 RAG 检索系统",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============= Session State 初始化 =============
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'index_built' not in st.session_state:
    st.session_state.index_built = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

INDEX_DIR = "./faiss_index"

# ============= 侧边栏：系统配置与文档上传 =============
with st.sidebar:
    st.title("🔧 系统配置")
    
    st.markdown("### 1. 上传文档")
    uploaded_files = st.file_uploader(
        "支持 PDF / Markdown / TXT",
        type=["pdf", "md", "txt"],
        accept_multiple_files=True,
        help="上传后点击'构建索引'，系统将自动分块并建立 HNSW 向量索引"
    )
    
    if uploaded_files and st.button("🚀 构建索引", use_container_width=True):
        with st.spinner("正在处理文档..."):
            try:
                ingestor = DocumentIngestor()
                
                # 保存上传的文件到临时目录
                temp_dir = "./temp_uploads"
                os.makedirs(temp_dir, exist_ok=True)
                
                for file in uploaded_files:
                    file_path = os.path.join(temp_dir, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getvalue())
                    ingestor.ingest_file(file_path)
                
                # 构建索引
                ingestor.build_index(INDEX_DIR)
                
                # 加载到 session state
                st.session_state.retriever = AdvancedRetriever(INDEX_DIR)
                st.session_state.index_built = True
                
                st.success(f"✅ 索引构建完成！共 {len(ingestor.chunks)} 个文本块")
                
            except Exception as e:
                st.error(f"❌ 构建失败: {str(e)}")
    
    st.divider()
    
    st.markdown("### 2. 检索策略")
    use_hyde = st.toggle("启用 HyDE 查询增强", value=True, 
                         help="将短查询扩展为虚拟文档，提升召回率")
    use_rerank = st.toggle("启用 Cross-Encoder 重排序", value=True,
                          help="使用精排模型校正 Top-K 结果（稍慢但更准）")
    
    top_k = st.slider("返回结果数 (Top-K)", 1, 10, 5)
    
    st.divider()
    st.caption(f"⏱️ 当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    if st.session_state.index_built:
        st.success("🟢 系统就绪")

# ============= 主界面：问答区 =============
st.title("🔍 Week 5 RAG 检索系统")
st.caption("基于 HNSW + HyDE + Cross-Encoder 的高级检索引擎")

# 检查索引状态
if not st.session_state.index_built:
    st.info("👈 请先在左侧上传文档并构建索引", icon="ℹ️")
    st.stop()

# 输入区
query = st.text_input("输入你的问题", placeholder="例如：什么是 ReAct？它与 Chain-of-Thought 有什么区别？")

col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    search_btn = st.button("🔍 检索", use_container_width=True, type="primary")
with col2:
    clear_btn = st.button("🗑️ 清空历史", use_container_width=True)

if clear_btn:
    st.session_state.chat_history = []
    st.rerun()

# ============= 检索与展示 =============
if search_btn and query:
    with st.spinner("正在检索..."):
        retriever = st.session_state.retriever
        
        # 执行检索
        results = retriever.retrieve(
            query, 
            use_hyde=use_hyde, 
            use_rerank=use_rerank,
            top_k=top_k
        )
        
        # 记录历史
        st.session_state.chat_history.append({
            "query": query,
            "results": results,
            "config": {"hyde": use_hyde, "rerank": use_rerank, "k": top_k}
        })

# 展示结果（保留历史）
for i, item in enumerate(reversed(st.session_state.chat_history)):
    with st.container(border=True):
        st.markdown(f"**Q: {item['query']}**")
        
        # 配置标签
        config_badges = []
        if item['config']['hyde']:
            config_badges.append("HyDE")
        if item['config']['rerank']:
            config_badges.append("Rerank")
        st.caption(" | ".join(config_badges))
        
        # 结果展示
        for j, doc in enumerate(item['results']):
            score = doc.get('rerank_score', doc['score'])
            source = doc['source']
            text = doc['text']
            
            with st.expander(f"[{j+1}] {source} (相关度: {score:.3f})"):
                st.markdown(f"```\n{text}\n```")
                
                # 高亮显示（简单实现）
                if len(text) > 200:
                    st.markdown(f"*... (共 {len(text)} 字符)*")

# ============= 系统信息（底部） =============
with st.expander("📊 系统技术详情"):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Embedding 模型", "BGE-base-zh")
        st.metric("维度", "768")
    with col2:
        st.metric("索引算法", "HNSW (FAISS)")
        st.metric("M / efConstruction", "16 / 200")
    with col3:
        st.metric("重排序", "BGE-Reranker")
        st.metric("分块大小", "512 tokens")