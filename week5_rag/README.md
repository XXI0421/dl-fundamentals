# Week 5: RAG 检索系统

## 系统介绍

Week 5 RAG 检索系统是一个基于 **HNSW + HyDE + Cross-Encoder** 的高级检索增强生成（Retrieval-Augmented Generation）系统，专为文档智能检索和问答设计。

## 功能特性

### 🚀 核心功能
- **智能文档处理**：支持 PDF、Markdown、TXT 格式文档
- **动态分块**：根据文件类型和内容自动调整分块策略
- **高效检索**：使用 HNSW 索引实现快速向量检索
- **HyDE 增强**：支持规则版和 Kimi API 增强版 HyDE
- **重排序**：使用 Cross-Encoder 进行精排，提升结果质量
- **智能回答**：基于检索结果生成总结性回答
- **用户友好界面**：Streamlit 前端，操作简单直观

### 🔧 技术特性
- **多语言支持**：自动检测中英文并使用合适的分块策略
- **性能优化**：模型单例模式，避免重复加载
- **错误处理**：完善的错误处理和回退机制
- **可扩展性**：模块化设计，易于扩展和定制
- **Kimi API 集成**：支持使用 Kimi API 进行增强

## 学习路径

### 阶段 1：理解 Embedding 与基础检索

1. **准备语料库**
   ```bash
   python prepare_corpus.py
   ```
   - 从 Hugging Face 加载 Wikipedia 文档和 MS MARCO 查询
   - 生成 `corpus.jsonl` 和 `queries.jsonl` 文件
   - 理解语料库构建和查询集准备的重要性

2. **构建向量数据库**
   ```bash
   python build_db.py
   ```
   - 使用 BGE 模型编码文档
   - 构建 HNSW 索引
   - 保存索引、向量和元数据
   - 理解 Embedding 技术和向量数据库的基本原理

3. **性能基准测试**
   ```bash
   python vector_db_benchmark.py
   ```
   - 测试不同 HNSW 参数的性能
   - 分析延迟与召回率的权衡
   - 生成性能报告和可视化图表
   - 理解 HNSW 算法的工作原理

### 阶段 2：理解更高级的 RAG 技术

1. **高级 RAG 示例**
   ```bash
   python advanced_rag.py
   ```
   - 体验更高级的 RAG 技术
   - 了解不同的检索和增强策略
   - 学习高级 RAG 架构和优化方法

2. **演示脚本**
   ```bash
   python demo.py
   ```
   - 运行预设的演示场景
   - 快速体验系统的核心功能
   - 理解 RAG 系统的完整工作流程

3. **Kimi API 集成**
   - 了解如何集成外部 LLM 增强 RAG 系统
   - 学习 HyDE 技术的高级应用
   - 掌握基于检索结果的智能回答生成

### 阶段 3：直接运用实战

1. **安装依赖**
   ```bash
   # 安装基本依赖
   pip install streamlit faiss-cpu sentence-transformers pypdf requests
   ```

2. **启动系统**
   ```bash
   cd rag_system
   streamlit run app.py
   ```
   系统将在浏览器中打开，默认地址为 `http://localhost:8501`。

3. **实战应用步骤**
   1. **上传文档**：在左侧侧边栏上传 PDF、Markdown 或 TXT 文件
   2. **构建索引**：点击 "🚀 构建索引" 按钮，系统会自动处理文档并构建 HNSW 索引
   3. **配置检索策略**：选择是否启用 HyDE 和重排序，设置返回结果数
   4. **配置 Kimi API**（可选）：输入 API Key 并启用 Kimi API
   5. **执行检索**：在主界面输入问题，点击 "🔍 检索" 按钮
   6. **查看结果**：查看生成的回答和检索到的相关文档

4. **实战项目**
   - **个人知识库**：上传自己的文档，构建专属知识库
   - **问答系统**：配置 Kimi API，构建智能问答系统
   - **性能优化**：调整检索参数，优化系统性能和准确性
   - **扩展应用**：基于 RAG 系统构建更复杂的应用场景

## 技术架构

### 系统组件

| 组件 | 功能 | 实现文件 |
|------|------|----------|
| 文档摄取器 | 解析文档、智能分块、生成嵌入 | `ingest.py` |
| 检索引擎 | 向量检索、HyDE 增强、重排序 | `retrieve.py` |
| 配置管理 | 系统参数和模型配置 | `config.py` |
| 前端界面 | 用户交互和结果展示 | `app.py` |

### 核心流程

1. **文档处理**：解析上传的文档，根据文件类型和内容动态分块
2. **向量编码**：使用 BGE 模型将文本块编码为向量
3. **索引构建**：使用 FAISS HNSW 算法构建高效向量索引
4. **查询处理**：接收用户查询，使用 HyDE 增强查询
5. **向量检索**：在 HNSW 索引中检索相关文本块
6. **重排序**：使用 Cross-Encoder 对检索结果进行精排
7. **回答生成**：基于检索结果生成总结性回答（支持 Kimi API）
8. **结果展示**：在前端界面展示生成的回答和检索结果

## 配置选项

### 系统配置

配置文件：`config.py`

```python
# 模型配置
EMBEDDING_MODEL = 'BAAI/bge-base-zh'  # 或 bge-base-en-v1.5
RERANK_MODEL = 'BAAI/bge-reranker-base'

# HNSW 参数
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 100

# 分块参数
CHUNK_SIZE = 150
CHUNK_OVERLAP = 30

# 检索配置
TOP_K_RETRIEVE = 50
TOP_K_RERANK = 10

# Kimi API 配置
KIMI_API_KEY = os.getenv('KIMI_API_KEY', 'your_kimi_api_key_here')
KIMI_API_URL = 'https://api.moonshot.cn/v1/chat/completions'
KIMI_MODEL = 'moonshot-v1-8k'
ENABLE_KIMI_API = False
```

### 前端配置

- **上传文档**：支持 PDF、Markdown、TXT 格式
- **检索策略**：
  - 启用 HyDE 查询增强
  - 启用 Cross-Encoder 重排序
  - 设置返回结果数 (1-10)
- **Kimi API 配置**：
  - 输入 API Key
  - 启用/禁用 Kimi API

## 核心功能详解

### 1. 智能分块

系统会根据文件类型和内容动态调整分块策略：

- **PDF 文件**：使用较大的分块大小（200 tokens）
- **Markdown 文件**：使用中等分块大小（180 tokens）
- **TXT 文件**：使用标准分块大小（150 tokens）
- **长文本**：自动增大分块大小
- **短文本**：自动减小分块大小
- **中文文本**：使用稍大的分块大小（考虑到中文字符密度）

### 2. HyDE 增强

支持两种 HyDE 模式：

- **规则版**：基于关键词扩展的简单 HyDE
- **Kimi API 版**：使用 Kimi API 生成高质量虚拟文档

### 3. 重排序

使用 BGE-Reranker 对检索结果进行精排，提升结果质量：

- 对每个检索结果计算与查询的相关度
- 按相关度对结果进行排序
- 返回最相关的 Top-K 结果

### 4. 智能回答

基于检索到的文档生成总结性回答：

- **Kimi API 版**：使用 Kimi API 生成详细、准确的回答
- **回退版**：当 Kimi API 不可用时，生成简单的总结

## 性能优化

### 1. 模型加载优化

使用单例模式管理模型实例，避免重复加载：

```python
class ModelManager:
    _instance = None
    _models = {}
    
    @classmethod
    def get_encoder(cls, model_name):
        if model_name not in cls._models:
            cls._models[model_name] = SentenceTransformer(model_name)
        return cls._models[model_name]
```

### 2. 存储优化

使用 NumPy 数组存储文本块和来源信息，提升加载速度：

```python
# 分离存储
np.save(f"{index_path}/chunks.npy", np.array(self.chunks, dtype=object))
np.save(f"{index_path}/sources.npy", np.array(self.sources, dtype=object))
```

### 3. 检索优化

使用 HNSW 索引实现快速向量检索：

```python
# HNSW 索引
index = faiss.IndexHNSWFlat(self.dim, HNSW_M)
index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
index.add(embeddings)
```

## 扩展建议

1. **添加更多文档格式支持**：
   - Word 文档 (.docx)
   - Excel 表格 (.xlsx)
   - PowerPoint 演示文稿 (.pptx)

2. **增强文档解析能力**：
   - 支持表格、图表的解析
   - 支持图片的 OCR 识别
   - 支持结构化数据的提取

3. **添加多语言支持**：
   - 自动检测文档语言
   - 根据语言选择合适的模型

4. **增强检索能力**：
   - 添加关键词检索（BM25）
   - 实现混合检索（向量 + 关键词）
   - 添加语义重写功能

5. **优化用户体验**：
   - 添加查询历史记录
   - 实现查询建议
   - 添加结果高亮显示

## 示例使用

### 示例 1：基本检索

1. 上传 `test.md` 文件
2. 构建索引
3. 输入查询："什么是 ReAct？"
4. 查看生成的回答和检索结果

### 示例 2：使用 Kimi API

1. 在侧边栏输入 Kimi API Key
2. 启用 Kimi API
3. 输入查询："ReAct 与 Chain-of-Thought 有什么区别？"
4. 查看 Kimi API 生成的详细回答

### 示例 3：处理 PDF 文档

1. 上传 PDF 文档
2. 系统会自动调整分块策略
3. 输入查询，查看检索结果

## 故障排除

### 常见问题

1. **PDF 解析失败**：
   - 确保已安装 `pypdf` 库
   - 检查 PDF 文件是否损坏

2. **Kimi API 调用失败**：
   - 检查 API Key 是否正确
   - 确保网络连接正常
   - 查看控制台错误信息

3. **检索结果质量差**：
   - 尝试调整分块大小
   - 启用 HyDE 和重排序
   - 检查文档内容是否相关

4. **系统运行缓慢**：
   - 减少返回结果数
   - 考虑使用更轻量级的模型
   - 检查系统资源使用情况

## 技术栈

- **后端**：Python 3.8+
- **向量数据库**：FAISS (HNSW)
- **嵌入模型**：BAAI/bge-base-zh
- **重排序模型**：BAAI/bge-reranker-base
- **前端**：Streamlit
- **API 集成**：Kimi API

## 项目结构

```
week5_rag/
├── rag_system/              # 主系统目录
│   ├── app.py               # Streamlit 前端
│   ├── config.py            # 配置管理
│   ├── ingest.py            # 文档摄取器
│   ├── retrieve.py          # 检索引擎
│   ├── faiss_index/         # 索引存储目录
│   ├── temp_uploads/        # 临时文件目录
│   ├── test.md              # 测试文件
│   ├── test_en.md           # 英文测试文件
│   ├── test_en.pdf          # PDF 测试文件
│   └── create_test_pdf.py   # PDF 生成脚本
├── advanced_rag.py          # 高级 RAG 示例
├── agent_data.py            # 代理数据（领域知识）
├── build_db.py              # 构建向量数据库
├── corpus.jsonl             # 语料库（Wikipedia 文档）
├── demo.py                  # 演示脚本
├── hnsw_benchmark_robust.png # 性能基准测试图表
├── prepare_corpus.py        # 语料库准备脚本
├── queries.jsonl            # 查询集（MS MARCO 真实用户查询）
└── vector_db_benchmark.py   # 向量数据库基准测试
```

## 脚本说明

### 1. 语料库准备脚本 (`prepare_corpus.py`)
- **功能**：从 Hugging Face 加载 Wikipedia 文档和 MS MARCO 查询
- **输出**：`corpus.jsonl`（语料库）和 `queries.jsonl`（查询集）
- **用途**：为基准测试和系统测试提供标准数据

### 2. 向量数据库构建脚本 (`build_db.py`)
- **功能**：使用 BGE 模型编码文档并构建 HNSW 索引
- **输入**：`agent_data.py` 中的领域知识
- **输出**：`agent_db/` 目录（包含索引、向量和元数据）
- **用途**：构建领域专用的向量数据库

### 3. 性能基准测试脚本 (`vector_db_benchmark.py`)
- **功能**：测试不同 HNSW 参数的性能
- **输入**：`corpus.jsonl` 和 `queries.jsonl`
- **输出**：`hnsw_benchmark_robust.png`（性能图表）
- **用途**：分析 HNSW 算法性能，优化参数配置

### 4. 高级 RAG 示例 (`advanced_rag.py`)
- **功能**：演示更高级的 RAG 技术
- **用途**：了解不同的检索和增强策略

### 5. 演示脚本 (`demo.py`)
- **功能**：运行预设的演示场景
- **用途**：快速体验系统的核心功能

### 6. PDF 生成脚本 (`create_test_pdf.py`)
- **功能**：生成测试用的 PDF 文件
- **用途**：测试 PDF 文档解析功能

## 许可证

MIT License

## 致谢

- [FAISS](https://github.com/facebookresearch/faiss) - 高效向量检索库
- [Sentence-BERT](https://github.com/UKPLab/sentence-transformers) - 句子嵌入模型
- [BAAI](https://github.com/FlagAI-Open/BGE) - BGE 模型
- [Streamlit](https://streamlit.io/) - 快速构建 Web 应用
- [Moonshot AI](https://platform.moonshot.cn/) - Kimi API

---

**Week 5 RAG 检索系统** - 一个功能强大、易于使用的文档检索和问答系统，为您的知识库提供智能访问能力。