# LangChain RAG 进阶教程（Day 2）

## 概述

本目录包含 LangChain RAG（检索增强生成）的进阶教学材料，重点介绍文本分割策略和高级检索器优化技术。

## 核心概念

### 文本分割策略

| 策略 | 原理 | 适用场景 |
|------|------|----------|
| **固定长度分割** | 按字符数硬切 | 简单场景，需要精确控制 chunk 大小 |
| **递归语义分割** | 按分隔符优先级递归切分 | 大多数场景，保持语义完整性 |

### 高级检索器优化

| 技术 | 原理 | 优势 |
|------|------|------|
| **MultiQuery** | LLM 生成同义查询并行检索 | 解决单一查询遗漏问题 |
| **Ensemble** | BM25 + 向量检索加权融合 | 兼顾关键词和语义匹配 |
| **ContextualCompression** | 召回后 LLM 提取关键信息 | 提升检索精度 |
| **FlashrankRerank** | 基于 Flashrank 箾排 | 提升检索质量 |

## 文件说明

### 1. split_test.py - 文本分割策略对比

```python
# 两种分割器对比
fixed_splitter = CharacterTextSplitter(chunk_size=80, separator="")
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=80, 
    chunk_overlap=20,
    separators=["\n\n", "\n", "。", "，", " ", ""]
)
```

**核心要点：**
- 固定长度分割：简单但可能切断语义
- 递归语义分割：优先在自然分隔符处分割，保持语义完整
- 中文文本建议使用 `len` 作为长度函数

### 2. rag.py - RAG 基础流程

**核心要点：**
- 文档加载 → 文本分割 → 向量库构建 → 基础检索
- 使用 `BAAI/bge-base-zh-v1.5` 中文嵌入模型
- 使用 `Chroma` 作为向量数据库

### 3. improved_rag.py - 优化检索器对比

**核心要点：**
- 实现三种优化检索器：MultiQuery、Ensemble、ContextualCompression
- 对比不同检索策略的效果
- 使用 `FlashrankRerank` 进行精排（需额外安装）

### 4. langchain_rag.py - 完整 LCEL RAG 链

**核心要点：**
- 整合高级检索器与 LCEL 管道
- 支持检索策略一键切换
- 包含旁路打印功能（调试检索结果）
- 涵盖 15 个 Agent 领域核心概念文档

### 5. generate_data.py - 测试数据生成

**功能说明：**
- 生成丰富的技术文档用于测试不同检索策略
- 包含 6 个文档文件（3 个 txt、2 个 md、1 个 pdf）
- 包含陷阱文档（看似相关但实际不相关的内容）

**生成的文件：**
| 文件 | 内容 | 用途 |
|------|------|------|
| `lcel_core.txt` | LCEL 核心技术详解 | 语义检索测试 |
| `rag_principles.md` | RAG 原理与流程 | 知识整合测试 |
| `retrieval_strategies.txt` | 检索策略深度对比 | 策略对比测试 |
| `agent_concepts.md` | Agent 智能体技术 | 跨文档关联测试 |
| `similar_concepts.txt` | 相似概念辨析 | 检索精度测试 |
| `trap_documents.txt` | 陷阱文档 | 噪声过滤测试 |
| `langchain_encyclopedia.pdf` | PDF 技术百科 | 多格式支持测试 |

**运行方式：**
```bash
pip install reportlab  # PDF 生成依赖
python generate_data.py
```

### 6. work.py - 综合练习：可配置检索策略的 RAG 系统

**核心特性：**
- 支持运行时通过命令行参数切换检索策略
- 自动加载 `data/` 目录下的 `.txt`、`.md`、`.pdf` 文件
- 向量库持久化到 `./chroma_db_week8`，二次启动秒加载
- 统一 LCEL 管道，四种策略共用同一套 Chain

**支持的检索策略：**

| 策略 | 实现 | 额外依赖 |
|------|------|----------|
| `base` | 基础向量检索 | 无 |
| `multi` | MultiQuery 检索（LLM 生成同义查询） | 需要 LLM |
| `ensemble` | BM25 + 向量混合检索 | 需要 BM25Retriever |
| `compress` | Ensemble + ContextualCompression压缩（LLM 提取关键信息）  | 需要 LLMChainExtractor |
| `rerank` | Ensemble + Flashrank 精排 | 需要 FlashrankRerank |

**运行方式：**
```bash
# 设置 API Key
$env:KIMI_API_KEY="your-api-key"

# 运行不同策略
python work.py --strategy base
python work.py --strategy multi
python work.py --strategy ensemble
python work.py --strategy compress
python work.py --strategy rerank
```

**输出示例：**
```
分割完成：28 块，平均长度 120
加载已有向量库

【问题】怎么把组件串起来

【策略】ensemble | 召回 4 条
  [1] LCEL 使用管道符号 | 连接组件，实现声明式编程... [lcel_core.txt]
  [2] 向量检索和 BM25 检索的加权融合... [retrieval_strategies.txt]
  [3] RunnablePassthrough 用于透传数据... [lcel_core.txt]
  [4] RunnableParallel 支持并行执行... [agent_concepts.md]

【回答】使用 LCEL 的管道符号 | 可以轻松将组件串起来...
```

**核心代码结构：**

```python
# 策略工厂模式
def get_retriever(strategy, vectorstore, llm, chunks):
    base = vectorstore.as_retriever(search_kwargs={"k": 4})
    if strategy == "base":
        return base
    elif strategy == "multi":
        return MultiQueryRetriever.from_llm(retriever=base, llm=llm)
    elif strategy == "compress":
        base = vectorstore.as_retriever(search_kwargs={"k": 3})
        compressor = LLMChainExtractor.from_llm(llm)  
        return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base)
    elif strategy == "rerank":
        base = vectorstore.as_retriever(search_kwargs={"k": 10})
        bm25 = BM25Retriever.from_documents(chunks)
        bm25.k = 10
        ensemble = EnsembleRetriever(retrievers=[bm25, base], weights=[0.3, 0.7])
        compressor = FlashrankRerank(top_n=4)
        return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=ensemble)

# 统一 LCEL Chain
chain = (
    {"context": retriever | log_retrieval | format_docs, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)
```

## 环境配置

### 安装依赖

```bash
pip install langchain langchain_openai langchain_community langchain_huggingface chromadb sentence-transformers
```

### 配置 API Key

```bash
# Windows PowerShell
$env:KIMI_API_KEY="your-api-key"

# Linux/macOS
export KIMI_API_KEY="your-api-key"
```

### 可选：安装 Flashrank（用于精排）

```bash
pip install flashrank
```

## 快速开始

### 运行文本分割测试

```bash
python split_test.py
```

**输出示例：**
```
========================================
【固定长度分割】共 6 块

--- Chunk 0 ---
# LangChain 入门指南

LangChain 是一个用于开发大语言模型应用的框架。它提供了模块化组件，帮助开发者快速构建复杂的

========================================
【递归语义分割】共 5 块

--- Chunk 0 ---
# LangChain 入门指南

LangChain 是一个用于开发大语言模型应用的框架。它提供了模块化组件，帮助开发者快速构建复杂的 AI 流水线。
```

### 运行基础 RAG

```bash
python rag.py
```

### 运行优化检索器对比

```bash
python improved_rag.py
```

### 运行完整 LCEL RAG 链

```bash
python langchain_rag.py
```

## LCEL 高级技巧

### 1. 策略切换

```python
# 只需修改一行即可切换检索策略
SELECTED_RETRIEVER = compression_retriever  # 或 ensemble_retriever / multi_retriever / base_retriever
```

### 2. 旁路打印

```python
def log_retrieval(docs):
    print("\n【检索结果】")
    for i, doc in enumerate(docs):
        print(f"{i+1}. {doc.page_content[:50]}...")
    return docs  # 返回数据继续传递

chain = (
    {"context": SELECTED_RETRIEVER | log_retrieval | format_docs, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)
```

### 3. 使用 def 而非 lambda

```python
# def 定义的函数，错误堆栈显示函数名
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# lambda 定义的函数，错误堆栈只显示 <lambda>
format_docs = lambda docs: "\n\n".join(d.page_content for d in docs)
```

## 常见问题

### Q1: 为什么推荐递归语义分割？

**A:** 递归语义分割优先在自然分隔符（如段落、句子）处分割，保持文本的语义完整性，避免固定长度分割可能切断句子的问题。

### Q2: Ensemble 检索的权重如何设置？

**A:** 根据查询类型调整：
- 语义查询为主：向量权重更高（如 0.7）
- 关键词查询为主：BM25 权重更高（如 0.5）

### Q3: ContextualCompression 和 Ensemble 可以组合使用吗？

**A:** 可以！`langchain_rag.py` 中展示了这种组合：
```python
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=ensemble_retriever  # Ensemble 作为基础检索器
)
```

## 学习路径

1. **基础阶段**：运行 `split_test.py` 理解文本分割差异
2. **进阶阶段**：运行 `rag.py` 掌握 RAG 基础流程
3. **优化阶段**：运行 `improved_rag.py` 对比不同检索策略
4. **综合阶段**：运行 `langchain_rag.py` 体验完整生产级 RAG 系统

## 参考资料

- [LangChain 文本分割文档](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [MultiQuery Retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/multi_query/)
- [Ensemble Retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/ensemble/)
- [Contextual Compression](https://python.langchain.com/docs/modules/data_connection/retrievers/contextual_compression/)