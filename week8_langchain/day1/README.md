# LangChain LCEL 入门教程

## 概述

本目录包含 LangChain Expression Language (LCEL) 的入门教学材料。LCEL 是 LangChain 的声明式管道语法，用于以简洁的方式组合不同的组件。

## 核心概念

### 什么是 LCEL？

LCEL (LangChain Expression Language) 是一种声明式的管道语法，通过 `|` 操作符连接不同的组件，使数据按顺序流动。

**核心优势：**
- 简洁的声明式语法
- 支持多种调用模式（`invoke`、`stream`、`batch`）
- 自动处理组件间的数据格式转换
- 易于调试和测试

### LCEL 基本组件

| 组件类型 | 作用 | 示例 |
|---------|------|------|
| **PromptTemplate** | 格式化输入为提示词 | `ChatPromptTemplate.from_template()` |
| **LLM** | 调用语言模型 | `ChatOpenAI()` |
| **OutputParser** | 解析模型输出 | `StrOutputParser()` |
| **Retriever** | 文档检索器 | `Chroma.as_retriever()` |
| **RunnablePassthrough** | 透传输入数据 | `RunnablePassthrough()` |
| **RunnableLambda** | 包装自定义函数 | `RunnableLambda(lambda x: x)` |

## 文件说明

### 1. hello.py - LCEL 迷你链演示

```python
chain = prompt | llm | parser
result = chain.invoke({"text": "你好世界"})
```

**核心要点：**
- 使用 `FakeListLLM` 模拟语言模型（无需 API Key）
- 展示最基础的 LCEL 管道结构
- 数据流向：输入 → 提示词格式化 → 模型生成 → 输出解析

### 2. rag.py - RAG 基础示例

**核心要点：**
- 使用 `RunnableLambda` 模拟文档检索器
- 展示字典形式的并行处理
- 理解 `RunnablePassthrough` 的透传机制

### 3. lcel_rag.py - 完整 RAG 链

**核心要点：**
- 使用真实的 Kimi API 和 Chroma 向量数据库
- 展示完整的检索增强生成流程
- 使用 HuggingFace 本地嵌入模型

### 4. work1.py - 练习：检索+格式化

**任务目标：**
- 使用 `RunnableLambda` 模拟检索器
- 实现文档格式化函数
- 用 `|` 连接组件

### 5. work2.py - 练习：完整 LCEL RAG Chain

**任务目标：**
- 准备多个 Document
- 使用 HuggingFaceEmbeddings 和 Chroma 构建检索器
- 定义提示词模板
- 组装完整的 LCEL 链
- 测试 `invoke` 和 `stream` 模式

## 环境配置

### 1. 安装依赖

```bash
pip install langchain langchain_openai langchain_community langchain_huggingface chromadb sentence-transformers
```

### 2. 配置 API Key

在 Moonshot AI 平台获取 API Key：
- 访问：https://platform.moonshot.cn/
- 设置环境变量：
  ```bash
  # Windows PowerShell
  $env:KIMI_API_KEY="your-api-key"
  
  # Linux/macOS
  export KIMI_API_KEY="your-api-key"
  ```

## 快速开始

### 运行基础示例（无需 API Key）

```bash
python hello.py
# 输出: Hello world
```

```bash
python rag.py
# 输出: 上下文: [检索结果] 关于 什么是LCEL 的文档内容...; 问题: 什么是LCEL
```

### 运行完整 RAG 示例（需要 API Key）

```bash
python lcel_rag.py
# 输出: 问题：什么是LCEL
#       回答：LCEL是LangChain的管道语法...
```

```bash
python work2.py
# 输出: 问题：什么是LCEL
#       LCEL是管道语法...
```

## LCEL 调用模式

### 1. invoke() - 同步调用

```python
result = chain.invoke("什么是LCEL")
print(result)  # 完整字符串输出
```

### 2. stream() - 流式调用

```python
for chunk in chain.stream("什么是LCEL"):
    print(chunk, end="", flush=True)  # 逐块输出字符串片段
```

### 3. batch() - 批量调用

```python
results = chain.batch(["什么是LCEL", "什么是Python"])
print(results)  # 返回结果列表
```

## 关键技巧

### 字典并行处理

```python
chain = (
    {
        "context": retriever | format_docs,  # 并行执行
        "question": RunnablePassthrough()     # 并行执行
    }
    | prompt | llm | parser
)
```

**说明：** 字典中的每个键值对会并行执行，然后合并传递给下一个组件。

### 链式组合

```python
# 子链作为组件
retrieval_chain = retriever | (lambda docs: "\n\n".join(d.page_content for d in docs))
full_chain = {"context": retrieval_chain, "question": RunnablePassthrough()} | prompt | llm | parser
```

## 练习任务

### 任务 1：修改 work1.py

**目标：** 修复 work1.py 使其正确工作

```python
# 当前问题：
# 1. retriever 应该返回 List[Document]
# 2. format_docs 应该将文档列表格式化为字符串

# 提示：使用 Document 类
from langchain_core.documents import Document
```

### 任务 2：扩展 work2.py

**目标：** 添加翻译前置步骤

```python
# 在 prompt 前添加一个步骤：将用户问题翻译成英文
# 提示：需要修改 chain 定义，添加翻译子链
```

## 常见问题

### Q1: stream() 的输出是什么样的？

**A:** `stream()` 的输出是字符串片段（string chunks），不是 AIMessage 对象。这是因为使用了 `StrOutputParser()` 将模型输出解析为纯字符串。

### Q2: 如何在 LCEL 中添加自定义步骤？

**A:** 使用 `RunnableLambda` 包装自定义函数：

```python
custom_step = RunnableLambda(lambda x: x.upper())
chain = custom_step | prompt | llm | parser
```

### Q3: 为什么需要 RunnablePassthrough？

**A:** `RunnablePassthrough` 用于透传输入数据，当需要在后续步骤中使用原始输入时非常有用。

## 学习路径

1. **基础阶段**：运行 `hello.py` 和 `rag.py`，理解 LCEL 管道概念
2. **进阶阶段**：运行 `lcel_rag.py`，理解完整的 RAG 流程
3. **练习阶段**：完成 `work1.py` 和 `work2.py` 的练习任务
4. **扩展阶段**：尝试添加新的组件（如翻译、总结等）

## 参考资料

- [LangChain 官方文档](https://python.langchain.com/docs/expression_language/)
- [Moonshot AI 平台](https://platform.moonshot.cn/)
- [HuggingFace Embeddings](https://python.langchain.com/docs/integrations/text_embedding/huggingface/)
- [Chroma 向量数据库](https://www.trychroma.com/)