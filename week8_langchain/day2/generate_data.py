# generate_data.py - 生成测试数据脚本
# 生成丰富的技术文档，用于测试不同检索策略的召回效果差异

import os

# 确保 data 目录存在
os.makedirs("data", exist_ok=True)

# ========== 1. 生成 LCEL 专题文档 ==========
lcel_content = """# LCEL 核心技术文档

## 1.1 LCEL 简介

LCEL（LangChain Expression Language）是 LangChain 0.1 版本引入的声明式管道语法。它允许开发者使用管道符号 | 来组合各种组件，构建复杂的 AI 工作流。

## 1.2 LCEL 核心特性

- 声明式组合：通过管道符号连接组件
- 类型安全：自动推断输入输出类型
- 支持多种调用模式：invoke、stream、batch
- 内置重试和错误处理

## 1.3 LCEL 基本模式

```python
# 基础链
chain = prompt | llm | parser

# 带检索的链
chain = {"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | parser
```

## 1.4 LCEL vs 传统 Chain

传统 Chain 需要手动编写循环和错误处理，而 LCEL 通过声明式语法自动处理这些细节。LCEL 还支持并行执行和流式输出。

## 1.5 LCEL 与 Runnable

LCEL 基于 Runnable 接口构建。所有 LCEL 组件都实现了 Runnable 接口，包括：
- RunnableLambda：包装函数
- RunnablePassthrough：透传数据
- RunnableParallel：并行执行
- RunnableSequence：顺序执行
"""

with open("data/lcel_core.txt", "w", encoding="utf-8") as f:
    f.write(lcel_content)

# ========== 2. 生成 RAG 专题文档 ==========
rag_content = """# RAG 检索增强生成

## 2.1 RAG 基本概念

检索增强生成（Retrieval-Augmented Generation）是一种将外部知识库与大语言模型结合的技术。它在生成回答前，先从文档库中检索相关信息，然后将这些信息作为上下文输入给模型。

## 2.2 RAG 核心流程

1. 文档加载：从多种来源加载文档
2. 文本分割：将长文档切分为合适大小的 chunk
3. 向量嵌入：将文本转换为向量
4. 向量存储：存储到向量数据库
5. 查询检索：根据用户问题检索相关文档
6. 生成回答：基于检索结果生成回答

## 2.3 RAG 与传统问答的区别

传统问答系统依赖模型的内部知识，可能产生幻觉。RAG 通过检索外部知识库，确保回答的准确性和时效性。

## 2.4 RAG 的关键挑战

- 检索质量：如何找到真正相关的文档
- 上下文管理：如何处理长上下文
- 多模态支持：如何处理图片、表格等非文本内容
"""

with open("data/rag_principles.md", "w", encoding="utf-8") as f:
    f.write(rag_content)

# ========== 3. 生成检索策略对比文档 ==========
retrieval_content = """# 检索策略深度对比

## 3.1 基础向量检索

基础向量检索使用余弦相似度匹配，适合语义查询。但对于关键词匹配可能效果不佳。

## 3.2 MultiQuery 检索

MultiQuery 使用 LLM 自动生成多个同义查询，然后合并去重。这可以有效解决单一查询可能遗漏相关文档的问题。

## 3.3 BM25 检索

BM25 是一种基于词频的检索算法，擅长关键词匹配。但对于语义理解能力较弱。

## 3.4 Ensemble 检索

Ensemble 将多种检索器组合，通过加权融合提升效果。常见组合：BM25 + 向量检索。

## 3.5 Contextual Compression

Contextual Compression 在召回后使用 LLM 或精排模型提取关键信息，减少噪音。

## 3.6 策略选择指南

| 查询类型 | 推荐策略 | 原因 |
|---------|---------|------|
| 关键词查询 | BM25 或 Ensemble | 关键词匹配准确 |
| 语义查询 | 向量检索或 MultiQuery | 理解语义关系 |
| 复杂查询 | Ensemble + Compression | 综合多种信息 |
"""

with open("data/retrieval_strategies.txt", "w", encoding="utf-8") as f:
    f.write(retrieval_content)

# ========== 4. 生成 Agent 概念文档 ==========
agent_content = """# Agent 智能体技术

## 4.1 Agent 定义

Agent 是能够感知环境、做出决策并执行行动的 AI 系统。它通常包含以下核心组件：

## 4.2 Agent 核心能力

- **Tool Use**: 调用外部工具的能力
- **Planning**: 任务规划和分解能力
- **Reflection**: 自我反思和纠错能力
- **Memory**: 长期记忆和上下文管理

## 4.3 Agent 架构模式

- **ReAct**: 推理与行动交替进行
- **Plan-and-Execute**: 先规划后执行
- **Reflexion**: 基于反思的迭代改进

## 4.4 Agent 与 RAG 的关系

Agent 可以使用 RAG 作为其知识库组件，通过检索获取外部信息来辅助决策。

## 4.5 Agent 应用场景

- 代码生成和调试
- 数据分析和报告
- 多步任务规划
- 自动研究和总结
"""

with open("data/agent_concepts.md", "w", encoding="utf-8") as f:
    f.write(agent_content)

# ========== 5. 生成相似概念区分文档（用于测试检索精度）==========
similar_concepts = """# 相似概念辨析

## 5.1 LCEL vs Chain

- **LCEL**: 声明式管道语法，自动处理类型和错误
- **Chain**: 传统链式调用，需要手动管理流程

## 5.2 RAG vs Fine-tuning

- **RAG**: 检索外部知识，无需重新训练
- **Fine-tuning**: 在特定数据上微调模型

## 5.3 Vector DB vs Traditional DB

- **Vector DB**: 存储向量，支持相似度检索
- **Traditional DB**: 存储结构化数据，支持 SQL 查询

## 5.4 Embedding vs Encoding

- **Embedding**: 将文本转换为语义向量
- **Encoding**: 将文本转换为字符编码

## 5.5 Token vs Embedding

- **Token**: 文本的最小单位（如单词片段）
- **Embedding**: 文本的语义向量表示

## 5.6 Context Window vs Embedding Dimension

- **Context Window**: 模型能处理的最大文本长度
- **Embedding Dimension**: 向量的维度（如 768、1024）
"""

with open("data/similar_concepts.txt", "w", encoding="utf-8") as f:
    f.write(similar_concepts)

# ========== 6. 生成陷阱文档（看似相关但实际不相关）==========
trap_documents = """# 技术文档（陷阱数据）

## 6.1 LLM Chain 介绍

LLM Chain 是一种将多个 LLM 调用串联起来的模式。例如，可以先用一个 LLM 生成问题，再用另一个 LLM 回答问题。

## 6.2 区块链技术

区块链是一种分布式账本技术，用于记录交易和数据。它具有去中心化、不可篡改等特点。

## 6.3 供应链管理

供应链管理涉及原材料采购、生产、运输和销售的全过程优化。

## 6.4 管道工程

管道工程涉及石油、天然气等流体的输送系统设计和建设。

## 6.5 链条传动

链条传动是一种机械传动方式，常用于自行车、摩托车等交通工具。

## 6.6 文本压缩算法

文本压缩算法如 Huffman 编码、LZ77 等，用于减小文件大小。
"""

with open("data/trap_documents.txt", "w", encoding="utf-8") as f:
    f.write(trap_documents)

# ========== 7. 生成 PDF 文档（使用 reportlab 支持中文）==========
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    
    # 获取系统字体目录
    font_folder = None
    known_paths = [
        "C:\\Windows\\Fonts",
        "C:\\WINNT\\Fonts",
        os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts"),
    ]
    for path in known_paths:
        if os.path.exists(path):
            font_folder = path
            break
    
    if font_folder is None:
        print("⚠ 无法获取字体目录，跳过 PDF 生成")
    else:
        # 注册中文字体
        chinese_fonts = [
            ("SimHei", "simhei.ttf"),
            ("SimSun", "simsun.ttc"),
            ("Microsoft YaHei", "msyh.ttc"),
            ("KaiTi", "kaiti.ttf"),
        ]
        font_name = None
        
        for name, filename in chinese_fonts:
            font_full_path = os.path.join(font_folder, filename)
            if os.path.exists(font_full_path):
                pdfmetrics.registerFont(TTFont(name, font_full_path))
                font_name = name
                break
        
        if font_name is None:
            font_name = "Helvetica"
            print("⚠ 未找到中文字体，可能无法正确显示中文")
        
        pdf = SimpleDocTemplate("data/langchain_encyclopedia.pdf", pagesize=A4)
        styles = {
            'title': ParagraphStyle(
                'Title',
                fontName=font_name,
                fontSize=16,
                bold=True,
                alignment=TA_CENTER,
                spaceAfter=20
            ),
            'heading': ParagraphStyle(
                'Heading',
                fontName=font_name,
                fontSize=14,
                bold=True,
                alignment=TA_LEFT,
                spaceAfter=10
            ),
            'body': ParagraphStyle(
                'Body',
                fontName=font_name,
                fontSize=12,
                alignment=TA_LEFT,
                spaceAfter=15,
                leading=18
            )
        }
        
        story = []
        story.append(Paragraph("LangChain 技术百科", styles['title']))
        story.append(Spacer(1, 10))
        
        story.append(Paragraph("LCEL 管道语法", styles['heading']))
        story.append(Paragraph("LCEL 使用管道符号 | 连接组件，实现声明式编程。核心组件包括 RunnablePassthrough、RunnableParallel、RunnableSequence 等。", styles['body']))
        
        story.append(Paragraph("RAG 检索优化", styles['heading']))
        story.append(Paragraph("MultiQuery 通过生成同义查询提升召回率，Ensemble 结合多种检索器优势，ContextualCompression 精排压缩提升精度。", styles['body']))
        
        story.append(Paragraph("Agent 核心能力", styles['heading']))
        story.append(Paragraph("智能体需要具备工具调用、任务规划、自我反思和记忆管理能力。ReAct 是最经典的 Agent 架构之一。", styles['body']))
        
        story.append(Paragraph("向量数据库选择", styles['heading']))
        story.append(Paragraph("Chroma 适合开发测试，FAISS 适合大规模数据，Pinecone 适合生产环境。BGE 是优秀的中文嵌入模型。", styles['body']))
        
        story.append(Paragraph("文本分割技巧", styles['heading']))
        story.append(Paragraph("使用 RecursiveCharacterTextSplitter，chunk_size 设置为 100-200 字符，保留适当重叠确保上下文连贯。", styles['body']))
        
        pdf.build(story)
        print("✓ PDF 文档生成成功")
except ImportError as e:
    # 如果没有 reportlab，跳过 PDF 生成
    print(f"⚠ 跳过 PDF 生成: {e}")
    print("  安装命令：pip install reportlab")

print("✅ 数据生成完成！")
print("已创建文件：")
print("- data/lcel_core.txt          # LCEL 核心技术")
print("- data/rag_principles.md      # RAG 原理")
print("- data/retrieval_strategies.txt  # 检索策略对比")
print("- data/agent_concepts.md      # Agent 概念")
print("- data/similar_concepts.txt   # 相似概念辨析")
print("- data/trap_documents.txt     # 陷阱文档（测试检索精度）")
print("- data/langchain_encyclopedia.pdf  # PDF 技术百科")
print("\n📝 测试查询建议：")
print("1. 'LCEL 是什么' - 基础关键词查询")
print("2. '怎么把组件串起来' - 语义描述查询")
print("3. 'RAG 和 Agent 的关系' - 需要综合多文档")
print("4. '链条传动原理' - 测试噪声过滤")