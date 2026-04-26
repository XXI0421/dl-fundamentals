# split_test.py
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_core.documents import Document

# 准备一段结构化的中文文本（包含标题、段落、列表、长句）
sample_text = """# LangChain 入门指南

LangChain 是一个用于开发大语言模型应用的框架。它提供了模块化组件，帮助开发者快速构建复杂的 AI 流水线。

## 核心概念

1. **组件（Components）**：LangChain 将提示模板、模型、解析器等抽象为可复用的组件。
2. **链（Chains）**：通过组合多个组件，形成端到端的工作流。
3. **代理（Agents）**：让模型自主决定调用哪些工具来完成任务。

## LCEL 语法

LCEL 使用管道符号 `|` 来连接组件。例如：prompt | llm | parser。这种语法简洁且类型安全。

## 注意事项

在使用 LangChain 时，需要注意版本兼容性问题。0.1 版本与 0.2 版本在导入路径上有较大差异。建议始终参考官方文档的最新版本。
"""

# 方法 A：固定长度分割（Week 5 风格）
fixed_splitter = CharacterTextSplitter(
    chunk_size=80,
    chunk_overlap=0,
    separator=""  # 没有语义分隔，硬切
)

# 方法 B：递归语义分割（LangChain 工业级）
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=80,
    chunk_overlap=20,
    separators=["\n\n", "\n", "。", "．", ". ", "！", "？", "；", "，", " ", ""],
    length_function=len  # 按字符数计算长度
)

# 执行分割
fixed_chunks = fixed_splitter.split_text(sample_text)
recursive_chunks = recursive_splitter.split_text(sample_text)

print("=" * 40)
print(f"【固定长度分割】共 {len(fixed_chunks)} 块")
for i, chunk in enumerate(fixed_chunks[:]):
    print(f"\n--- Chunk {i} ---")
    print(chunk[:100] + ("..." if len(chunk) > 100 else ""))

print("\n" + "=" * 40)
print(f"【递归语义分割】共 {len(recursive_chunks)} 块")
for i, chunk in enumerate(recursive_chunks[:]):
    print(f"\n--- Chunk {i} ---")
    print(chunk[:100] + ("..." if len(chunk) > 100 else ""))
