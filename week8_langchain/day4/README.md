# LangChain LangGraph RAG 进阶教程（Day 4）

## 概述

本目录包含 LangGraph 的高级 RAG 工作流教学材料，重点介绍**检索-生成-评估-重写**的闭环流程。通过 LangGraph 的可视化状态机，实现智能的自动优化检索策略。

## 核心概念

### LangGraph 工作流

| 组件 | 说明 |
|------|------|
| **State** | 所有节点共享的状态字典，存储问题、上下文、答案、评估结果等 |
| **Node** | 图中的节点，每个节点是一个函数，接收 state 并返回 state 更新 |
| **Edge** | 节点之间的连接，可以是无条件边或条件边 |
| **Conditional Edge** | 根据状态值动态决定下一个节点 |
| **END** | 图的终止节点 |

### RAG 评估循环流程

```
用户输入 → retrieve(检索) → generate(生成) → grade(评估)
                                              ↓
                         ┌───────────────────┴───────────────────┐
                         │                                     │
                    【足够好？】                            【不够好？】
                         │                                     │
                    返回答案                              rewrite_query(重写)
                                                              ↓
                                                        retrieve(重新检索)
```

### 评估策略

| 策略 | 说明 |
|------|------|
| **硬编码规则** | 根据回答中是否包含"信息不足"等关键词快速判断 |
| **LLM 精排评估** | 使用 LLM 深度分析回答是否充分回答了问题 |

## 文件说明

### 1. simulate.py - LangGraph 基础模拟

**功能说明：**
- 演示 LangGraph 的基本概念和工作流程
- 模拟检索、生成、评估、重写的完整循环
- 帮助理解状态机和条件路由机制

**核心代码：**

```python
# 定义 State 结构（所有节点共享）
class GraphState(TypedDict):
    question: str          # 用户问题
    context: str          # 检索到的上下文
    answer: str            # 生成的答案
    grade: Literal["yes", "no"]  # 评估结果
    loop_count: int          # 循环计数器

# 定义条件路由
def decide_next(state: GraphState) -> Literal["rewrite", "__end__"]:
    if state["grade"] == "yes":
        return "__end__"
    if state.get("loop_count", 0) > 3:
        return "__end__"
    return "rewrite"
```

**运行方式：**
```bash
python simulate.py
```

### 2. langgraph_rag.py - 完整的 LangGraph RAG 系统

**功能说明：**
- 整合真实的向量库检索和 LLM 生成
- 实现"检索→生成→评估→重写"闭环
- 支持最大迭代次数限制（防止无限循环）

**核心组件：**

| 节点 | 功能 |
|------|------|
| `retrieve` | 从向量库检索相关文档 |
| `generate` | 基于上下文生成回答 |
| `grade_answer` | 评估回答质量（硬编码 + LLM 双重判断） |
| `rewrite_query` | 根据评估结果重写检索 Query |

**评估机制：**

```python
def grade_answer(state: RAGState):
    answer = state.get("answer", "")
    
    # 第一层：硬编码规则
    insufficient_signals = ["信息不足", "不知道", "无法回答", "没有相关", "未找到"]
    if any(signal in answer for signal in insufficient_signals):
        return {"grade": "insufficient"}
    
    # 第二层：LLM 精排评估
    # 使用 LLM 判断回答是否充分
```

**条件路由：**

```python
def route(state: RAGState):
    if state.get("grade") == "sufficient":
        return END  # 回答足够好，结束
    if state.get("loop_count", 0) >= 2:  # 最多重写 2 次
        return END
    return "rewrite"  # 需要重写查询
```

**运行方式：**
```bash
python langgraph_rag.py
```

**输出示例：**
```
==================================================
【问题】LangGraph 和 CrewAI 有什么区别
【检索】LangGraph 和 CrewAI 有什么区别...
【生成】信息不足...
【评估】insufficient (硬编码命中)
【改写】LangGraph 和 CrewAI 有什么区别... → LangGraph与CrewAI的核心差异对比...
【检索】LangGraph与CrewAI的核心差异对比...
【生成】LangGraph 和 CrewAI 都是用于构建多智能体系统的框架...
【评估】sufficient (LLM判断)

【最终答案】LangGraph 和 CrewAI 都是用于构建多智能体系统的框架，但它们在设计理念和应用场景上有所不同...
【迭代次数】1
```

## 环境配置

### 安装依赖

```bash
pip install langchain langchain_openai langchain_huggingface langchain_chroma langgraph
```

### 配置 API Key

```bash
# Windows PowerShell
$env:KIMI_API_KEY="your-api-key"

# Linux/macOS
export KIMI_API_KEY="your-api-key"
```

### 准备向量库

```bash
# 使用 day2 生成的数据构建向量库
cd ../day2
python generate_data.py
python work.py --strategy base  # 构建向量库

# 复制向量库到 day4
cp -r chroma_db_week8 ../day4/chroma_db
```

## 快速开始

### 1. 运行模拟示例

```bash
python simulate.py
```

**输出示例：**
```
========================================
【测试 B】复杂问题（触发循环）
【节点: retrieve】问题: 为什么需要 LangGraph
【节点: generate】上下文: 关于 为什么需要 LangGraph 的知识...
【节点: grade】评估答案: 为什么需要 LangGraph 很重要...
【节点: rewrite】重写问题: 为什么需要 LangGraph
【节点: retrieve】问题: 基于已有回答，请更深入说明：为什么需要 LangGraph 的核心机制
【节点: generate】上下文: 关于 基于已有回答...
【节点: grade】评估答案: 基于已有回答，请更深入说明...
【节点: rewrite】重写问题: 基于已有回答，请更深入说明：为什么需要 LangGraph 的核心机制
...
```

### 2. 运行完整 RAG 系统

```bash
python langgraph_rag.py
```

## LangGraph 核心概念详解

### 状态管理

```python
from typing import TypedDict, Literal

class RAGState(TypedDict):
    question: str
    context: str
    answer: str
    grade: Literal["sufficient", "insufficient"]
    loop_count: int
```

**关键点：**
- `TypedDict` 提供类型安全
- 所有节点通过修改 state 来传递数据
- 节点函数接收 state 并返回部分更新

### 节点定义

```python
def retrieve(state: RAGState):
    docs = retriever.invoke(state["question"])
    context = "\n\n".join(d.page_content for d in docs)
    return {"context": context}  # 只返回需要更新的字段
```

### 条件路由

```python
def route(state: RAGState):
    if state.get("grade") == "sufficient":
        return END
    if state.get("loop_count", 0) >= 2:
        return END
    return "rewrite"

builder.add_conditional_edges(
    "grade",           # 起点节点
    route,             # 路由函数
    {"rewrite": "rewrite", END: END}  # 映射表
)
```

### 图的构建流程

```python
# 1. 创建 builder
builder = StateGraph(RAGState)

# 2. 添加节点
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
builder.add_node("grade", grade_answer)
builder.add_node("rewrite", rewrite_query)

# 3. 设置入口点
builder.set_entry_point("retrieve")

# 4. 添加无条件边
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", "grade")
builder.add_edge("rewrite", "retrieve")

# 5. 添加条件边
builder.add_conditional_edges("grade", route, {"rewrite": "rewrite", END: END})

# 6. 编译图
graph = builder.compile()
```

## 常见问题

### Q1: 为什么需要评估环节？

**A:** 评估环节可以检测回答质量，如果回答不充分（如"信息不足"），系统会自动重写查询并重新检索，直到得到满意的答案。

### Q2: 如何设置最大迭代次数？

**A:** 在条件路由函数中检查 `loop_count`：
```python
if state.get("loop_count", 0) >= 2:
    return END  # 最多重写 2 次
```

### Q3: 评估策略有哪些？

**A:** 
1. **硬编码规则**：快速判断（如检测"信息不足"关键词）
2. **LLM 精排**：使用 LLM 深度评估回答质量

两者结合可以在保证性能的同时提高准确性。

### Q4: 如何自定义评估逻辑？

**A:** 修改 `grade_answer` 函数：
```python
def grade_answer(state: RAGState):
    # 添加自定义评估逻辑
    # 返回 {"grade": "sufficient"} 或 {"grade": "insufficient"}
```

## 学习路径

1. **基础阶段**：运行 `simulate.py` 理解 LangGraph 状态机机制
2. **进阶阶段**：运行 `langgraph_rag.py` 体验完整 RAG 闭环
3. **扩展阶段**：修改节点逻辑，添加自定义评估策略
4. **实战阶段**：集成更多工具（如搜索、计算器）到图中

## 进阶主题

### 添加新节点

```python
def summarize(state: RAGState):
    """总结节点：对最终答案进行总结"""
    prompt = f"""对以下回答进行总结：{state['answer']}"""
    summary = llm.invoke(prompt).content
    return {"summary": summary}

builder.add_node("summarize", summarize)
builder.add_edge(END, "summarize")  # 在结束前添加总结节点
```

### 可视化图结构

```python
# 安装依赖
pip install pydot graphviz

# 生成可视化图
from langchain_core.runnables import RunnableSequence

graph.get_graph().draw_mermaid_png(output_file="rag_graph.png")
```

### 多分支路由

```python
def decide_next(state: RAGState):
    if state["grade"] == "sufficient":
        return "summarize"
    elif state["loop_count"] >= 2:
        return "fallback"  # 回退策略
    return "rewrite"

builder.add_conditional_edges(
    "grade",
    decide_next,
    {"summarize": "summarize", "fallback": "fallback", "rewrite": "rewrite"}
)
```

## 参考资料

- [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/)
- [LangGraph 核心概念](https://langchain-ai.github.io/langgraph/concepts/)
- [RAG 评估策略](https://python.langchain.com/docs/evaluation/)
- [条件路由示例](https://langchain-ai.github.io/langgraph/tutorials/rag/)
