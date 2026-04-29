# LangChain Agent 进阶教程（Day 3）

## 概述

本目录包含 LangChain Agent（智能体）的基础教学材料，重点介绍 Tool Calling Agent、记忆机制和多轮对话交互。

## 核心概念

### Tool Calling Agent

| 组件 | 说明 |
|------|------|
| **Agent** | 负责思考和决策的 LLM |
| **Tools** | Agent 可以调用的外部函数 |
| **AgentExecutor** | 执行 Agent 决策的运行时 |
| **Prompt** | 定义 Agent 行为和工具使用规则的模板 |

### 记忆机制

| 类型 | 说明 |
|------|------|
| **ConversationBufferMemory** | 存储全部对话历史 |
| **ConversationBufferWindowMemory** | 只存储最近 k 轮对话 |

### 工具类型

| 工具 | 功能 | 参数 |
|------|------|------|
| `search` | 从向量库检索相关文档 | `query: str` |
| `calculate` | 安全执行数学计算 | `expression: str` |
| `get_current_time` | 获取当前时间 | 无参数 |
| `python_executor` | 安全执行 Python 代码 | `code: str` |

## 文件说明

### 1. vectorstore.py - 向量库工具模块

**功能说明：**
- 从 `data/` 目录加载文档（.txt, .md, .pdf）
- 构建和持久化 Chroma 向量库
- 提供多种检索策略获取接口

**核心函数：**

```python
from vectorstore import load_and_split, build_vectorstore, get_retriever

chunks = load_and_split(data_dir="data")
vectorstore = build_vectorstore(chunks)
retriever = get_retriever("base", vectorstore, chunks)
```

**检索策略：**

| 策略 | 实现 |
|------|------|
| `base` | 基础向量检索 |
| `multi` | MultiQuery 检索 |
| `ensemble` | BM25 + 向量混合 |
| `compress` | LLMChainExtractor 压缩 |
| `rerank` | FlashrankRerank 精排 |

### 2. tools.py - 工具定义

**功能说明：**
- 定义 Agent 可使用的工具
- 包含 4 个预定义工具
- 支持安全执行 Python 代码

**工具列表：**

```python
@tool
def search(query: str) -> str:
    """用于从知识库检索相关信息。输入应该是具体的关键词或问题。"""

@tool
def calculate(expression: str) -> str:
    """用于执行数学计算。输入应该是合法的 Python 数学表达式。"""

@tool
def get_current_time() -> str:
    """返回当前时间。无需输入参数。"""

@tool
def python_executor(code: str) -> str:
    """执行 Python 代码。支持数据处理、网络请求和生成图表。
    预注入：plt、requests、json、math。禁止文件读写和系统命令。"""
```

**python_executor 安全机制：**
- 白名单模块：`matplotlib`, `requests`, `json`, `math`, `numpy`, `pandas`
- 限制文件操作：只允许 `./output/` 目录
- 安全的 `__import__`：防止导入危险模块

### 3. tool_agent.py - 基础 Tool Calling Agent

**功能说明：**
- 创建支持多工具调用的 Agent
- 展示 Agent 执行过程（Thought → Action → Observation）
- 处理工具调用解析错误

**核心代码：**

```python
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor

agent = create_tool_calling_agent(llm, tools, prompt=prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,          # 打印每一步 Thought/Action/Observation
    max_iterations=5,     # 防止死循环
    handle_parsing_errors=True,  # 自动处理 JSON 解析错误
)
```

**运行方式：**
```bash
python tool_agent.py
```

**输出示例：**
```
> Entering new AgentExecutor chain...

Invoking: `search` with `{'query': 'LCEL'}`
...
Invoking: `calculate` with `{'expression': '2**10'}`
...
【最终答案】 LCEL 是 LangChain 0.1 版本引入的声明式管道语法，它允许开发者使用管道符号 | 来组合各种组件，构建复杂的 AI 工作流。其核心特性包括声明式组合、类型安全、支持多种调用模式以及内置重试和错误处理。基本模式基于 Runnable 接口构建，包括 RunnableLambda、RunnablePassthrough 和 RunnableParallel 等组件。

2 的 10 次方等于 1024。
```

### 4. memory_agent.py - 带记忆的 Agent

**功能说明：**
- 支持多轮对话记忆
- 展示 `ConversationBufferMemory` 和 `ConversationBufferWindowMemory` 区别
- 测试记忆是否正确保存和检索

**记忆类型对比：**

```python
# 方案 A：Buffer（存全部对话历史）
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 方案 B：Window（只存最近 k 轮）
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    return_messages=True,
    k=1  # 只保留最近 1 轮
)
```

**Prompt 中的记忆注入点：**

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个智能助手..."),
    MessagesPlaceholder("chat_history"),  # ← 记忆注入点
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])
```

**运行方式：**
```bash
python memory_agent.py
```

**输出示例：**
```
========================================
【Round 1】问：LCEL 是什么
答：LCEL 是 LangChain Expression Language 的缩写...

【Round 2】问：那它支持哪些调用模式
答：LCEL 支持 invoke、stream、batch 三种调用模式...

【Round 3】问：刚才我们聊的第一个话题是什么
答：我们聊的第一个话题是 LCEL...
```

### 5. langchain_agent.py - 完整 Agent 系统

**功能说明：**
- 整合记忆机制和 Python 执行器
- 支持复杂任务分解和执行
- 展示 Agent 的 Tool Calling 能力

**核心特性：**
- `search` 工具：从知识库检索
- `python_executor` 工具：执行 Python 代码生成图表
- `ConversationBufferWindowMemory`：保存最近 3 轮对话

**运行方式：**
```bash
python langchain_agent.py
```

**测试任务示例：**
```
请执行 Python 代码完成以下任务：
1. 使用 requests 访问 GitHub API，获取 Python 仓库中收藏前 10 个仓库
2. 从返回的 JSON 中提取仓库名（full_name）和 stars 数量（stargazers_count）
3. 使用 matplotlib 绘制水平条形图（barh），标题"Top 10 Python Repositories on GitHub"
4. 保存到 ./output/github_repos.png
5. 打印每个仓库的 stars 数量
```

**Agent 执行流程：**
```
> Entering new AgentExecutor chain...

Invoking: `python_executor` with `{'code': "import requests\nimport matplotlib.pyplot as plt\n\n# GitHub API URL for Python repositories sorted by stars\nurl = 'https://api.github.com/search/repositories?q=language:python&sort=stars&order=desc&per_page=10'\n\n# Send a GET request to the GitHub API\nresponse = requests.get(url)\n\n# Check if the request was successful\nif response.status_code == 200:\n    repos = response.json()\n    repo_names = [repo['full_name'] for repo in repos['items']]\n    stars_counts = [repo['stargazers_count'] for repo in repos['items']]\n    \n    # Plotting the horizontal bar chart\n    plt.figure(figsize=(10, 8))\n    plt.barh(repo_names, stars_counts, color='skyblue')\n    plt.xlabel('Stars')\n    plt.title('Top 10 Python Repositories on GitHub')\n    plt.tight_layout()\n    \n    # Save the plot to a file\n    plt.savefig('./output/github_repos.png')\n    \n    # Print the stars count for each repository\n    for name, stars in zip(repo_names, stars_counts):\n        print(f'{name}: {stars}')\nelse:\n    print('Failed to retrieve data from GitHub API')"}`


执行成功。生成图表: ./output\github_repos.png

控制台输出:
public-apis/public-apis: 428842
EbookFoundation/free-programming-books: 387053
donnemartin/system-design-primer: 346195
vinta/awesome-python: 295029
TheAlgorithms/Python: 220398
Significant-Gravitas/AutoGPT: 183881
AUTOMATIC1111/stable-diffusion-webui: 162662
huggingface/transformers: 160082
yt-dlp/yt-dlp: 159561
521xueweihan/HelloGitHub: 153904
我已经完成了您的请求。以下是每个 Python 仓库的 stars 数量：

- public-apis/public-apis: 428842
- EbookFoundation/free-programming-books: 387053
- donnemartin/system-design-primer: 346195
- vinta/awesome-python: 295029
- TheAlgorithms/Python: 220398
- Significant-Gravitas/AutoGPT: 183881
- AUTOMATIC1111/stable-diffusion-webui: 162662
- huggingface/transformers: 160082
- yt-dlp/yt-dlp: 159561
- 521xueweihan/HelloGitHub: 153904

此外，我还使用 matplotlib 绘制了一个水平条形图，标题为 "Top 10 Python Repositories on GitHub"，并 将其保存到了 `./output/github_repos.png` 文件中。您可以查看该文件以获取图表。
```

## 环境配置

### 安装依赖

```bash
pip install langchain langchain_openai langchain_community langchain_huggingface langchain_classic chromadb sentence-transformers
```

### 可选：安装 Flashrank（用于精排）

```bash
pip install flashrank
```

### 配置文件

```bash
# Windows PowerShell
$env:KIMI_API_KEY="your-api-key"

# Linux/macOS
export KIMI_API_KEY="your-api-key"
```

## 快速开始

### 1. 准备向量库（使用 day2 生成的数据）

```bash
# 确保 data/ 目录存在且有文档
ls data/

# 如需重新生成数据
cd ../day2
python generate_data.py
cd ../day3
```

### 2. 测试工具定义

```bash
python tools.py
```

**输出示例：**
```
============================================================
工具定义信息
============================================================

【工具名称】search
【工具描述】用于从知识库检索相关信息。输入应该是具体的关键词或问题。
【参数 Schema】{'properties': {'query': {...}}, 'required': ['query'], 'type': 'object'}

【工具名称】calculate
【工具描述】用于执行数学计算...
【参数 Schema】...

【工具名称】get_current_time
【工具描述】返回当前时间。无需输入参数。
【参数 Schema】{'properties': {}, 'type': 'object'}

【工具名称】python_executor
【工具描述】执行 Python 代码...
【参数 Schema】...
```

### 3. 运行基础 Agent

```bash
python tool_agent.py
```

### 4. 运行带记忆的 Agent

```bash
python memory_agent.py
```

### 5. 运行完整 Agent 系统

```bash
python langchain_agent.py
```

## Agent 执行机制详解

### Tool Calling 流程

```
用户输入 → LLM 思考 → 选择工具 → 执行工具 → 观察结果 → LLM 再次思考 → ...
```

### Prompt 模板结构

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个智能助手，可以使用工具来回答问题。"),
    MessagesPlaceholder("chat_history"),        # 对话历史
    ("human", "{input}"),                      # 用户输入
    MessagesPlaceholder("agent_scratchpad"),   # Agent 思考过程
])
```

### handle_parsing_errors 机制

```python
# 当工具调用解析失败时，框架自动把错误信息喂回 LLM
# LLM 会根据错误信息重试，直到成功或达到 max_iterations
handle_parsing_errors=True
```

## 常见问题

### Q1: Agent 报 "tool not found" 错误？

**A:** 检查 `create_tool_calling_agent` 的 `tools` 参数是否正确传递：
```python
tools = [search, calculate, get_current_time]
agent = create_tool_calling_agent(llm, tools, prompt=prompt)
```

### Q2: 记忆没有正确保存？

**A:** 检查两点：
1. `memory` 是否正确传入 `AgentExecutor`
2. Prompt 中是否有 `MessagesPlaceholder("chat_history")`

### Q3: python_executor 执行失败？

**A:** 常见原因：
1. 导入了不在白名单的模块
2. 代码中有文件读写操作（只能写 `./output/`）
3. `__import__` 参数传递错误

### Q4: max_iterations 太小导致中断？

**A:** 对于复杂任务，增加 `max_iterations`：
```python
AgentExecutor(agent=agent, tools=tools, max_iterations=10)
```

## 学习路径

1. **基础阶段**：运行 `tool_agent.py` 理解 Tool Calling 机制
2. **记忆阶段**：运行 `memory_agent.py` 掌握多轮对话
3. **进阶阶段**：运行 `langchain_agent.py` 体验完整 Agent
4. **扩展阶段**：修改 `tools.py` 添加自定义工具

## 进阶主题

### 添加自定义工具

```python
from langchain_core.tools import tool

@tool
def your_tool(param: str) -> str:
    """工具描述"""
    # 工具实现
    return result

# 添加到工具列表
tools.append(your_tool)
```

### 更换记忆类型

```python
# 使用向量数据库存储记忆
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# 将对话历史存入向量库
memory_vectorstore = Chroma.from_documents(
    documents=[Document(page_content=msg.content) for msg in messages],
    embedding=embedding_model
)
```

### 多 Agent 协作

```python
# 多个 Agent 可以分工协作
research_agent = create_agent(tools=[search])
coder_agent = create_agent(tools=[python_executor])

# research_agent 负责研究
# coder_agent 负责编码
```

## 参考资料

- [LangChain Agent 文档](https://python.langchain.com/docs/modules/agents/)
- [Tool Calling Agent](https://python.langchain.com/docs/modules/agents/agent_types/tool_calling_agent)
- [Conversation Memory](https://python.langchain.com/docs/modules/memory/)
- [Tool 调用错误处理](https://python.langchain.com/docs/modules/agents/agent_types/tool_calling_agent#handle_parsing_errors)