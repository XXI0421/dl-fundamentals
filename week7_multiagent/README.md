# Week 7: 多智能体协作系统 (Multi-Agent System)

## 系统介绍

Week 7 多智能体协作系统是一个基于 **ReAct 范式** 的高级智能代理协作框架，支持多个 Agent 之间的通信、任务分配和协作执行。该系统能够自动分析复杂任务，分配给合适的 Agent，协调多 Agent 协同工作，并汇总最终结果。

## 功能特性

### 🚀 核心功能

- **多智能体协作**：支持多个 Agent 并行执行和协作
- **消息总线系统**：Agent 之间通过消息总线进行通信
- **任务分配机制**：自动将复杂任务分解并分配给合适的 Agent
- **协调器架构**：MultiAgentCoordinator 负责任务调度和结果汇总
- **会话记忆管理**：支持短期记忆和长期记忆系统
- **工具调用扩展**：支持多种工具调用（代码沙箱、文件操作、网络搜索）

### 🔧 技术特性

- **模块化设计**：Agent、工具、记忆系统分离
- **异步通信**：基于消息总线的异步通信机制
- **可扩展性**：易于添加新的 Agent 类型和工具
- **安全性**：进程级隔离的代码沙箱
- **日志系统**：完善的日志记录和调试支持

## 学习路径

### 阶段 1：理解多智能体架构

1. **学习多智能体协作原理**
   - 理解 Agent 之间的通信机制
   - 学习消息总线系统设计
   - 参考 `message_bus.py`
2. **基础 Agent 示例**
   ```bash
   python test_react_agent.py
   ```
   - 体验基础的 ReAct Agent
   - 学习工具调用和记忆管理

### 阶段 2：理解协调器机制

1. **协调器架构**
   - 阅读 `multi_agent_coordinator.py` 源码
   - 理解任务分配和结果汇总逻辑
   - 学习 Agent 注册和管理
2. **消息总线**
   - 查看 `message_bus.py` 实现
   - 理解发布/订阅模式
   - 学习消息路由机制

### 阶段 3：直接运用实战

1. **安装依赖**
   ```bash
   # 安装基本依赖
   pip install requests matplotlib numpy pandas
   ```
2. **运行演示程序**
   ```bash
   streamlit run app.py
   ```
3. **实战应用步骤**
   1. **输入复杂任务**：如 "帮我分析 GitHub 上 Python 和 JavaScript 的趋势对比"
   2. **观察执行**：Coordinator 会分解任务并分配给多个 Agent
   3. **查看结果**：获取汇总的分析报告
4. **实战项目**
   - **多任务并行**：多个 Agent 同时执行不同任务
   - **数据分析协作**：数据收集 Agent + 分析 Agent + 可视化 Agent
   - **知识共享**：通过消息总线共享信息

## 技术架构

### 系统组件

| 组件          | 功能                 | 实现文件                         |
| ----------- | ------------------ | ---------------------------- |
| 协调器         | 任务分配、Agent 管理、结果汇总 | `multi_agent_coordinator.py` |
| 消息总线        | Agent 之间的异步通信      | `message_bus.py`             |
| ReAct Agent | 单个 Agent 的核心逻辑     | `react_agent.py`             |
| 工具注册表       | 工具注册、查找、调用         | `tools/base.py`              |
| 代码沙箱        | 安全执行 Python 代码     | `tools/python_sandbox.py`    |
| 实用工具        | 文件操作、搜索、计算         | `tools/real_tools.py`        |
| 记忆系统        | 短期记忆 + 长期记忆 + 反思   | `memory/`                    |
| LLM 客户端     | Kimi API 调用        | `llm_client.py`              |

### 核心流程

```
用户任务 → Coordinator → 任务分解 → Agent 分配 → 并行执行 → 结果汇总 → 最终回答
              ↓              ↓              ↓              ↓
           消息总线        Agent A        Agent B        Agent C
              ↓              ↓              ↓              ↓
           状态同步        工具调用        工具调用        工具调用
```

### 多智能体协作模式

```
┌─────────────────────────────────────────────────────────────┐
│              Multi-Agent Coordinator                        │
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │  Agent   │    │  Agent   │    │  Agent   │              │
│  │    A     │    │    B     │    │    C     │              │
│  └────┬─────┘    └────┬─────┘    └────┬─────┘              │
│       │               │               │                     │
│       └───────────────┼───────────────┘                     │
│                       ↓                                     │
│              ┌───────────────┐                              │
│              │  Message Bus  │                              │
│              │  (Publish/    │                              │
│              │   Subscribe)  │                              │
│              └───────────────┘                              │
└─────────────────────────────────────────────────────────────┘
```

## 配置选项

### 协调器配置

```python
from multi_agent_coordinator import MultiAgentCoordinator
from message_bus import MessageBus

# 创建消息总线
message_bus = MessageBus()

# 创建协调器
coordinator = MultiAgentCoordinator(
    message_bus=message_bus,
    max_agents=5,              # 最大 Agent 数量
    timeout=60,                # 任务超时时间（秒）
    retry_count=3              # 重试次数
)
```

### Agent 配置

```python
from react_agent import ReActAgent

agent = ReActAgent(
    llm_client=llm_client,
    tools=tool_registry,
    memory=memory_system,
    max_iterations=8,          # 最大迭代次数
    name="DataAnalyzer"        # Agent 名称
)
```

### 沙箱配置

```python
from tools.python_sandbox import python_sandbox

# 基础模式（仅标准库）
result = python_sandbox("print('Hello World')", mode='default')

# 数据分析模式（包含 numpy, pandas）
result = python_sandbox("import pandas; print(pandas.__version__)", mode='data_analysis')

# 图表模式（包含数据分析库 + 图表库）
result = python_sandbox("import matplotlib.pyplot as plt; plt.plot([1,2,3])", mode='chart')

# no_run 模式（仅生成脚本，不执行）
result = python_sandbox("print('Test')", mode='no_run')
```

## 核心功能详解

### 1. 多智能体协调

协调器负责：

- 任务分解：将复杂任务分解为子任务
- Agent 选择：根据任务类型选择合适的 Agent
- 并行执行：多个 Agent 同时执行不同子任务
- 结果汇总：汇总所有 Agent 的执行结果

### 2. 消息总线

基于发布/订阅模式：

- **发布**：Agent 发布消息到特定主题
- **订阅**：Agent 订阅感兴趣的主题
- **路由**：消息总线负责消息路由和传递
- **异步**：支持异步消息传递，不阻塞执行

### 3. 记忆系统

**短期记忆**：保留最近对话历史
**长期记忆**：使用向量检索存储关键事实
**反思引擎**：任务完成后自动总结并存储新知识

### 4. 安全代码沙箱

- **进程级隔离**：代码在独立进程中执行
- **多模式支持**：default、data\_analysis、chart、full、no\_run
- **资源限制**：超时时间、内存限制
- **脚本类型检测**：自动检测 Python/JavaScript/混合代码

## 扩展建议

1. **添加更多 Agent 类型**：
   - 数据分析 Agent
   - 可视化 Agent
   - 报告生成 Agent
   - 总结 Agent
2. **增强协调器能力**：
   - 添加任务优先级管理
   - 实现负载均衡
   - 添加故障转移机制
3. **扩展记忆系统**：
   - 添加记忆编辑功能
   - 实现记忆过期机制
   - 添加记忆检索优化
4. **优化消息总线**：
   - 添加消息持久化
   - 实现消息优先级
   - 添加消息确认机制

## 示例使用

### 示例 1：基础协作

```bash
streamlit run app.py

你: 帮我分析 GitHub 上 Python 和 JavaScript 的趋势对比

🤔 正在处理...
📋 Coordinator: 任务已分解为 2 个子任务
🚀 Agent[DataCollector-Python]: 开始执行
🚀 Agent[DataCollector-JavaScript]: 开始执行
✅ Agent[DataCollector-Python]: 完成
✅ Agent[DataCollector-JavaScript]: 完成
📊 Coordinator: 结果汇总中...

🤖 Agent: 分析完成！以下是 Python 和 JavaScript 在 GitHub 上的趋势对比：
...
```

### 示例 2：数据分析协作

```bash
streamlit run app.py

你: 分析 2025 年数据科学领域热门技术，并生成可视化报告

🤔 正在处理...
# Coordinator 会：
# 1. 分配数据收集任务
# 2. 分配数据分析任务
# 3. 分配可视化任务
# 4. 汇总结果并生成报告

🤖 Agent: 已完成分析，报告已保存到 data_science_report.png
```

### 示例 3：文件操作协作

```bash
streamlit run app.py

你: 创建一个项目文档，包含项目概述、技术栈和使用说明

🤔 正在处理...
# Agent 会：
# 1. 创建文档内容
# 2. 保存到文件

🤖 Agent: 文档已保存到 project_documentation.md
```

## 故障排除

### 常见问题

1. **Agent 注册失败**：
   - 检查 Agent 名称是否唯一
   - 确认 Agent 实现是否正确
   - 查看日志中的错误信息
2. **消息总线通信失败**：
   - 检查消息主题是否正确
   - 确认 Agent 是否已订阅主题
   - 查看消息总线日志
3. **沙箱执行失败**：
   - 检查代码是否包含危险操作
   - 确认使用了正确的执行模式
   - 查看 stderr 中的错误信息
4. **任务超时**：
   - 检查单个 Agent 的执行时间
   - 考虑增加超时时间
   - 优化任务分配策略

## 技术栈

- **后端**：Python 3.8+
- **LLM**：Kimi API (moonshot-v1-128k)
- **消息队列**：自定义消息总线（发布/订阅模式）
- **向量检索**：FAISS (用于长期记忆)
- **可视化**：matplotlib, plotly
- **数据分析**：numpy, pandas
- **网络请求**：requests

## 项目结构

```
week7_multiagent/
├── refacter/                     # 主代码目录
│   ├── __init__.py               # 模块初始化
│   ├── app.py                    # 应用入口
│   ├── config.py                 # 配置文件
│   ├── llm_client.py             # Kimi API 客户端
│   ├── message_bus.py            # 消息总线系统
│   ├── multi_agent_coordinator.py # 多智能体协调器
│   ├── react_agent.py            # ReAct Agent 核心
│   ├── memory/                   # 记忆系统
│   │   ├── __init__.py           # 记忆模块初始化
│   │   ├── short_term.py         # 短期记忆
│   │   ├── long_term.py          # 长期记忆
│   │   └── reflection.py         # 反思引擎
│   ├── tools/                    # 工具目录
│   │   ├── __init__.py           # 工具模块初始化
│   │   ├── base.py               # 工具基类和注册表
│   │   ├── logger.py             # 日志工具
│   │   ├── python_sandbox.py     # 安全代码沙箱
│   │   └── real_tools.py         # 实用工具（文件操作、搜索等）
│   ├── test_llm_client.py        # LLM 客户端测试
│   ├── test_memory.py            # 记忆系统测试
│   ├── test_multi_agent.py       # 多智能体测试
│   ├── test_react_agent.py       # ReAct Agent 测试
│   └── test_tools.py             # 工具测试
└── README.md                     # 本文件
```

## 脚本说明

### 1. 应用入口 (`app.py`)

- **功能**：完整的多智能体协作演示程序
- **用途**：日常使用和测试

### 2. 协调器 (`multi_agent_coordinator.py`)

- **功能**：多智能体协调器核心实现
- **用途**：任务分配和结果汇总

### 3. 消息总线 (`message_bus.py`)

- **功能**：Agent 之间的异步通信机制
- **用途**：支持 Agent 之间的消息传递

### 4. ReAct Agent (`react_agent.py`)

- **功能**：单个 Agent 的核心逻辑
- **用途**：执行具体任务和工具调用

### 5. 代码沙箱 (`tools/python_sandbox.py`)

- **功能**：安全执行 Python 代码
- **用途**：数据分析、图表生成等

### 6. 测试文件

- `test_llm_client.py`：LLM 客户端测试
- `test_memory.py`：记忆系统测试
- `test_multi_agent.py`：多智能体协作测试
- `test_react_agent.py`：ReAct Agent 测试
- `test_tools.py`：工具功能测试

## 许可证

MIT License

## 致谢

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- [Moonshot AI](https://platform.moonshot.cn/) - Kimi API
- [FAISS](https://github.com/facebookresearch/faiss) - 向量检索
- [matplotlib](https://matplotlib.org/) - 可视化库

***

**Week 7 多智能体协作系统** - 一个功能强大、可扩展的多智能体协作框架，为您的复杂任务提供高效的自动化解决方案。
