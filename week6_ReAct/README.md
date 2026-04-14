# Week 6: ReAct 多工具协作 Agent

## 系统介绍

Week 6 ReAct 多工具协作 Agent 是一个基于 **ReAct 范式** 的高级智能代理系统，支持多工具调用、错误恢复和会话记忆管理。该系统能够自动分析任务需求，选择合适的工具，执行代码，并处理执行结果。

## 功能特性

### 🚀 核心功能

- **多工具协作**：支持搜索、代码执行、文件读写等多种工具
- **安全代码沙箱**：进程级隔离执行 Python 代码，支持网络请求控制
- **错误恢复机制**：工具调用失败时自动重试（指数退避）
- **无效JSON修正**：自动检测并修正 LLM 生成的无效工具调用
- **会话记忆系统**：短期记忆 + 长期记忆 + 反思引擎
- **网络请求支持**：沙箱支持网络模式，可访问外部 API

### 🔧 技术特性

- **多模式支持**：default、data\_analysis、chart、full 四种执行模式
- **进程级隔离**：安全执行用户代码，防止恶意操作
- **资源限制**：超时时间、内存限制、递归深度限制
- **模块化设计**：工具注册表、Agent 核心、沙箱分离
- **Kimi API 集成**：支持使用 Kimi API 进行推理

## 学习路径

### 阶段 1：理解 ReAct 范式

1. **学习 ReAct 循环原理**
   - 理解 Thought-Action-Observation 循环
   - 学习工具调用格式和解析逻辑
   - 参考 `understand/`
2. **基础工具调用示例**
   ```bash
   python test_day5_demo.py
   ```
   - 体验基础的 ReAct Agent
   - 学习会话管理和工具调用
   - 理解短期记忆和长期记忆

### 阶段 2：理解多工具协作

1. **高级 Agent 架构**
   - 阅读 `react_agent_v4.py` 源码
   - 理解 MultiToolAgent 类的设计
   - 学习重试机制和错误修正逻辑
2. **工具开发**
   - 查看 `tools/` 目录下的工具实现
   - 学习工具注册和调用流程
   - 理解工具参数和返回值格式

### 阶段 3：直接运用实战

1. **安装依赖**
   ```bash
   # 安装基本依赖
   pip install requests matplotlib numpy pandas
   ```
2. **运行演示程序**
   ```bash
   python final_demo.py
   ```
3. **实战应用步骤**
   1. **输入问题**：如 "帮我分析 2025 年 Python 在 GitHub 上的趋势"
   2. **观察执行**：Agent 会自动选择工具、执行代码、保存结果
   3. **查看结果**：获取生成的图表和分析报告
4. **实战项目**
   - **数据分析**：使用沙箱执行 Python 代码进行数据分析
   - **图表生成**：生成可视化图表并保存
   - **文件操作**：读写文件，保存分析报告
   - **网络请求**：调用外部 API 获取实时数据

## 技术架构

### 系统组件

| 组件       | 功能             | 实现文件                      |
| -------- | -------------- | ------------------------- |
| Agent 核心 | 任务分析、工具选择、结果总结 | `react_agent_v4.py`       |
| 工具注册表    | 工具注册、查找、调用     | `tools/base.py`           |
| 代码沙箱     | 安全执行 Python 代码 | `tools/python_sandbox.py` |
| 实用工具     | 文件操作、搜索、计算     | `tools/real_tools.py`     |
| LLM 客户端  | Kimi API 调用    | `llm_client.py`           |

### 核心流程

```
用户问题 → 分析需求 → 选择工具 → 调用工具 → 获取结果 → 总结回答
              ↓              ↓              ↓
           短期记忆       工具执行       反思学习
              ↓              ↓              ↓
           长期记忆       错误恢复       知识存储
```

### ReAct 循环

```
┌─────────────────────────────────────────────┐
│  ReAct Loop (per iteration)                 │
│                                             │
│  Input: Query + Trajectory History          │
│      ↓                                      │
│  Prompt Construction                        │
│      ↓                                      │
│  LLM.generate()                             │
│      ↓                                      │
│  Parse → Thought_i + Action_i               │
│      ↓                                      │
│  if Action == Finish[answer]: STOP          │
│      ↓                                      │
│  Execute Tool → Observation_i               │
│      ↓                                      │
│  Append to Trajectory                        │
│      ↓                                      │
│  Loop (max_iterations=8)                    │
└─────────────────────────────────────────────┘
```

## 配置选项

### Agent 配置

```python
from react_agent_v4 import MultiToolAgent

agent = MultiToolAgent(
    llm_client,
    tool_registry,
    max_iterations=8,        # 最大迭代次数
    memory_k=5,              # 短期记忆保留步数
    ltm_threshold=0.3,       # 长期记忆阈值
    ltm_max_facts=3,         # 最大长期记忆数量
    max_retries=3,           # 最大重试次数
    retry_base_delay=1.0,    # 重试基础延迟
    retry_multiplier=2.0     # 重试延迟倍数（指数退避）
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

# 完整模式（允许所有操作，包括网络）
result = python_sandbox("import requests; r = requests.get('https://api.github.com')", mode='full')

# 自定义网络权限
result = python_sandbox(code, mode='data_analysis', allow_network=True)
```

## 核心功能详解

### 1. 多模式沙箱

沙箱支持四种执行模式：

| 模式                 | 功能   | 允许的模块                                  |
| ------------------ | ---- | -------------------------------------- |
| **default**        | 基础模式 | 标准库                                    |
| **data\_analysis** | 数据分析 | 标准库 + numpy + pandas                   |
| **chart**          | 图表生成 | 数据分析模块 + matplotlib + plotly + seaborn |
| **full**           | 完整模式 | 所有模块 + 网络请求                            |

### 2. 安全机制

- **进程级隔离**：代码在独立进程中执行
- **资源限制**：超时时间、内存限制、递归深度限制
- **危险代码拦截**：禁止 exec、eval、系统命令执行
- **文件写入限制**：仅允许写入临时目录
- **网络控制**：可配置是否允许网络请求

### 3. 错误恢复机制

**重试机制（指数退避）**：

```python
# 重试策略
# 第1次重试：1秒后
# 第2次重试：2秒后（1 * 2）
# 第3次重试：4秒后（2 * 2）
```

**无效 JSON 修正**：

- 检测 LLM 生成的无效 JSON 格式
- 提示 LLM 重新生成正确的工具调用
- 支持多次修正尝试

### 4. 会话记忆系统

**短期记忆**：保留最近 K 步对话历史
**长期记忆**：使用向量检索存储关键事实
**反思引擎**：任务完成后自动总结并存储新知识

## 扩展建议

1. **添加更多工具**：
   - Web 搜索工具（已实现）
   - 数据库查询工具
   - 邮件发送工具
   - API 调用工具
2. **增强沙箱能力**：
   - 添加更多数据处理库
   - 支持更多可视化库
   - 添加机器学习模型训练支持
3. **优化 Agent 决策**：
   - 添加工具选择策略
   - 实现工具优先级排序
   - 添加成本估算（API 调用次数、执行时间）
4. **增强记忆系统**：
   - 添加记忆编辑功能
   - 实现记忆过期机制
   - 添加记忆检索优化

## 示例使用

### 示例 1：数据分析

```bash
python final_demo.py

你: 帮我生成一个包含100个随机数的折线图并保存

🤔 正在处理...
🤖 Agent: 已生成图表并保存到 chart.png
```

### 示例 2：网络请求

```bash
python final_demo.py

你: 获取 GitHub 上 Python 热门仓库数据

🤔 正在处理...
🤖 Agent: 已获取 30 个热门仓库数据，Star 数量最高的是 xxx
```

### 示例 3：文件读写

```bash
python final_demo.py

你: 创建一个包含 Python 学习笔记的文件

🤔 正在处理...
🤖 Agent: 已将笔记保存到 python_notes.txt

你: 读取刚才创建的文件

🤔 正在处理...
🤖 Agent: 文件内容如下：...
```

### 示例 4：复杂任务

```bash
python final_demo.py

你: 分析 2025 年 Python 在 GitHub 上的趋势，生成可视化报告

🤔 正在处理...
# Agent 会：
# 1. 使用沙箱获取 GitHub 数据
# 2. 分析数据
# 3. 生成图表
# 4. 保存报告

🤖 Agent: 已完成分析，报告已保存到 python_github_trends.png
```

## 故障排除

### 常见问题

1. **Kimi API 调用失败**：
   - 检查 API Key 是否正确设置
   - 确保网络连接正常
   - 查看控制台错误信息
2. **沙箱执行失败**：
   - 检查代码是否包含危险操作
   - 确认使用了正确的执行模式
   - 查看 stderr 中的错误信息
3. **工具调用重试多次失败**：
   - 检查工具参数是否正确
   - 确认工具是否正确注册
   - 查看工具实现是否有问题
4. **图表生成失败**：
   - 确保安装了 matplotlib
   - 确认使用了 chart 或 full 模式
   - 检查代码是否正确

## 技术栈

- **后端**：Python 3.8+
- **LLM**：Kimi API (moonshot-v1-128k)
- **向量检索**：FAISS (用于长期记忆)
- **可视化**：matplotlib, plotly
- **数据分析**：numpy, pandas
- **网络请求**：requests

## 项目结构

```
week6_ReAct/
├── action/                  # 主代码目录
│   ├── react_agent_v3.py    # ReAct Agent V3（基础版）
│   ├── react_agent_v4.py    # MultiToolAgent（完整版）
│   ├── llm_client.py        # Kimi API 客户端
│   ├── tools/               # 工具目录
│   │   ├── base.py          # 工具基类和注册表
│   │   ├── real_tools.py    # 实用工具（文件操作、搜索等）
│   │   └── python_sandbox.py # 安全代码沙箱
│   ├── test_day5_demo.py    # 基础演示程序
│   ├── test_multi_tool_demo.py # 多工具演示
│   └── final_demo.py        # 最终演示程序
├── understand/              # 理解部分
│   └── react_explained.ipynb # ReAct 原理讲解
└── README.md                # 本文件
```

## 脚本说明

### 1. 基础演示 (`test_day5_demo.py`)

- **功能**：基础的 ReAct Agent 交互演示
- **用途**：学习会话管理和基础工具调用

### 2. 多工具演示 (`test_multi_tool_demo.py`)

- **功能**：演示复杂任务的多工具协作
- **用途**：体验 GitHub 趋势分析等复杂任务

### 3. 最终演示 (`final_demo.py`)

- **功能**：完整的交互式演示程序
- **用途**：日常使用和测试

### 4. ReAct 原理讲解 (`understand/react_explained.ipynb`)

- **功能**：Jupyter Notebook 讲解 ReAct 范式
- **用途**：学习 ReAct 原理

## 许可证

MIT License

## 致谢

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- [Moonshot AI](https://platform.moonshot.cn/) - Kimi API
- [FAISS](https://github.com/facebookresearch/faiss) - 向量检索
- [matplotlib](https://matplotlib.org/) - 可视化库

***

**Week 6 ReAct 多工具协作 Agent** - 一个功能强大、安全可靠的智能代理系统，为您的复杂任务提供自动化解决方案。
