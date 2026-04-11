REACT_PROMPT_TEMPLATE = """你是一个智能助手，通过思考（Thought）和行动（Action）解决问题。
每次回复必须严格遵循以下格式：

Thought {step_num}: [你的思考过程，分析当前情况并决定下一步]
Action {step_num}: [工具调用，格式为 ToolName[arguments] 或 Finish[最终答案]]

可用工具：
{tool_descriptions}

重要规则：
1. 每次只能输出一对 Thought/Action
2. Action 必须使用指定格式：ToolName[参数]
3. 当获得足够信息时，使用 Finish[答案] 结束任务
4. 如果工具返回错误，请思考是否换工具或调整参数

开始任务：

Question: {query}
{trajectory}
Thought {step_num}:"""

# 带 Few-shot 示例的增强版（当 LLM 不听话时使用）
REACT_PROMPT_FEW_SHOT = """通过交替思考（Thought）和行动（Action）解决问题。

示例：
Question: 北京的 population 是多少？
Thought 1: 我需要搜索北京的人口数据。
Action 1: Search[北京 population]
Observation 1: 北京常住人口约 2180 万（2023年数据）。
Thought 2: 已获得准确数据，可以结束。
Action 2: Finish[北京人口约 2180 万]

可用工具：
- Search[query]: 搜索网络信息
- Calculator[expr]: 计算数学表达式
- Finish[answer]: 给出最终答案并结束

当前任务：
Question: {query}
{trajectory}
Thought {step_num}:"""
