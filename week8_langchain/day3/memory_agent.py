import os
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.memory import ConversationBufferMemory, ConversationBufferWindowMemory

# ========== 1. 工具 ==========
from tools import search, calculate, get_current_time
tools = [search, calculate, get_current_time]

# ========== 2. LLM ==========
llm = ChatOpenAI(
    model="moonshot-v1-128k",
    api_key=os.getenv("KIMI_API_KEY") or "your-key",
    base_url="https://api.moonshot.cn/v1",
    temperature=0
)

# ========== 3. Prompt（必须加 chat_history 占位符）==========
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个智能助手，可以使用工具来回答问题。请尽量准确。"),
    MessagesPlaceholder("chat_history"),  # ← 记忆注入点
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

# ========== 4. 记忆初始化 ==========
# 方案 A：Buffer（存全部）
# memory = ConversationBufferMemory(
#     memory_key="chat_history",
#     return_messages=True  # 返回消息列表，不是字符串
# )

# 方案 B：Window（只存最近 n 轮，适合长对话）
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    return_messages=True,
    k=1  # 只保留最近 1 轮（用户+AI = 1轮）
)

# ========== 5. Agent + Executor ==========
agent = create_tool_calling_agent(llm, tools, prompt=prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5,
    memory=memory,  # ← 注入记忆
)

# ========== 6. 多轮对话测试 ==========
if __name__ == "__main__":
    # 第一轮
    r1 = agent_executor.invoke({"input": "LCEL 是什么"})
    print(f"\n【Round 1】{r1['output']}")
    
    # 第二轮（依赖第一轮上下文）
    r2 = agent_executor.invoke({"input": "那它支持哪些调用模式"})
    print(f"\n【Round 2】{r2['output']}")
    
    # 第三轮（测试记忆是否丢失早期信息）
    r3 = agent_executor.invoke({"input": "刚才我们聊的第一个话题是什么"})
    # 注：LLM 可能会从 Round 2 的"调用模式"反推出 Round 1 是 LCEL
    print(f"\n【Round 3】{r3['output']}")


