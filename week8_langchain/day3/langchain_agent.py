# langchain_agent.py
import os
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.memory import ConversationBufferWindowMemory
from tools import search, python_executor


def create_agent(memory=None, verbose=True, max_iterations=5):
    """创建 Tool Calling Agent，支持记忆注入和策略切换"""
    llm = ChatOpenAI(
        model="moonshot-v1-128k",
        api_key=os.getenv("KIMI_API_KEY") or "your-key",
        base_url="https://api.moonshot.cn/v1",
        temperature=0
    )
    
    tools = [search, python_executor]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个智能助手，可以使用工具来回答问题。请尽量准确。"),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt=prompt)
    
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        max_iterations=max_iterations,
        memory=memory,
    )

if __name__ == "__main__":
    # 使用 WindowMemory，k=3 适合中等长度对话
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=3
    )
    
    agent = create_agent(memory=memory, verbose=True)
    
    # 测试
    query = """
请执行 Python 代码完成以下任务：
1. 使用 requests 访问 GitHub API，获取 Python 仓库中收藏前 10 个仓库
2. 从返回的 JSON 中提取仓库名（full_name）和 stars 数量（stargazers_count）
3. 使用 matplotlib 绘制水平条形图（barh），标题"Top 10 Python Repositories on GitHub"
4. 保存到 ./output/github_repos.png
5. 打印每个仓库的 stars 数量
"""

    print(f"\n{'='*40}")
    print(f"【问题】{query}")
    result = agent.invoke({"input": query})
    print(f"【回答】{result['output']}")
