import os
from langchain_openai import ChatOpenAI
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor  
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tools import search, calculate, get_current_time
tools = [search, calculate, get_current_time]

llm = ChatOpenAI(
    model="moonshot-v1-128k",
    api_key=os.getenv("KIMI_API_KEY") or "your-key",
    base_url="https://api.moonshot.cn/v1",
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个智能助手，可以使用工具来回答问题。请尽量准确。"),
    MessagesPlaceholder("chat_history", optional=True),  # 对话记忆占位符
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),  
])

agent = create_tool_calling_agent(llm, tools, prompt=prompt)  

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,          # 打印每一步 Thought/Action/Observation
    max_iterations=5,      # 防止死循环
    handle_parsing_errors=True,  # 自动处理 JSON 解析错误：把错误信息喂回 LLM，让它重试
)

if __name__ == "__main__":
    query = "LCEL 是什么，以及 2 的 10 次方是多少"
    result = agent_executor.invoke({"input": query})
    print("\n【最终答案】", result["output"])