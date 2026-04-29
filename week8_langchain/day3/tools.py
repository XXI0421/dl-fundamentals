from langchain_core.tools import tool
from datetime import datetime
from vectorstore import load_and_split, build_vectorstore, get_retriever

chunks = load_and_split(data_dir="data")
vectorstore = build_vectorstore(chunks)
retriever = get_retriever("base", vectorstore, chunks)

@tool
def search(query: str) -> str:
    """用于从知识库检索相关信息。输入应该是具体的关键词或问题。"""
    if retriever is None:
        return "错误：向量库未初始化"
    docs = retriever.invoke(query)
    return "\n".join(d.page_content for d in docs)

@tool
def calculate(expression: str) -> str:
    """用于执行数学计算。输入应该是合法的 Python 数学表达式，如 '2+2' 或 'math.sqrt(16)'。"""
    try:
        return str(eval(expression, {"__builtins__": {}}, {"math": __import__("math")}))
    except Exception as e:
        return f"计算错误: {e}"

@tool
def get_current_time() -> str:
    """返回当前时间。无需输入参数。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

tools = [search, calculate, get_current_time]

if __name__ == "__main__":
    print("=" * 60)
    print("工具定义信息")
    print("=" * 60)
    for t in tools:
        print(f"\n【工具名称】{t.name}")
        print(f"【工具描述】{t.description}")
        print(f"【参数 Schema】{t.args}")
    print("\n" + "=" * 60)

    # print("\n【测试检索】")
    # result = search.invoke("LCEL 是什么")
    # print(f"检索结果：{result[:100]}...")

    print("\n【测试计算】")
    result = calculate.invoke("2**10")
    print(f"计算结果：{result}") # 1024

    print("\n【测试时间时间】")
    result = get_current_time.invoke("")
    print(f"当前时间：{result}")
