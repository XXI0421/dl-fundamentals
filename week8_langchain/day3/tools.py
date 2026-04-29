from langchain_core.tools import tool
from datetime import datetime
from vectorstore import load_and_split, build_vectorstore, get_retriever

# 模块级变量声明，延迟初始化
_chunks = None
_vectorstore = None
_retriever = None

def _ensure_initialized():
    """按需初始化全局变量，确保只执行一次"""
    global _chunks, _vectorstore, _retriever
    if _retriever is None:
        _chunks = load_and_split(data_dir="data")
        _vectorstore = build_vectorstore(_chunks)
        _retriever = get_retriever("base", _vectorstore, _chunks)

@tool
def search(query: str) -> str:
    """用于从知识库检索相关信息。输入应该是具体的关键词或问题。"""
    _ensure_initialized()
    if _retriever is None:
        return "错误：向量库未初始化"
    docs = _retriever.invoke(query)
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

@tool
def python_executor(code: str) -> str:
    """执行 Python 代码。支持数据处理、网络请求和生成图表。
    预注入：plt、requests、json、math。禁止文件读写和系统命令。"""
    try:
        import io, contextlib, os, glob, json

        os.makedirs("./output", exist_ok=True)
        for f in glob.glob("./output/*.png"):
            os.remove(f)

        _allowed = {"matplotlib", "matplotlib.pyplot", "requests", "json", "math", "numpy", "pandas", "numpy.random"}

        def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name in _allowed or any(name.startswith(a + ".") for a in _allowed):
                return __import__(name, globals, locals, fromlist, level)
            raise ImportError(f"模块 {name} 不在白名单中")

        safe_globals = {
            "__builtins__": {
                "print": print, "range": range, "len": len, "sum": sum,
                "max": max, "min": min, "abs": abs, "int": int, "float": float,
                "str": str, "list": list, "dict": dict, "tuple": tuple,
                "enumerate": enumerate, "zip": zip, "sorted": sorted,
                "round": round, "type": type, "isinstance": isinstance,
                "__import__": _safe_import,
            },
            "json": json,
            "math": __import__("math"),
            "plt": __import__("matplotlib.pyplot"),
            "requests": __import__("requests"),
        }

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exec(code, safe_globals, {})

        output = stdout.getvalue()

        images = glob.glob("./output/*.png")
        if images:
            return f"执行成功。生成图表: {images[0]}\n\n控制台输出:\n{output}"

        return output or "执行成功，无输出"
    except Exception as e:
        return f"执行错误: {e}"

tools = [search, calculate, get_current_time, python_executor]

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

    # print("\n【测试计算】")
    # result = calculate.invoke("2**10")
    # print(f"计算结果：{result}") # 1024

    # print("\n【测试时间时间】")
    # result = get_current_time.invoke({})
    # print(f"当前时间：{result}")

    print("\n【测试Python执行器】")
    result = python_executor.invoke("print('Hello, World!')")
    print(f"执行结果：{result}")
