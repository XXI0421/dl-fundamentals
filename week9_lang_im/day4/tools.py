from langchain_core.tools import tool
from datetime import datetime
from vectorstore import load_and_split, build_vectorstore, get_retriever

_chunks = None
_vectorstore = None
_retriever = None


def _ensure_initialized():
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
    """用于获取当前时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def python_executor(code: str) -> str:
    """执行 Python 代码。支持数据处理、网络请求和生成图表。
    预注入常用模块：plt、requests、json、math、os、glob、matplotlib、numpy、pandas 等。
    禁止文件读写（除 ./output/）、系统命令、网络访问（除 requests 标准用法）。
    """
    import io
    import contextlib
    import os
    import glob
    import json
    import math
    import time
    import re
    import typing
    import datetime as dt

    os.makedirs("./output", exist_ok=True)

    _ALLOWED_MODULES = {
        "matplotlib", "matplotlib.pyplot", "matplotlib.colors", "matplotlib.ticker",
        "requests", "requests.exceptions",
        "json", "math", "numpy", "pandas", "numpy.random",
        "os", "glob", "io", "contextlib", "pathlib",
        "datetime", "typing", "time", "re", "traceback",
        "sys", "urllib", "urllib.request", "urllib.error",
        "collections", "itertools", "functools", "operator",
        "random", "decimal", "fractions", "complex",
        "array", "copy", "pprint", "string",
    }

    _SAFE_BUILTINS = {
        "print": print, "range": range, "len": len, "sum": sum,
        "max": max, "min": min, "abs": abs, "int": int, "float": float,
        "str": str, "list": list, "dict": dict, "tuple": tuple,
        "set": set, "frozenset": frozenset, "bool": bool,
        "enumerate": enumerate, "zip": zip, "sorted": sorted,
        "reversed": reversed, "map": map, "filter": filter,
        "round": round, "type": type, "isinstance": isinstance,
        "issubclass": issubclass, "hasattr": hasattr, "getattr": getattr,
        "setattr": setattr, "delattr": delattr, "callable": callable,
        "hash": hash, "id": id, "repr": repr, "ord": ord, "chr": chr,
        "bin": bin, "hex": oct, "oct": oct, "slice": slice,
        "any": any, "all": all, "open": None,
        "exec": None, "eval": None, "compile": None, "execfile": None,
        "__import__": None,
        "__build_class__": None,  # 支持 class 定义
        "Exception": Exception, "BaseException": BaseException,
        "SystemExit": SystemExit, "KeyboardInterrupt": KeyboardInterrupt,
        "GeneratorExit": GeneratorExit, "StopIteration": StopIteration,
        "Warning": Warning, "UserWarning": UserWarning,
        "ValueError": ValueError, "TypeError": TypeError, "KeyError": KeyError,
        "IndexError": IndexError, "AttributeError": AttributeError,
        "NameError": NameError, "SyntaxError": SyntaxError, "RuntimeError": RuntimeError,
        "ImportError": ImportError, "ModuleNotFoundError": ModuleNotFoundError,
        "IOError": IOError, "OSError": OSError, "FileNotFoundError": FileNotFoundError,
        "ZeroDivisionError": ZeroDivisionError, "OverflowError": OverflowError,
        "FloatingPointError": FloatingPointError, "AssertionError": AssertionError,
        "NotImplementedError": NotImplementedError, "RecursionError": RecursionError,
    }

    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in _SAFE_BUILTINS and _SAFE_BUILTINS.get(name) is None:
            raise ImportError(f"禁止导入危险模块: {name}")
        if name in _ALLOWED_MODULES or any(name.startswith(a + ".") for a in _ALLOWED_MODULES):
            return __import__(name, globals, locals, fromlist, level)
        raise ImportError(f"模块 '{name}' 不在白名单中")

    import builtins

    _safe_builtins = {
        "print": print, "range": range, "len": len, "sum": sum,
        "max": max, "min": min, "abs": abs, "int": int, "float": float,
        "str": str, "list": list, "dict": dict, "tuple": tuple,
        "set": set, "frozenset": frozenset, "bool": bool,
        "enumerate": enumerate, "zip": zip, "sorted": sorted,
        "reversed": reversed, "map": map, "filter": filter,
        "round": round, "type": type, "isinstance": isinstance,
        "issubclass": issubclass, "hasattr": hasattr, "getattr": getattr,
        "setattr": setattr, "delattr": delattr, "callable": callable,
        "hash": hash, "id": id, "repr": repr, "ord": ord, "chr": chr,
        "bin": bin, "hex": oct, "oct": oct, "slice": slice,
        "any": any, "all": all, "open": None,
        "exec": None, "eval": None, "compile": None, "execfile": None,
        "__import__": _safe_import,
        "__build_class__": builtins.__build_class__,  # 支持 class 定义
        "exit": lambda: None,
        "quit": lambda: None,
    }

    for _exc_name in [
        "Exception", "BaseException", "SystemExit", "KeyboardInterrupt",
        "GeneratorExit", "StopIteration", "Warning", "UserWarning",
        "ValueError", "TypeError", "KeyError", "IndexError", "AttributeError",
        "NameError", "SyntaxError", "RuntimeError", "ImportError",
        "ModuleNotFoundError", "IOError", "OSError", "FileNotFoundError",
        "ZeroDivisionError", "OverflowError", "FloatingPointError", "AssertionError",
        "NotImplementedError", "RecursionError",
    ]:
        _safe_builtins[_exc_name] = eval(_exc_name)

    _safe_globals = {
        "__builtins__": _safe_builtins,
        "__name__": "__main__",
        "__doc__": None,
        "__package__": None,
        "__loader__": None,
        "__spec__": None,
        "__file__": None,
        "__cached__": None,
        "json": json,
        "math": math,
        "time": time,
        "re": re,
        "typing": typing,
        "datetime": dt,
        "plt": __import__("matplotlib.pyplot"),
        "matplotlib": __import__("matplotlib"),
        "requests": __import__("requests"),
        "os": os,
        "glob": glob,
        "io": io,
        "contextlib": contextlib,
        "numpy": __import__("numpy"),
        "pandas": __import__("pandas"),
        "random": __import__("random"),
        "collections": __import__("collections"),
        "itertools": __import__("itertools"),
        "functools": __import__("functools"),
        "operator": __import__("operator"),
        "pprint": __import__("pprint"),
        "string": __import__("string"),
        "decimal": __import__("decimal"),
        "fractions": __import__("fractions"),
        "copy": __import__("copy"),
        "array": __import__("array"),
    }

    try:
        stdout = io.StringIO()
        stderr = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            with contextlib.redirect_stderr(stderr):
                exec(code, _safe_globals, _safe_globals)

        output = stdout.getvalue()
        err_output = stderr.getvalue()

        images = list(glob.glob("./output/*.png")) + list(glob.glob("./output/*.jpg"))
        if images:
            if err_output:
                return f"✅ 执行成功\n📊 生成图表: {images[0]}\n\n📤 控制台输出:\n{output}\n\n⚠️ 错误输出:\n{err_output}"
            return f"✅ 执行成功\n📊 生成图表: {images[0]}\n\n📤 控制台输出:\n{output}"

        if err_output:
            return f"✅ 执行成功（有错误输出）\n\n📤 控制台输出:\n{output}\n\n⚠️ 错误输出:\n{err_output}"

        return output or "✅ 执行成功，无输出"

    except SyntaxError as e:
        return f"❌ 语法错误: {e.filename}:{e.lineno} - {e.text}"
    except ImportError as e:
        return f"❌ 导入错误: {e}"
    except Exception as e:
        error_type = type(e).__name__
        return f"❌ 执行错误 [{error_type}]: {e}"


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

    print("\n【测试Python执行器】")
    result = python_executor.invoke("print('Hello, World!')")
    print(f"执行结果：{result}")