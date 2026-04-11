# tools/real_tools.py
import requests
import json
from typing import Optional
from .base import tool

@tool
def search_duckduckgo(query: str, topn: int = 3) -> str:
    """
    使用 DuckDuckGo 搜索网络信息。适用于查询实时信息、概念解释、新闻等。
    注意：国内网络可能需要代理。
    """
    try:
        # 使用 ddgs 库（pip install ddgs）
        from ddgs import DDGS
        
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=topn)
            texts = [f"{i+1}. {r['title']}: {r['body']}" 
                    for i, r in enumerate(results)]
            return "\n".join(texts) if texts else "未找到相关结果"
    except Exception as e:
        return f"搜索失败：{str(e)}。建议检查网络连接或使用模拟数据。"

@tool
def python_calculator(expression: str, precision: int = 2) -> str:
    """
    执行数学计算表达式。支持 +, -, *, /, **, 括号等。
    示例：expression="(2026 - 1956) * 12"
    """
    try:
        # 安全校验：只允许数学字符
        allowed = set('0123456789+-*/.() **% ')
        if not all(c in allowed for c in expression):
            return "错误：表达式包含非法字符"
        
        # 使用 eval 计算（生产环境应使用 asteval 等安全库）
        result = eval(expression)
        return f"{result:.{precision}f}"
    except Exception as e:
        return f"计算错误：{str(e)}"

@tool
def python_executor(code: str, timeout: int = 5) -> str:
    """
    执行 Python 代码并返回输出。用于数据处理、绘图（生成 base64 图片）等。
    注意：在安全沙箱中运行，禁止导入 os/sys/subprocess。
    """
    # 危险模块检查
    forbidden = ['import os', 'import sys', 'import subprocess', 
                   'exec(', 'eval(', '__import__', 'open(', 'file']
    for f in forbidden:
        if f in code:
            return f"错误：检测到危险操作 '{f}'，已禁止执行"
    
    try:
        # 使用受限环境执行（简单版，生产用 subprocess 沙箱）
        import io
        import contextlib
        
        # 捕获标准输出
        output_buffer = io.StringIO()
        
        # 创建受限 globals
        safe_globals = {
            "__builtins__": {
                "len": len, "range": range, "enumerate": enumerate,
                "print": print, "str": str, "int": int, "float": float,
                "list": list, "dict": dict, "sum": sum, "max": max, "min": min
            }
        }
        
        with contextlib.redirect_stdout(output_buffer):
            exec(code, safe_globals, {})
        
        output = output_buffer.getvalue()
        return output if output else "代码执行成功，无输出"
        
    except Exception as e:
        return f"代码执行错误：{str(e)}"

@tool
def get_current_year() -> int:
    """
    获取当前年份。适用于计算年龄、时效性判断等。
    """
    from datetime import datetime
    return datetime.now().year

@tool
def master_test() -> str:
    """
    测试
    """
    return "主人您好！"
