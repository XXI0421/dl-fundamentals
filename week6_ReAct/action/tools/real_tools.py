# tools/real_tools.py
import requests
import json
import os
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

@tool
def save_file(content: str, filename: str, directory: Optional[str] = None) -> str:
    """
    将内容保存到文件。适用于保存报告、图表数据、分析结果等。
    
    参数:
        content: 要保存的内容（字符串）
        filename: 文件名（如 'report.txt', 'data.json', 'chart.png'）
        directory: 保存目录（可选，默认为当前目录）
    
    返回:
        保存结果消息
    """
    try:
        # 安全检查：防止路径遍历攻击
        if '..' in filename or '/' in filename or '\\' in filename:
            return "错误：文件名不允许包含路径分隔符"
        
        # 限制文件大小（最大10MB）
        if len(content) > 10 * 1024 * 1024:
            return f"错误：文件大小超过限制（最大10MB，当前{len(content)//1024}KB）"
        
        # 确定保存路径
        if directory:
            save_path = os.path.join(directory, filename)
            # 确保目录存在
            os.makedirs(directory, exist_ok=True)
        else:
            save_path = filename
        
        # 写入文件
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return f"文件已成功保存到: {os.path.abspath(save_path)}"
    
    except Exception as e:
        return f"保存失败：{str(e)}"

@tool
def read_file(filename: str, directory: Optional[str] = None) -> str:
    """
    读取文件内容。适用于读取配置文件、数据文件等。
    
    参数:
        filename: 文件名
        directory: 文件所在目录（可选）
    
    返回:
        文件内容（前2000字符）
    """
    try:
        if '..' in filename or '/' in filename or '\\' in filename:
            return "错误：文件名不允许包含路径分隔符"
        
        if directory:
            file_path = os.path.join(directory, filename)
        else:
            file_path = filename
        
        if not os.path.exists(file_path):
            return f"错误：文件不存在: {file_path}"
        
        # 限制读取大小
        max_size = 2000
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read(max_size)
        
        if len(content) == max_size:
            content += "\n\n...（文件内容已截断）"
        
        return content
    
    except Exception as e:
        return f"读取失败：{str(e)}"
