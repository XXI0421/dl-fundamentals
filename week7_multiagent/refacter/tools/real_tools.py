"""
实际工具实现 - 文件读写、计算器等
"""
import os
import math
from datetime import datetime
from pathlib import Path
from tools.base import ToolRegistry

def get_current_year() -> str:
    """获取当前年份"""
    return f"当前年份: {datetime.now().year}"

def python_calculator(expression: str) -> str:
    """
    数学计算器 - 安全执行数学表达式
    
    支持的运算：加减乘除、幂运算、三角函数、对数等
    """
    # 安全检查：只允许数学相关的函数和常量
    allowed_names = {
        'abs': abs, 'pow': pow, 'round': round,
        'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
        'asin': math.asin, 'acos': math.acos, 'atan': math.atan,
        'sinh': math.sinh, 'cosh': math.cosh, 'tanh': math.tanh,
        'log': math.log, 'log10': math.log10, 'log2': math.log2,
        'sqrt': math.sqrt, 'exp': math.exp,
        'pi': math.pi, 'e': math.e,
        '+': lambda a, b: a + b,
        '-': lambda a, b: a - b,
        '*': lambda a, b: a * b,
        '/': lambda a, b: a / b if b != 0 else float('inf'),
        '**': lambda a, b: a ** b
    }
    
    try:
        # 使用eval但限制命名空间
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算失败: {str(e)}"

def save_file(filename: str, content: str) -> str:
    """
    保存内容到文件
    
    Args:
        filename: 文件名（仅支持纯文件名，不支持路径）
    """
    try:
        # 安全清理：只保留字母数字和少量特殊字符，防止路径遍历攻击
        # 首先使用 basename 获取纯文件名，去除路径信息
        base_name = os.path.basename(filename)
        
        # 过滤非法字符，只允许字母、数字、下划线、点和连字符
        safe_filename = "".join(c for c in base_name if c.isalnum() or c in "._-").strip()
        
        if not safe_filename:
            safe_filename = "output.txt"
        
        # 确保目录存在
        output_dir = Path(os.environ.get('OUTPUT_DIR', './output'))
        output_dir.mkdir(exist_ok=True)
        
        # 始终使用 output 目录，防止路径遍历攻击
        full_path = output_dir / safe_filename
        
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return f"文件已保存: {full_path}"
    except Exception as e:
        return f"保存文件失败: {str(e)}"

def read_file(filename: str) -> str:
    """
    读取文件内容
    
    Args:
        filename: 文件名（仅支持纯文件名，不支持路径）
    """
    try:
        # 安全清理：只保留字母数字和少量特殊字符，防止路径遍历攻击
        # 首先使用 basename 获取纯文件名，去除路径信息
        base_name = os.path.basename(filename)
        
        # 过滤非法字符，只允许字母、数字、下划线、点和连字符
        safe_filename = "".join(c for c in base_name if c.isalnum() or c in "._-").strip()
        
        if not safe_filename:
            return "文件名无效"
        
        # 始终使用 output 目录，防止路径遍历攻击
        output_dir = Path(os.environ.get('OUTPUT_DIR', './output'))
        full_path = output_dir / safe_filename
        
        if not full_path.exists():
            return f"文件不存在: {full_path}"
        
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return content
    except Exception as e:
        return f"读取文件失败: {str(e)}"

def get_system_info() -> str:
    """获取系统信息"""
    info = []
    info.append(f"操作系统: {os.name}")
    info.append(f"当前目录: {os.getcwd()}")
    info.append(f"Python版本: {os.sys.version.split()[0]}")
    return "\n".join(info)

def web_search(query: str) -> str:
    """
    网络搜索 - 使用搜索引擎获取最新信息
    
    Args:
        query: 搜索关键词
    """
    try:
        import requests
        from bs4 import BeautifulSoup
        
        # 使用Bing搜索API
        url = "https://www.bing.com/search"
        params = {"q": query, "count": 5}
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        results = []
        
        for item in soup.find_all('li', class_='b_algo')[:5]:
            title = item.find('h2').get_text(strip=True) if item.find('h2') else ""
            link = item.find('a')['href'] if item.find('a') else ""
            desc = item.find('p').get_text(strip=True) if item.find('p') else ""
            
            if title and link:
                results.append(f"【{title}】\n{desc}\n链接: {link}\n")
        
        if results:
            return "\n".join(results)
        else:
            return f"未找到关于 '{query}' 的搜索结果"
            
    except ImportError:
        return "错误：需要安装 requests 和 beautifulsoup4 库"
    except Exception as e:
        return f"搜索失败: {str(e)}"

def init_default_tools(registry: ToolRegistry):
    """初始化默认工具集"""
    registry.register(get_current_year, description="获取当前年份")
    registry.register(python_calculator, description="数学计算器，支持三角函数、对数等")
    registry.register(save_file, description="保存内容到文件")
    registry.register(read_file, description="读取文件内容")
    registry.register(get_system_info, description="获取系统信息")
    registry.register(web_search, description="网络搜索，获取最新信息和新闻")
