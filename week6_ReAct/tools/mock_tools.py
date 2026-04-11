# tools/mock_tools.py - 用于 Day 1 验证，无需真实 API

def mock_search(query: str) -> str:
    """虚拟搜索工具 - 返回预设答案"""
    knowledge_base = {
        "python": "Python 是由 Guido van Rossum 于 1991 年创建的编程语言。",
        "guido": "Guido van Rossum 是 Python 创始人，1956 年 1 月 31 日出生于荷兰。",
        "2024": "当前是 2024 年。",
        "2026": "当前是 2026 年。"
    }
    
    # 简单关键词匹配
    for key, value in knowledge_base.items():
        if key in query.lower():
            return value
    return f"搜索结果：关于 '{query}' 的模拟信息（无具体数据）"

def mock_calculator(expr: str) -> str:
    """虚拟计算器 - 安全计算简单表达式"""
    try:
        # 白名单：只允许数字和运算符
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expr):
            return "错误：表达式包含非法字符"
        
        # 使用 eval 计算（仅演示用，生产环境需用 ast.literal_eval 或安全解析器）
        result = eval(expr)
        return f"计算结果：{result}"
    except Exception as e:
        return f"计算错误：{str(e)}"

def mock_lookup(keyword: str) -> str:
    """虚拟知识库查询"""
    data = {
        "git": "Git 是 Linus Torvalds 于 2005 年创建的分布式版本控制系统。",
        "github": "GitHub 是全球最大的代码托管平台，拥有超过 1 亿开发者。"
    }
    return data.get(keyword.lower(), f"未找到 '{keyword}' 的信息")
