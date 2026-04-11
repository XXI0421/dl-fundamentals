import subprocess
import tempfile
import os
import sys
import json
import base64
import signal
import time
from typing import Optional, Dict, Any, List
from pathlib import Path
from .base import tool

class SecureSandbox:
    """进程级安全沙箱（支持 Linux/Mac/Windows）"""
    
    def __init__(self, 
                 timeout: int = 10,
                 max_memory_mb: int = 512,
                 allowed_modules: Optional[List[str]] = None):
        self.timeout = timeout
        self.max_memory = max_memory_mb * 1024 * 1024
        self.allowed_modules = allowed_modules or [
            'math', 'random', 'datetime', 'json', 're', 'statistics',
            'itertools', 'collections', 'functools', 'decimal', 'fractions',
            'typing', 'inspect', 'textwrap', 'string', 'hashlib', 'uuid'
        ]
        # 图表库特别许可
        self.chart_libs = ['matplotlib', 'matplotlib.pyplot', 'plt', 
                          'numpy', 'pandas', 'plotly', 'seaborn']
        self.allowed_modules.extend(self.chart_libs)
        
        # 危险代码模式（正则匹配）
        self.forbidden_patterns = [
            r'import\s+os\b', r'import\s+sys\b', r'import\s+subprocess\b',
            r'import\s+socket\b', r'import\s+requests\b', r'import\s+urllib',
            r'exec\s*\(', r'eval\s*\(', r'__import__\s*\(', 
            r'open\s*\(', r'file\s*\(', r'subprocess\.',
            r'os\.system', r'os\.popen', r'os\.remove', r'os\.unlink',
            r'shutil\.rmtree', r'wget\b', r'curl\b', r'rm\s+-rf',
            r'>\s*/etc/', r'>\s*/proc/', r'>\s*/sys/'
        ]
        import re
        self.forbidden_regex = [re.compile(p, re.IGNORECASE) for p in self.forbidden_patterns]
    
    def validate_code(self, code: str) -> tuple[bool, str]:
        """静态安全检查"""
        for pattern in self.forbidden_regex:
            if pattern.search(code):
                matched = pattern.search(code).group(0)
                return False, f"安全拦截：检测到危险代码模式 '{matched}'"
        return True, "OK"
    
    def execute(self, code: str, save_artifacts: bool = False, 
                work_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        在子进程中执行 Python 代码
        
        Returns:
            dict with keys: stdout, stderr, exit_code, artifacts, execution_time, success
        """
        is_safe, msg = self.validate_code(code)
        if not is_safe:
            return {'stdout': '', 'stderr': msg, 'exit_code': -1, 
                   'artifacts': [], 'execution_time': 0, 'success': False}
        
        # 创建临时工作目录
        temp_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp())
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # 构建安全头（注入限制）
        header = """
                import sys
                import os
                import builtins
                import datetime
                sys.setrecursionlimit(1000)
                sys.path = [p for p in sys.path if 'site-packages' not in p]

                # 禁用文件操作（简单粗暴但安全）
                def _blocked_open(*args, **kwargs):
                    raise PermissionError("文件操作已被禁止")

                builtins.open = _blocked_open

                # 监控资源
                import tracemalloc
                tracemalloc.start()
                """
        
        script_path = temp_dir / "script.py"
        script_path.write_text(header + "\n" + code, encoding='utf-8')
        
        try:
            start_time = time.time()
            
            # 构建命令（使用当前 Python 解释器）
            cmd = [sys.executable, str(script_path)]
            
            # 执行（跨平台兼容）
            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=str(temp_dir),
                    text=True,
                    encoding='utf-8'
                )
                
                stdout, stderr = process.communicate(timeout=self.timeout)
                exit_code = process.returncode
                
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                exit_code = -2
                stderr += f"\n[系统] 执行超时（>{self.timeout}秒），进程已强制终止"
            
            execution_time = time.time() - start_time
            
            # 收集产物（图表文件）
            artifacts = []
            if save_artifacts:
                for ext in ['*.png', '*.jpg', '*.jpeg', '*.pdf', '*.csv', '*.json']:
                    for file_path in temp_dir.glob(ext):
                        try:
                            with open(file_path, 'rb') as f:
                                content = f.read()
                                artifacts.append({
                                    'name': file_path.name,
                                    'type': file_path.suffix[1:],
                                    'size': len(content),
                                    'content': base64.b64encode(content).decode('utf-8')[:1000]  # 截断
                                })
                        except Exception as e:
                            print(f"读取产物失败 {file_path}: {e}")
            
            return {
                'stdout': stdout,
                'stderr': stderr,
                'exit_code': exit_code,
                'artifacts': artifacts,
                'execution_time': round(execution_time, 2),
                'success': exit_code == 0 and not stderr.strip()
            }
            
        except Exception as e:
            return {'stdout': '', 'stderr': f"系统错误: {str(e)}", 
                   'exit_code': -3, 'artifacts': [], 'execution_time': 0, 'success': False}
        finally:
            # 清理临时文件（保留产物文件用于调试）
            try:
                script_path.unlink(missing_ok=True)
            except:
                pass

@tool
def python_sandbox(code: str, generate_chart: bool = False, timeout: int = 10) -> str:
    """
    在安全沙箱中执行 Python 代码。支持数学计算、数据处理、生成图表。
    
    参数:
        code: Python 代码字符串。禁止：文件删除、网络请求、系统命令。
        generate_chart: 是否允许生成图表（代码需包含 plt.savefig('xxx.png')）
        timeout: 最大执行时间（秒，默认10秒，最大60秒）
    
    返回:
        执行结果、错误信息、或图表 base64 数据（如果生成）
    """
    timeout = min(timeout, 60)  # 硬性上限
    
    sandbox = SecureSandbox(timeout=timeout)
    
    # 如果用户想生成图表但代码里没有 savefig，自动注入
    if generate_chart and 'savefig' not in code:
        code += "\n\nimport matplotlib.pyplot as plt\nplt.savefig('chart.png', dpi=150, bbox_inches='tight')"
    
    result = sandbox.execute(code, save_artifacts=generate_chart)
    
    output_lines = []
    
    if result['stdout']:
        output_lines.append("[标准输出]\n" + result['stdout'])
    
    if result['stderr']:
        output_lines.append("[错误/警告]\n" + result['stderr'])
    
    if generate_chart and result['artifacts']:
        for art in result['artifacts']:
            if art['type'] in ['png', 'jpg', 'jpeg']:
                output_lines.append(f"[生成图表] {art['name']} (大小: {art['size']} bytes)")
                # 返回 base64 供前端显示（截断避免过长）
                output_lines.append(f"data:image/png;base64,{art['content'][:200]}...")
    
    if not output_lines:
        return f"代码执行完成（耗时 {result['execution_time']}s），无输出"
    
    return "\n\n".join(output_lines)

@tool
def calculator(expression: str, precision: int = 2) -> str:
    """
    安全计算器。支持数学表达式：+ - * / ** % ( ) abs round 等。
    示例：expression="(2026 - 1956) * 12 + 100"
    """
    allowed_chars = set('0123456789+-*/.() **% abs round ')
    if not all(c in allowed_chars for c in expression):
        return "错误：表达式包含非法字符（只允许数字和数学运算符）"
    
    try:
        # 使用安全 eval（限制 globals）
        safe_dict = {
            "abs": abs, "round": round, "max": max, "min": min,
            "__builtins__": None
        }
        result = eval(expression, safe_dict, {})
        return f"{result:.{precision}f}"
    except Exception as e:
        return f"计算错误：{str(e)}"
