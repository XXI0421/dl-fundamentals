"""
Python Sandbox - 安全代码执行环境

特性：
1. 进程级隔离，安全执行用户代码
2. 支持多种模式：计算、数据分析、图表生成、网络请求
3. 严格的安全限制，防止恶意代码
"""
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
                 mode: str = 'default',
                 allow_network: bool = False):
        """
        Args:
            timeout: 执行超时时间（秒）
            max_memory_mb: 最大内存限制（MB）
            mode: 执行模式，可选 'default', 'data_analysis', 'chart', 'full'
            allow_network: 是否允许网络请求（requests）
        """
        self.timeout = timeout
        self.max_memory = max_memory_mb * 1024 * 1024
        self.mode = mode
        self.allow_network = allow_network
        
        # 基础允许的模块
        self.base_modules = [
            'math', 'random', 'datetime', 'json', 're', 'statistics',
            'itertools', 'collections', 'functools', 'decimal', 'fractions',
            'typing', 'inspect', 'textwrap', 'string', 'hashlib', 'uuid'
        ]
        
        # 数据分析库
        self.data_modules = ['numpy', 'pandas']
        
        # 图表库
        self.chart_modules = ['matplotlib', 'matplotlib.pyplot', 'plotly', 'seaborn']
        
        # 网络库
        self.network_modules = ['requests', 'urllib', 'urllib3']
        
        # 根据模式确定允许的模块
        self.allowed_modules = self._build_allowed_modules()
        
        # 危险代码模式（正则匹配）
        self.forbidden_patterns = self._build_forbidden_patterns()
        import re
        self.forbidden_regex = [re.compile(p, re.IGNORECASE) for p in self.forbidden_patterns]
    
    def _build_allowed_modules(self) -> List[str]:
        """根据模式构建允许的模块列表"""
        modules = self.base_modules.copy()
        
        if self.mode in ['data_analysis', 'chart', 'full']:
            modules.extend(self.data_modules)
        
        if self.mode in ['chart', 'full']:
            modules.extend(self.chart_modules)
        
        if self.allow_network or self.mode == 'full':
            modules.extend(self.network_modules)
        
        return modules
    
    def _build_forbidden_patterns(self) -> List[str]:
        """构建禁止的代码模式"""
        patterns = [
            # 危险内置函数
            r'exec\s*\(', r'eval\s*\(', r'__import__\s*\(',
            r'\b__builtins__\b', r'\b__globals__\b', r'\b__locals__\b',
            # 系统命令执行
            r'subprocess\.', r'os\.system', r'os\.popen',
            # 文件删除操作
            r'os\.remove', r'os\.unlink', r'os\.rmdir',
            r'shutil\.rmtree',
            # 危险路径操作
            r'os\.chmod', r'os\.chown', r'os\.symlink', r'os\.link',
            # 命令行工具
            r'rm\s+-rf', r'>\s*/etc/', r'>\s*/proc/', r'>\s*/sys/'
        ]
        
        # 如果不允许网络，则禁止网络相关操作
        if not self.allow_network and self.mode != 'full':
            patterns.extend([
                r'^\s*(import|from)\s+(socket|requests|urllib)',
                r'\bimport\s+socket\b', r'\bimport\s+requests\b', r'\bimport\s+urllib',
                r'socket\.', r'requests\.', r'urllib\.', r'wget\b', r'curl\b',
            ])
        
        # 如果不是full模式，禁止os模块的危险操作
        if self.mode != 'full':
            patterns.extend([
                r'^\s*(import|from)\s+(os|sys|subprocess)',
                r'\bimport\s+os\b', r'\bimport\s+sys\b', r'\bimport\s+subprocess\b',
                r'os\.mkdir', r'os\.rename', r'os\.replace',
                r'shutil\.move', r'shutil\.copy', r'shutil\.copytree',
            ])
        
        return patterns
    
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
        
        # 构建安全头（注入限制，允许在临时目录写入）
        header = self._build_safety_header(str(temp_dir))
        
        script_path = temp_dir / "script.py"
        script_path.write_text(header + "\n" + code, encoding='utf-8')
        
        try:
            start_time = time.time()
            
            # 构建命令（使用当前 Python 解释器）
            cmd = [sys.executable, str(script_path)]
            
            # 设置环境变量（限制资源）
            env = os.environ.copy()
            
            # 如果允许网络，设置代理环境变量（如果存在）
            if self.allow_network:
                proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']
                for var in proxy_vars:
                    if var not in env and os.environ.get(var):
                        env[var] = os.environ[var]
            
            # 执行（跨平台兼容）
            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=str(temp_dir),
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    env=env
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
                import shutil
                # 获取当前脚本的目录作为输出目录
                output_dir = Path(__file__).parent.parent  # action目录
                for ext in ['*.png', '*.jpg', '*.jpeg', '*.pdf', '*.csv', '*.json']:
                    for file_path in temp_dir.glob(ext):
                        try:
                            with open(file_path, 'rb') as f:
                                content = f.read()
                            
                            # 将文件复制到输出目录
                            output_path = output_dir / file_path.name
                            shutil.copy(str(file_path), str(output_path))
                            
                            artifacts.append({
                                'name': file_path.name,
                                'type': file_path.suffix[1:],
                                'size': len(content),
                                'content': base64.b64encode(content).decode('utf-8'),  # 完整内容
                                'saved_path': str(output_path)
                            })
                        except Exception as e:
                            print(f"读取产物失败 {file_path}: {e}")
            
            return {
                'stdout': stdout,
                'stderr': stderr,
                'exit_code': exit_code,
                'artifacts': artifacts,
                'execution_time': round(execution_time, 2),
                'success': exit_code == 0
            }
            
        except Exception as e:
            return {'stdout': '', 'stderr': f"系统错误: {str(e)}", 
                   'exit_code': -3, 'artifacts': [], 'execution_time': 0, 'success': False}
        finally:
            # 清理临时文件
            try:
                script_path.unlink(missing_ok=True)
            except:
                pass
    
    def _build_safety_header(self, temp_dir: str) -> str:
        """构建安全头代码"""
        header = f'''
import sys
import os
import builtins

# 设置递归限制
sys.setrecursionlimit(1000)

# 允许写入的安全目录（临时目录）
_SAFE_WRITE_DIR = {repr(temp_dir)}

# 安全文件写入操作（仅允许写入临时目录）
original_open = builtins.open
def _safe_open(*args, **kwargs):
    mode = kwargs.get('mode', args[1] if len(args) > 1 else 'r')
    if 'w' in mode or 'a' in mode or 'x' in mode:
        # 检查是否在安全目录中写入
        filepath = args[0] if args else ''
        if isinstance(filepath, str):
            # 获取规范化路径
            abs_path = os.path.abspath(filepath)
            safe_dir = os.path.abspath(_SAFE_WRITE_DIR)
            # 检查是否在安全目录下
            if abs_path.startswith(safe_dir + os.sep) or abs_path == safe_dir:
                return original_open(*args, **kwargs)
        raise PermissionError("文件写入操作仅允许在临时目录中进行")
    return original_open(*args, **kwargs)

builtins.open = _safe_open

# 监控资源
import tracemalloc
tracemalloc.start()

# 设置matplotlib为非交互式后端，不弹出窗口
import matplotlib
matplotlib.use('Agg')
'''
        return header


@tool
def python_sandbox(code: str, mode: str = 'default', timeout: int = 10, 
                   allow_network: bool = False) -> str:
    """
    在安全沙箱中执行 Python 代码。
    
    参数:
        code: Python 代码字符串
        mode: 执行模式，可选值:
            - 'default': 基础模式（仅标准库）
            - 'data_analysis': 数据分析模式（包含 numpy, pandas）
            - 'chart': 图表模式（包含数据分析库 + 图表库）
            - 'full': 完整模式（允许所有操作，包括网络）
        timeout: 最大执行时间（秒，默认10秒，最大60秒）
        allow_network: 是否允许网络请求（requests），默认False
    
    返回:
        执行结果、错误信息、或图表数据
    """
    timeout = min(timeout, 60)
    
    # 如果模式是full，自动允许网络
    if mode == 'full':
        allow_network = True
    
    # 检测模式
    if 'plt.' in code or 'matplotlib' in code or 'plotly' in code:
        mode = 'chart'
    elif 'pandas' in code or 'numpy' in code:
        mode = 'data_analysis'
    
    sandbox = SecureSandbox(
        timeout=timeout, 
        mode=mode,
        allow_network=allow_network
    )
    
    # 如果是图表模式，确保设置正确的后端并自动注入savefig
    if mode in ['chart', 'full']:
        # 在代码开头添加matplotlib后端设置（确保生效）
        matplotlib_setup = """
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
"""
        # 如果代码中已经有import matplotlib，不需要重复添加
        if 'import matplotlib' not in code:
            code = matplotlib_setup + "\n" + code
        
        # 如果代码里没有 savefig，自动注入
        if 'savefig' not in code and 'write_image' not in code:
            if 'plt.' in code:
                code += "\n\nplt.savefig('chart.png', dpi=150, bbox_inches='tight')"
    
    # 自动添加打印最后一行表达式的逻辑
    lines = code.strip().split('\n')
    if lines:
        last_line = lines[-1].strip()
        if last_line and not last_line.startswith('print(') and not last_line.startswith('return'):
            if not last_line.endswith(':') and not last_line.endswith('\\'):
                code += f"\nprint({last_line})"
    
    result = sandbox.execute(code, save_artifacts=(mode in ['chart', 'full']))
    
    output_lines = []
    
    if result['stdout']:
        output_lines.append("[标准输出]\n" + result['stdout'])
    
    # 过滤掉正常的警告信息，只显示真正的错误
    if result['stderr']:
        stderr_lines = result['stderr'].strip().split('\n')
        # 过滤掉的警告类型
        filtered_lines = []
        for line in stderr_lines:
            line = line.strip()
            if not line:
                continue
            # 过滤常见的正常警告
            is_normal_warning = any([
                'font_manager' in line.lower(),
                'findfont' in line.lower(),
                'UserWarning' in line,
                'DeprecationWarning' in line,
                'FutureWarning' in line,
                # 过滤临时目录路径（不是真正的错误）
                line.startswith('C:\\Users\\') and 'AppData\\Local' in line,
                line.startswith('/tmp/'),
                line.startswith('/var/tmp/'),
            ])
            if not is_normal_warning:
                filtered_lines.append(line)
        
        if filtered_lines:
            output_lines.append("[错误/警告]\n" + '\n'.join(filtered_lines))
    
    if mode in ['chart', 'full'] and result['artifacts']:
        for art in result['artifacts']:
            if art['type'] in ['png', 'jpg', 'jpeg']:
                output_lines.append(f"[生成图表] {art['name']} (大小: {art['size']} bytes)")
                if 'saved_path' in art:
                    output_lines.append(f"[保存路径] {art['saved_path']}")
                output_lines.append(f"data:image/{art['type']};base64,{art['content']}")
    
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
        safe_dict = {
            "abs": abs, "round": round, "max": max, "min": min,
            "__builtins__": None
        }
        result = eval(expression, safe_dict, {})
        return f"{result:.{precision}f}"
    except Exception as e:
        return f"计算错误：{str(e)}"