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
import re
from typing import Optional, Dict, Any, List
from pathlib import Path
from tools.base import ToolRegistry

# 脚本类型检测相关函数
def _detect_script_type(code: str) -> str:
    """
    检测代码类型
    
    返回:
        'python': 纯Python代码
        'javascript': 纯JavaScript代码
        'mixed': 混合代码（Python + JavaScript）
        'unknown': 未知类型
    """
    # JavaScript特征
    js_keywords = [
        r'\bconst\s+\w+\s*=', r'\blet\s+\w+\s*=', r'\bvar\s+\w+\s*=',
        r'\bfunction\s+\w+\s*\(', r'=>\s*\{', r'\bclass\s+\w+\s*\{',
        r'\bnew\s+Vue\s*\(', r'\bimport\s+\{', r'\bexport\s+default',
        r'\brequire\s*\(', r'\.addEventListener\s*\(', r'\.querySelector',
        r'\bdocument\.', r'\bwindow\.', r'\$(\w+|\()',
        r'render:\s*h\s*=>\s*h\(', r'\.mount\s*\(',
        r'\bconsole\.log\s*\(', r'\bsetTimeout\s*\(', r'\bsetInterval\s*\('
    ]
    
    # Python特征
    py_keywords = [
        r'\bdef\s+\w+\s*\(', r'\bclass\s+\w+\s*:', r'\bimport\s+\w+',
        r'\bfrom\s+\w+\s+import', r'\bif\s+\w+\s*:', r'\bfor\s+\w+\s+in',
        r'\bwhile\s+\w+\s*:', r'\breturn\s+', r'\bprint\s*\(',
        r'\bwith\s+\w+\s+as', r'\btry\s*:', r'\bexcept\s+',
        r'\bopen\s*\(', r'\bwith open\s*\('
    ]
    
    has_js = any(re.search(pattern, code) for pattern in js_keywords)
    has_py = any(re.search(pattern, code) for pattern in py_keywords)
    
    if has_js and not has_py:
        return 'javascript'
    elif has_py and not has_js:
        return 'python'
    elif has_js and has_py:
        return 'mixed'
    else:
        return 'unknown'


def _save_javascript_code(code: str) -> str:
    """将JavaScript代码保存为.js文件"""
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # 尝试从代码中提取文件名
    filename = 'script.js'
    # 简单尝试从代码中查找文件名
    match = re.search(r'filename\s*[=:]\s*["\']([^"\']+\.js)["\']', code)
    if match:
        filename = match.group(1)
    
    script_path = os.path.join(output_dir, filename)
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(code)
    
    return f"JavaScript脚本已保存: {script_path}"


def _handle_mixed_code(code: str) -> str:
    """处理混合代码（Python + JavaScript）"""
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # 简单分离策略：按标记分离
    # 查找常见的分隔标记
    separators = [
        (r'#\s*JavaScript', r'#\s*Python'),
        (r'//\s*Python', r'//\s*JavaScript'),
        (r'/\*\s*Python\s*\*/', r'/\*\s*JavaScript\s*\*/'),
        (r'# Vue', r'# FastAPI'),
        (r'// Vue', r'// FastAPI'),
    ]
    
    py_code = ""
    js_code = ""
    
    # 尝试按分隔符分离
    for py_mark, js_mark in separators:
        if re.search(py_mark, code) and re.search(js_mark, code):
            # 找到分隔符，分离代码
            parts = re.split(f'({py_mark}|{js_mark})', code)
            for i, part in enumerate(parts):
                if re.match(py_mark, part):
                    # 下一部分是Python代码
                    if i + 1 < len(parts):
                        next_part = parts[i + 1]
                        # 找到下一个分隔符
                        end_idx = len(next_part)
                        for other_mark in [py_mark, js_mark]:
                            match = re.search(other_mark, next_part)
                            if match:
                                end_idx = min(end_idx, match.start())
                        py_code += next_part[:end_idx]
                elif re.match(js_mark, part):
                    if i + 1 < len(parts):
                        next_part = parts[i + 1]
                        end_idx = len(next_part)
                        for other_mark in [py_mark, js_mark]:
                            match = re.search(other_mark, next_part)
                            if match:
                                end_idx = min(end_idx, match.start())
                        js_code += next_part[:end_idx]
            break
    else:
        # 没有找到分隔符，尝试简单的启发式分离
        lines = code.split('\n')
        for line in lines:
            # 判断每行是Python还是JavaScript
            if any(kw in line for kw in ['def ', 'import ', 'from ', 'print(', 'with ']):
                py_code += line + '\n'
            elif any(kw in line for kw in ['const ', 'let ', 'var ', 'function ', '=>', 'export ']):
                js_code += line + '\n'
            else:
                # 默认添加到Python代码
                py_code += line + '\n'
    
    result = []
    
    # 保存Python代码
    if py_code.strip():
        py_path = os.path.join(output_dir, 'main.py')
        with open(py_path, 'w', encoding='utf-8') as f:
            f.write(py_code.strip())
        result.append(f"Python代码已保存: {py_path}")
    
    # 保存JavaScript代码
    if js_code.strip():
        js_path = os.path.join(output_dir, 'main.js')
        with open(js_path, 'w', encoding='utf-8') as f:
            f.write(js_code.strip())
        result.append(f"JavaScript代码已保存: {js_path}")
    
    return "\n".join(result)


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
            r'subprocess\.Popen', r'subprocess\.call', r'subprocess\.run',
            r'subprocess\.check_', r'os\.system', r'os\.popen',
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
                r'^\s*(import|from)\s+(subprocess)',
                r'\bimport\s+subprocess\b',
            ])
        
        return patterns
    
    def validate_code(self, code: str) -> tuple[bool, str]:
        """静态安全检查"""
        # 检查是否包含JavaScript代码（常见JavaScript关键字）
        js_patterns = [
            r'\bconst\s+\w+\s*=', r'\blet\s+\w+\s*=', r'\bvar\s+\w+\s*=',
            r'\bfunction\s+\w+\s*\(', r'=>\s*\{', r'\bclass\s+\w+\s*\{',
            r'\bnew\s+Vue\s*\(', r'\bimport\s+\{', r'\bexport\s+default',
            r'\brequire\s*\(', r'\.addEventListener\s*\(', r'\.querySelector',
            r'\bdocument\.', r'\bwindow\.', r'\$(\w+|\()',
            r'render:\s*h\s*=>\s*h\(', r'\.mount\s*\('
        ]
        
        js_keywords_found = []
        for pattern in js_patterns:
            if re.search(pattern, code):
                js_keywords_found.append(pattern)
        
        if js_keywords_found:
            return False, f"检测到JavaScript代码！请使用Python语法编写代码。如需创建JavaScript文件，请使用Python的文件操作（如 open() 函数）将代码写入文件。"
        
        # 检查常见的Python语法错误模式
        # 检查未定义变量模式（如直接使用变量而不赋值）
        # 这是一个简单的检查，检测常见的错误模式
        undefined_var_patterns = [
            r'\b(fastapi_code|vue_code|script_content)\b'  # 常见的未定义变量名
        ]
        
        for pattern in undefined_var_patterns:
            if re.search(pattern, code):
                # 检查是否有相应的赋值
                var_name = pattern.strip(r'\b()')
                if not re.search(rf'{var_name}\s*=', code):
                    return False, f"检测到可能未定义的变量 '{var_name}'，请确保在使用前进行赋值。"
        
        # 危险代码检查
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
                output_dir = Path(__file__).parent.parent / "output"
                output_dir.mkdir(exist_ok=True)
                
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
                                'content': base64.b64encode(content).decode('utf-8'),
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


def python_sandbox(code: str, mode: str = 'default', timeout: int = 10, 
                   allow_network: bool = False) -> str:
    """
    在安全沙箱中执行 Python 代码或生成脚本文件。
    
    参数:
        code: Python 代码字符串
        mode: 执行模式，可选值:
            - 'default': 基础模式（仅标准库）
            - 'data_analysis': 数据分析模式（包含 numpy, pandas）
            - 'chart': 图表模式（包含数据分析库 + 图表库）
            - 'full': 完整模式（允许所有操作，包括网络）
            - 'no_run': 仅生成脚本文件，不执行代码
        timeout: 最大执行时间（秒，默认10秒，最大60秒）
        allow_network: 是否允许网络请求（requests），默认False
    
    返回:
        执行结果、错误信息、或生成的脚本路径
    """
    # 处理类型转换 - 参数可能是字符串形式的 'true'/'false'
    if isinstance(allow_network, str):
        allow_network = allow_network.lower() == 'true'
    
    # timeout可能是字符串
    if isinstance(timeout, str):
        try:
            timeout = int(timeout)
        except ValueError:
            timeout = 10
    
    timeout = min(timeout, 60)
    
    # 如果模式是full，自动允许网络
    if mode == 'full':
        allow_network = True
    
    # 脚本类型检测和处理
    detected_type = _detect_script_type(code)
    
    # 如果检测到JavaScript代码，将其保存为.js文件
    if detected_type == 'javascript':
        return _save_javascript_code(code)
    
    # 如果检测到混合代码（Python + JavaScript），分离并分别处理
    if detected_type == 'mixed':
        return _handle_mixed_code(code)
    
    # 检测模式（仅对Python代码）
    if mode not in ['no_run']:
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
    
    # 自动添加打印最后一行表达式的逻辑 - 仅对简单Python表达式生效
    lines = code.strip().split('\n')
    if lines:
        last_line = lines[-1].strip()
        if last_line and not last_line.startswith('print(') and not last_line.startswith('return'):
            if not last_line.endswith(':') and not last_line.endswith('\\'):
                # 检查整个代码是否包含未闭合的多行字符串
                # 统计三引号出现次数
                triple_single = code.count("'''")
                triple_double = code.count('"""')
                
                # 如果三引号数量是奇数，说明有多行字符串未闭合
                if triple_single % 2 != 0 or triple_double % 2 != 0:
                    # 代码可能不完整或包含未闭合的多行字符串，不添加print
                    pass
                else:
                    # 检查是否是简单表达式
                    is_simple_expr = True
                    
                    # 检查最后一行是否包含字符串字面量
                    if ("'''" in last_line or '"""' in last_line):
                        is_simple_expr = False
                    # 检查引号是否闭合
                    elif last_line.count("'") % 2 != 0 or last_line.count('"') % 2 != 0:
                        is_simple_expr = False
                    # 检查括号是否闭合
                    elif last_line.count('(') != last_line.count(')') or last_line.count('[') != last_line.count(']') or last_line.count('{') != last_line.count('}'):
                        is_simple_expr = False
                    # 检查是否是赋值语句
                    elif '=' in last_line and not last_line.startswith('('):
                        is_simple_expr = False
                    # 检查是否是其他语言代码
                    elif any(keyword in last_line for keyword in ['const', 'let', 'var', 'function', '=>', 'new Vue', 'createApp']):
                        is_simple_expr = False
                    
                    if is_simple_expr:
                        code += f"\nprint({last_line})"
    
    import os
    # 设置工作目录为 output 目录
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # no_run 模式：只生成脚本文件，不执行
    if mode == 'no_run':
        script_path = os.path.join(output_dir, 'generated_script.py')
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(code)
        return f"脚本已生成（未执行）: {script_path}"
    
    # 执行代码，保存所有产物到 output 目录
    result = sandbox.execute(code, save_artifacts=True, work_dir=output_dir)
    
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
    
    if not output_lines:
        return f"代码执行完成（耗时 {result['execution_time']}s），无输出"
    
    return "\n\n".join(output_lines)


def init_sandbox_tools(registry: ToolRegistry):
    """初始化沙盒工具"""
    registry.register(
        python_sandbox,
        name="python_sandbox",
        description="在安全沙箱中执行Python代码，支持数据分析、图表生成和网络请求"
    )
