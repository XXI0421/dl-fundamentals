# tools/base.py
import inspect
import json
import datetime
from typing import Callable, Dict, Any, get_type_hints, List, Optional
from docstring_parser import parse  # 需要：pip install docstring-parser

class Tool:
    def __init__(self, func: Callable):
        self.func = func
        self.name = func.__name__
        self.description = self._parse_docstring(func)
        self.schema = self._generate_schema(func)
    
    def _parse_docstring(self, func) -> str:
        """提取函数 docstring 作为工具描述"""
        doc = parse(func.__doc__ or "")
        return doc.short_description or func.__name__
    
    def _generate_schema(self, func) -> Dict[str, Any]:
        """从类型注解生成 JSON Schema（OpenAI Function Calling 格式）"""
        sig = inspect.signature(func)
        hints = get_type_hints(func)
        
        properties = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            param_type = hints.get(param_name, str)
            json_type = self._py_type_to_json(param_type)
            
            properties[param_name] = {
                "type": json_type,
                "description": f"参数 {param_name}"
            }
            
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }
    
    def _py_type_to_json(self, py_type) -> str:
        mapping = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object"
        }
        return mapping.get(py_type, "string")

def tool(func: Callable) -> Tool:
    """装饰器：将函数转为 Tool 对象"""
    return Tool(func)

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._history: list = []  # 记录工具增删历史
    
    def register(self, tool_obj: Tool, overwrite: bool = False) -> 'ToolRegistry':
        """注册工具，可选覆盖"""
        if tool_obj.name in self._tools and not overwrite:
            raise ValueError(f"工具 {tool_obj.name} 已存在，设置 overwrite=True 覆盖")
        
        self._tools[tool_obj.name] = tool_obj
        self._history.append({"action": "add", "tool": tool_obj.name, "time": datetime.datetime.now()})
        return self
    
    def unregister(self, tool_name: str) -> 'ToolRegistry':
        """动态删除工具"""
        if tool_name in self._tools:
            del self._tools[tool_name]
            self._history.append({"action": "remove", "tool": tool_name, "time": datetime.datetime.now()})
        return self
    
    def get_tool(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)
    
    def list_tools(self) -> List[str]:
        """列出所有可用工具"""
        return list(self._tools.keys())
    
    def get_schemas(self) -> List[Dict]:
        """获取 OpenAI 格式的工具定义"""
        return [t.schema for t in self._tools.values()]
    
    def load_tool_from_path(self, python_file: str):
        """
        动态加载外部工具（高级功能）
        允许在不重启 Agent 的情况下，加载新的 .py 文件作为工具
        """
        import importlib.util
        spec = importlib.util.spec_from_file_location("dynamic_tool", python_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # 查找模块中的 @tool 装饰器函数
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, Tool):
                self.register(attr)
                print(f"动态加载工具: {attr_name}")

