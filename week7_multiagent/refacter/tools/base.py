"""
工具基础类 - 定义工具注册和调用框架
"""
from typing import Dict, Callable, Any, List, Optional
import json
import inspect

class ToolRegistry:
    """工具注册表 - 管理可用工具的注册和调用"""
    
    def __init__(self):
        self.tools: Dict[str, Dict] = {}
    
    def register(
        self,
        func: Callable = None,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        schema: Optional[Dict] = None
    ):
        """
        注册工具
        
        支持两种调用方式：
        1. @registry.register
        2. registry.register(func, name="tool_name", description="...")
        """
        if func is None:
            # 返回装饰器模式
            def decorator(f):
                self._register_tool(f, name, description, schema)
                return f
            return decorator
        
        # 直接注册模式
        self._register_tool(func, name, description, schema)
    
    def _register_tool(
        self,
        func: Callable,
        name: Optional[str],
        description: Optional[str],
        schema: Optional[Dict]
    ):
        """内部注册逻辑"""
        tool_name = name or func.__name__
        
        # 自动生成schema
        if schema is None:
            schema = self._generate_schema(func, tool_name, description)
        
        self.tools[tool_name] = {
            "func": func,
            "description": description or func.__doc__ or "",
            "schema": schema
        }
    
    def _generate_schema(self, func: Callable, name: str, description: str) -> Dict:
        """从函数签名自动生成JSON schema"""
        signature = inspect.signature(func)
        parameters = {}
        
        for param_name, param in signature.parameters.items():
            param_type = "string"  # 默认类型
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int:
                    param_type = "integer"
                elif param.annotation == float:
                    param_type = "number"
                elif param.annotation == bool:
                    param_type = "boolean"
                elif param.annotation == dict:
                    param_type = "object"
                elif param.annotation == list:
                    param_type = "array"
            
            parameters[param_name] = {
                "type": param_type,
                "description": ""
            }
        
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description or "",
                "parameters": {
                    "type": "object",
                    "properties": parameters,
                    "required": list(parameters.keys())
                }
            }
        }
    
    def list_tools(self) -> List[str]:
        """列出所有已注册的工具名称"""
        return list(self.tools.keys())
    
    def get_tool(self, name: str) -> Optional[Dict]:
        """获取工具定义"""
        return self.tools.get(name)
    
    def execute(self, name: str, arguments: Dict[str, Any]) -> Any:
        """
        执行工具
        
        Args:
            name: 工具名称
            arguments: 参数字典
        
        Returns:
            工具执行结果
        
        Raises:
            ValueError: 如果工具不存在
        """
        if name not in self.tools:
            raise ValueError(f"工具 '{name}' 未注册")
        
        tool = self.tools[name]
        func = tool["func"]
        
        try:
            return func(**arguments)
        except Exception as e:
            return f"工具调用失败: {str(e)}"
    
    def get_tool_schemas(self) -> List[Dict]:
        """获取所有工具的schema列表，用于LLM调用"""
        schemas = []
        for name, tool in self.tools.items():
            schemas.append(tool["schema"])
        return schemas
