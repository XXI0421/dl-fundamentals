"""
ReAct Agent - 实现思考-行动循环的智能Agent
"""
import re
from typing import List, Dict, Any, Optional
from xml.etree import ElementTree as ET
from llm_client import BaseLLMClient, get_llm_client
from memory.short_term import ShortTermMemory
from tools.base import ToolRegistry
from tools.real_tools import init_default_tools
from tools.python_sandbox import init_sandbox_tools
from tools.logger import get_logger


class ReActAgent:
    """
    ReAct Agent实现
    
    核心功能：
    1. 思考-行动循环
    2. 工具调用解析
    3. 记忆管理
    """
    
    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        tools: Optional[ToolRegistry] = None,
        max_iterations: int = 5,
        verbose: bool = True
    ):
        self.llm_client = llm_client or get_llm_client()
        self.tools = tools or self._create_default_tools()
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        # 日志记录器
        self.logger = get_logger()
        
        # 短期记忆
        self.short_memory = ShortTermMemory()
        
        # 统计信息
        self.stats = {
            "total_steps": 0,
            "successful_steps": 0,
            "failed_steps": 0,
            "tools_used": set()
        }
        
        # 执行日志
        self.execution_log = []
    
    def _create_default_tools(self) -> ToolRegistry:
        """创建默认工具注册器"""
        registry = ToolRegistry()
        init_default_tools(registry)
        init_sandbox_tools(registry)
        return registry
    
    def _parse_tool_calls(self, content: str) -> List[Dict]:
        """解析工具调用 - 支持XML格式"""
        tool_calls = []
        
        # 查找所有<tool_call>标签
        pattern = r'<tool_call>(.*?)</tool_call>'
        matches = re.findall(pattern, content, re.DOTALL)
        
        self.logger.info(f"🔍 发现 {len(matches)} 个工具调用标签")
        
        for i, match in enumerate(matches):
            cleaned_match = match.strip()
            self.logger.debug(f"📝 原始内容[{i+1}]: {cleaned_match[:100]}...")
            
            # 尝试解析XML
            parsed = self._try_parse_xml_tool_call(cleaned_match)
            if parsed:
                tool_calls.extend(parsed)
        
        self.logger.info(f"📊 最终解析出 {len(tool_calls)} 个有效工具调用")
        return tool_calls
    
    def _try_parse_xml_tool_call(self, content: str) -> List[Dict]:
        """尝试解析XML格式的工具调用"""
        # 方法0: 先尝试简单匹配（处理常见的正确XML格式）
        try:
            # 直接匹配 <function name="xxx">...</function> 模式
            func_pattern = r'<function\s+name\s*=\s*["\']([^"\']+)["\']([\s\S]*?)</function>'
            func_match = re.search(func_pattern, content, re.DOTALL)
            
            if func_match:
                tool_name = func_match.group(1)
                inner_content = func_match.group(2)
                
                # 先查找 <arguments> 标签内的内容
                args_pattern = r'<arguments>([\s\S]*?)</arguments>'
                args_match = re.search(args_pattern, inner_content, re.DOTALL)
                
                if args_match:
                    args_content = args_match.group(1)
                else:
                    # 如果没有 <arguments> 标签，直接使用内部内容
                    args_content = inner_content
                
                # 提取参数（在 arguments 内查找 <param>value</param> 模式）
                arguments = {}
                param_pattern = r'<(\w+)>([\s\S]*?)</\1>'
                params = re.findall(param_pattern, args_content)
                
                for key, value in params:
                    arguments[key.strip()] = value.strip()
                
                # 如果参数为空，尝试从整个内容中提取（处理不标准的格式）
                if not arguments:
                    params = re.findall(param_pattern, inner_content)
                    for key, value in params:
                        if key != 'arguments':  # 跳过 arguments 标签本身
                            arguments[key.strip()] = value.strip()
                
                if arguments:
                    self.logger.debug(f"✅ 简单匹配成功: {tool_name}")
                    return [{"name": tool_name, "arguments": arguments}]
        except Exception as e:
            self.logger.debug(f"简单匹配失败: {str(e)[:20]}")
        
        # 方法1: 尝试解析<function>标签
        try:
            # 先尝试不转义的原始内容
            xml_content = f"<root>{content}</root>"
            root = ET.fromstring(xml_content)
            
            tool_calls = []
            
            # 查找所有function标签
            for func_elem in root.findall('.//function'):
                tool_name = func_elem.get('name', '')
                
                if not tool_name:
                    name_elem = func_elem.find('name')
                    if name_elem is not None:
                        tool_name = name_elem.text or ''
                
                if not tool_name:
                    continue
                
                arguments = {}
                
                # 查找arguments标签
                args_elem = func_elem.find('arguments')
                if args_elem is not None:
                    for child in args_elem:
                        arguments[child.tag] = child.text or ''
                
                # 对参数值中的特殊字符进行反转义
                for key, value in arguments.items():
                    if isinstance(value, str):
                        arguments[key] = self._unescape_xml_special_chars(value)
                
                tool_calls.append({"name": tool_name, "arguments": arguments})
            
            return tool_calls
        except ET.ParseError as e:
            self.logger.warning(f"⚠️ XML解析失败: {str(e)[:30]}")
        
        # 方法2: 使用正则提取工具名和参数
        try:
            # 匹配 <function name="xxx"> 或 <function><name>xxx</name>
            name_pattern = r'<function[^>]*name\s*=\s*["\']([^"\']+)["\']'
            name_match = re.search(name_pattern, content)
            
            if not name_match:
                name_pattern2 = r'<name>([^<]+)</name>'
                name_match = re.search(name_pattern2, content)
            
            if name_match:
                tool_name = name_match.group(1)
                arguments = {}
                
                # 提取arguments中的键值对
                args_pattern = r'<arguments>(.*?)</arguments>'
                args_match = re.search(args_pattern, content, re.DOTALL)
                
                if args_match:
                    args_content = args_match.group(1)
                    # 匹配 <key>value</key> - 使用更稳健的方式处理多行内容
                    param_pattern = r'<(\w+)>(.*?)</\1>'
                    matches = re.findall(param_pattern, args_content, re.DOTALL)
                    for key, value in matches:
                        arguments[key.strip()] = self._unescape_xml_special_chars(value.strip())
                
                return [{"name": tool_name, "arguments": arguments}]
        except Exception as e:
            self.logger.error(f"❌ 正则解析失败: {str(e)[:30]}")
        
        # 方法3: 如果上述方法都失败，尝试更直接的逐参数提取（仅在arguments标签内）
        try:
            name_pattern = r'<function[^>]*name\s*=\s*["\']([^"\']+)["\']'
            name_match = re.search(name_pattern, content)
            
            if not name_match:
                name_pattern2 = r'<name>([^<]+)</name>'
                name_match = re.search(name_pattern2, content)
            
            if name_match:
                tool_name = name_match.group(1)
                arguments = {}
                
                # 先找到arguments标签内的内容
                args_pattern = r'<arguments>(.*?)</arguments>'
                args_match = re.search(args_pattern, content, re.DOTALL)
                
                if args_match:
                    args_content = args_match.group(1)
                    # 只在arguments内部提取参数
                    common_params = ['code', 'content', 'filename', 'expression', 
                                    'mode', 'timeout', 'allow_network', 'url', 'query']
                    
                    for param in common_params:
                        pattern = rf'<{param}>([\s\S]*?)</{param}>'
                        match = re.search(pattern, args_content, re.DOTALL)
                        if match:
                            arguments[param] = self._unescape_xml_special_chars(match.group(1).strip())
                
                if arguments:
                    return [{"name": tool_name, "arguments": arguments}]
        except Exception as e:
            self.logger.error(f"❌ 方法3解析失败: {str(e)[:30]}")
        
        return []
    
    def _escape_xml_special_chars(self, content: str) -> str:
        """转义XML特殊字符 - 只转义标签内容，不转义标签本身"""
        escape_map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&apos;'
        }
        
        # 使用正则表达式，只在标签内容中（不在尖括号内）转义特殊字符
        result = content
        
        # 处理标签内的内容（在>和<之间的内容），支持多行
        def escape_content(match):
            content = match.group(1)
            for char, escaped in escape_map.items():
                content = content.replace(char, escaped)
            return f">{content}<"
        
        # 匹配 >内容< 的模式，支持多行（使用[\s\S]*?匹配任意字符包括换行）
        result = re.sub(r'>([\s\S]*?)<', escape_content, result)
        
        return result
    
    def _unescape_xml_special_chars(self, content: str) -> str:
        """反转义XML特殊字符"""
        escape_map = {
            '&amp;': '&',
            '&lt;': '<',
            '&gt;': '>',
            '&quot;': '"',
            '&apos;': "'"
        }
        result = content
        for escaped, char in escape_map.items():
            result = result.replace(escaped, char)
        return result
    
    def _validate_tool_arguments(self, tool_name: str, arguments: dict) -> dict:
        """验证并过滤工具参数，只保留工具支持的参数"""
        # 定义每个工具支持的参数
        tool_params = {
            'python_sandbox': ['code', 'mode', 'timeout', 'allow_network'],
            'save_file': ['content', 'filename'],
            'read_file': ['filename'],
            'python_calculator': ['expression'],
            'get_current_year': [],
            'get_system_info': [],
        }
        
        # 获取工具支持的参数列表
        supported_params = tool_params.get(tool_name, [])
        
        # 如果工具不在列表中，返回所有参数（保持向后兼容）
        if not supported_params:
            return arguments
        
        # 过滤参数，只保留支持的参数
        validated = {}
        for key, value in arguments.items():
            if key in supported_params:
                validated[key] = value
            else:
                self.logger.warning(f"⚠️ 工具 {tool_name} 不支持参数 '{key}'，已忽略")
        
        return validated
    
    def _clean_invalid_tool_calls(self, content: str) -> str:
        """清理无效的工具调用标签"""
        # 移除 <tool_call>...</tool_call> 标签及其内容
        content = re.sub(r'<tool_call>.*?</tool_call>', '', content, flags=re.DOTALL)
        
        # 移除可能残留的无效标签
        content = re.sub(r'<\s*function[^>]*>.*?</\s*function\s*>', '', content, flags=re.DOTALL)
        content = re.sub(r'<\s*arguments[^>]*>.*?</\s*arguments\s*>', '', content, flags=re.DOTALL)
        
        # 移除孤立的工具标签（如 <python_calculator>...</python_calculator>）
        content = re.sub(r'<\s*(\w+)\s*>.*?</\s*\1\s*>', '', content, flags=re.DOTALL)
        
        # 清理多余的空白字符
        content = re.sub(r'\s+', ' ', content).strip()
        
        return content
    
    def _parse_arguments_from_string(self, args_str: str) -> Dict:
        """从字符串中解析参数（XML格式）"""
        arguments = {}
        
        # 尝试XML格式解析
        try:
            xml_content = f"<root>{args_str}</root>"
            root = ET.fromstring(xml_content)
            for child in root:
                arguments[child.tag] = child.text or ''
            return arguments
        except ET.ParseError:
            pass
        
        # 尝试简单的键值对解析
        try:
            key_value_pattern = r'(\w+)\s*=\s*["\']?([^"\']*)["\']?'
            for k, v in re.findall(key_value_pattern, args_str):
                arguments[k.strip()] = v.strip()
        except:
            pass
        
        return arguments
    
    def run(self, user_input: str) -> str:
        """
        执行Agent主循环
        
        Args:
            user_input: 用户输入
        
        Returns:
            最终响应
        """
        self.execution_log = []
        self.stats["total_steps"] = 0
        self.stats["successful_steps"] = 0
        self.stats["failed_steps"] = 0
        self.stats["tools_used"] = set()
        
        # 添加用户输入到短期记忆
        self.short_memory.add_message("user", user_input)
        
        for iteration in range(self.max_iterations):
            # 构建prompt
            prompt = self._build_prompt()
            
            # 调用LLM
            tool_schemas = self.tools.get_tool_schemas()
            response = self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                tools=tool_schemas,
                tool_choice="auto"
            )
            
            self.stats["total_steps"] += 1
            
            content = response.get("content", "") or ""
            
            # 处理工具调用 - 仅支持XML格式
            tool_calls = []
            if "<tool_call>" in content:
                tool_calls = self._parse_tool_calls(content)
            
            if tool_calls:
                self.logger.info(f"🚀 开始执行 {len(tool_calls)} 个工具调用")
                all_successful = True
                
                for j, tool_call in enumerate(tool_calls):
                    tool_name = tool_call.get("name", "")
                    self.logger.info(f"🔧 执行工具[{j+1}]: {tool_name}")
                    try:
                        arguments = tool_call.get("arguments", {})
                        if isinstance(arguments, str):
                            arguments = self._parse_arguments_from_string(arguments)
                        args_str = str(arguments)[:100]
                        self.logger.debug(f"📋 参数: {args_str}...")
                    except Exception as e:
                        arguments = {}
                        self.logger.warning(f"⚠️ 参数解析失败: {str(e)}")
                    
                    # 参数验证：确保只传递工具支持的参数
                    validated_args = self._validate_tool_arguments(tool_name, arguments)
                    
                    # 执行工具
                    result = self.tools.execute(tool_name, validated_args)
                    self.stats["tools_used"].add(tool_name)
                    
                    # 使用专门的日志方法记录完整的工具调用信息
                    self.logger.log_tool_call(tool_name, arguments, str(result))
                    
                    # 记录日志
                    self.execution_log.append({
                        "step": iteration + 1,
                        "action": f"工具调用: {tool_name}",
                        "arguments": arguments,
                        "result": str(result)[:200]
                    })
                    
                    if "失败" in str(result) or "错误" in str(result):
                        self.stats["failed_steps"] += 1
                        self.short_memory.add_message("assistant", f"工具调用失败: {tool_name}")
                        all_successful = False
                    else:
                        self.stats["successful_steps"] += 1
                        self.short_memory.add_message("assistant", f"工具调用成功[{tool_name}]: {result}")
                
                # 如果所有工具调用都成功，直接生成总结回答
                if all_successful:
                    summary_prompt = f"""
根据对话历史和工具执行结果，用自然、友好的语言总结回答用户的问题。

对话历史：
{self.short_memory.get_context()}

请直接给出总结回答，不要调用工具。
"""
                    summary_response = self.llm_client.chat_completion(
                        messages=[{"role": "user", "content": summary_prompt}]
                    )
                    summary_content = summary_response.get("content", "") or "已完成任务。"
                    self.short_memory.add_message("assistant", summary_content)
                    return summary_content
                
                # 继续循环
                continue
            
            # 没有工具调用，返回最终响应
            # 清理无效的工具调用标签
            content = self._clean_invalid_tool_calls(content)
            self.short_memory.add_message("assistant", content)
            
            return content
        
        # 达到最大迭代次数
        return "抱歉，任务执行超时，请简化需求后重试。"
    
    def _build_prompt(self) -> str:
        """构建ReAct提示词"""
        tools_info = "\n".join([
            f"- {name}: {tool['description']}"
            for name, tool in self.tools.tools.items()
        ])
        
        context = self.short_memory.get_context()
        
        prompt = f"""
你是一个智能助手，可以使用工具来完成任务。

可用工具：
{tools_info}

工具使用策略：
- 当你需要获取最新信息、新闻、技术文档或不确定的知识时，请使用 web_search 工具
- 当你需要计算数学表达式时，请使用 python_calculator 工具
- 当你需要执行Python代码时，请使用 python_sandbox 工具
- 当你需要生成Python代码而不运行时，请使用 python_sandbox 中的 no_run 参数
- 当你需要保存内容到文件时，请使用 save_file 工具
- 当你需要读取文件内容时，请使用 read_file 工具

对话历史：
{context}

请按照以下格式输出：
1. 思考：分析当前状态和需要做什么，明确说明是否需要使用工具
2. 行动：如果需要调用工具，请使用 <tool_call> 标签；如果不需要工具，请直接回答

格式要求：
- 如果需要调用工具：<tool_call><function name="工具名"><arguments><参数名>值</参数名></arguments></function></tool_call>
- 如果直接回答：直接输出回答内容，不要使用工具调用格式

重要提示：
- 对于需要最新信息的问题（如新闻、技术更新、市场动态等），请先使用 web_search 工具搜索
- 在回答涉及当前事件、技术趋势等问题时，务必使用网络搜索获取最新数据
- 当工具调用成功并获得结果后，如果问题已经解决，请直接总结结果给用户，不要再调用工具
- 如果已经获得了工具执行结果，应该直接用自然语言总结回答，而不是再次调用工具
"""
        
        return prompt
    
    def get_memory_debug(self) -> str:
        """获取记忆调试信息"""
        info = []
        info.append("=== 短期记忆 ===")
        info.append(self.short_memory.get_context())
        return "\n".join(info)
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "total_steps": self.stats["total_steps"],
            "successful_steps": self.stats["successful_steps"],
            "failed_steps": self.stats["failed_steps"],
            "tools_used": list(self.stats["tools_used"])
        }
