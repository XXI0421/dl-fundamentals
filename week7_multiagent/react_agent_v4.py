"""
ReAct Agent V4 - 多工具协作架构
核心特性：
1. 多工具协作流程（搜索→代码执行→文件保存）
2. 工具调用重试机制（指数退避 + 随机抖动）
3. 无效工具调用的自我修正（LLM生成无效JSON时重新生成）
4. 会话隔离：每个会话有独立的长期记忆存储
5. 增强的错误处理和状态追踪

架构设计：
- 工具层：提供搜索、代码执行、文件操作等能力
- 执行引擎：负责工具调用、重试、错误修正
- 记忆系统：短期记忆（会话内）+ 长期记忆（跨会话）
- 反思引擎：自动提取并保存关键信息到长期记忆
"""
import json
import time
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

try:
    from llm_client import KimiClient
    from memory.short_term import ConversationSummaryMemory
    from memory.long_term import get_long_term_memory, LongTermMemory, list_sessions
    from memory.reflection import ReflectionEngine, ReflectionResult
except ImportError:
    # 备用导入路径
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from llm_client import KimiClient
    from memory.short_term import ConversationSummaryMemory
    from memory.long_term import get_long_term_memory, LongTermMemory, list_sessions
    from memory.reflection import ReflectionEngine, ReflectionResult


class MockLLM:
    """
    模拟LLM客户端，用于测试
    支持完整的接口：chat_completion 和 generate
    """
    
    def generate(self, prompt: str, **kwargs) -> str:
        """模拟生成响应（文本模式）"""
        if "YES" in prompt or "NO" in prompt:
            # 反思判断场景
            return "YES"
        if "总结" in prompt or "反思" in prompt:
            # 反思引擎期望的JSON格式
            return '''{
                "summary": "对话总结",
                "facts": [{"text": "用户需要设计一个RPG游戏", "category": "fact", "importance": 0.8}],
                "should_save": true
            }'''
        if "分析" in prompt:
            return "分析完成。"
        return "这是一个模拟响应。"
    
    def chat(self, messages: List[Dict], **kwargs) -> str:
        """模拟聊天响应"""
        return self.generate(str(messages), **kwargs)
    
    def chat_completion(self, messages=None, tools=None, tool_choice="auto", temperature=0.7, **kwargs):
        """
        模拟chat_completion接口（OpenAI格式）
        返回: {"content": str, "finish_reason": str, "tool_calls": list}
        """
        messages = messages or []
        tools = tools or []
        
        # 提取用户消息
        user_content = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_content = msg.get("content", "")
        
        # 判断是否需要调用工具
        need_tool = False
        tool_name = ""
        
        # 根据工具列表决定是否调用工具
        for tool in tools:
            # 处理嵌套格式 {"type": "function", "function": {...}}
            tool_data = tool.get("function", tool)
            tool_desc = tool_data.get("description", "").lower()
            tool_name = tool_data.get("name", "")
            
            if "检索" in tool_desc and ("设计" in user_content or "查询" in user_content):
                need_tool = True
                break
            if "文件" in tool_desc and ("保存" in user_content or "写入" in user_content):
                need_tool = True
                break
            if "prd" in tool_desc and ("prd" in user_content.lower() or "文档" in user_content):
                need_tool = True
                break
        
        # 工具调用模式（代码期望的格式：{"name": "...", "arguments": {...}}）
        if need_tool and tool_choice != "none":
            # 根据工具类型生成正确的参数
            if "检索" in tool_name or "retriever" in tool_name.lower():
                args = {"query": user_content}
            elif "file" in tool_name.lower():
                args = {"content": user_content[:50], "filename": "output.md"}
            elif "prd" in tool_name.lower():
                args = {}
            elif "tech_spec" in tool_name.lower():
                args = {"json_content": '{"architecture": "mvc"}'}
            else:
                args = {"query": user_content[:20]}
            
            return {
                "content": None,
                "finish_reason": "tool_call",
                "tool_calls": [
                    {
                        "id": "call_001",
                        "name": tool_name,
                        "arguments": args
                    }
                ]
            }
        
        # 直接回答模式
        return {
            "content": f"模拟回答: {user_content[:30]}...",
            "finish_reason": "stop",
            "tool_calls": None
        }


@dataclass
class ToolExecutionStep:
    """记录单个工具执行步骤"""
    action: str
    action_input: Dict[str, Any]
    observation: str
    step_num: int
    success: bool = True
    error_message: str = ""
    retry_count: int = 0
    execution_time: float = 0.0


@dataclass
class ToolCallAttempt:
    """记录工具调用尝试详情"""
    tool_name: str
    arguments: str
    attempt: int
    max_retries: int
    success: bool
    result: str
    error: Optional[str] = None
    retry_delay: float = 0.0


class MultiToolAgent:
    """
    多工具协作Agent架构
    
    核心组件：
    1. LLM客户端：负责生成思考和工具调用
    2. 工具注册表：管理可用工具
    3. 短期记忆：会话内上下文（滑动窗口）
    4. 长期记忆：跨会话持久存储（FAISS向量检索）
    5. 反思引擎：自动提取并保存长期记忆
    
    错误恢复机制：
    - 指数退避重试：网络超时等可重试错误自动重试
    - JSON修正：LLM生成无效JSON时自动重新生成
    - 工具验证：调用前验证工具存在性和参数格式
    """
    
    def __init__(
        self,
        llm_client=None,
        tool_registry=None,
        max_iterations: int = 8,
        memory_k: int = 5,
        use_long_term: bool = True,
        use_reflection: bool = True,
        ltm_threshold: float = 0.3,
        ltm_max_facts: int = 3,
        session_id: Optional[str] = None,
        # 重试配置
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        retry_multiplier: float = 2.0,
        # 角色定义参数（MetaGPT风格）
        role: str = "Assistant",
        goal: str = "Help user",
        backstory: str = "You are a helpful AI.",
    ):
        # LLM和工具（支持None作为默认值）
        if llm_client is None:
            from llm_client import KimiClient
            try:
                self.llm = KimiClient()
            except:
                # 如果无法初始化真实LLM，使用模拟客户端
                self.llm = MockLLM()
        else:
            self.llm = llm_client
            
        if tool_registry is None:
            from tools.base import ToolRegistry
            self.tools = ToolRegistry()
        else:
            self.tools = tool_registry
        
        # 生成唯一会话ID（如果未提供）
        if session_id is None:
            import uuid
            self.session_id = f"session_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        else:
            self.session_id = session_id
        # 角色定义（MetaGPT风格）
        self.role = role
        self.goal = goal
        self.backstory = backstory
        
        # 执行配置
        self.max_iterations = max_iterations
        
        # 记忆配置
        self.use_long_term = use_long_term
        self.use_reflection = use_reflection
        self.ltm_threshold = ltm_threshold
        self.ltm_max_facts = ltm_max_facts
        
        # 重试配置
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_multiplier = retry_multiplier
        
        # 短期记忆（会话内）
        self.short_memory = ConversationSummaryMemory(k=memory_k)
        
        # 长期记忆（跨会话）- 延迟加载
        self.long_memory = None
        
        # 反思引擎
        if use_reflection:
            self.reflection = ReflectionEngine(self.llm, session_id=session_id)
        else:
            self.reflection = None
        
        # 执行轨迹
        self.trajectory: List[ToolExecutionStep] = []
        self.current_dialog: List[Dict] = []
        
        # 已保存的事实（用于去重）
        self.saved_facts_cache = set()
        # 反思判断缓存（避免重复判断相似对话）
        self._reflect_decision_cache = {}  # hash -> decision
        
        # 状态追踪
        self._current_state = "idle"  # idle, thinking, executing, completed
        
        # 注册到MessageBus
        if self.bus is not None:
            self.bus.register(self.role)
    
    def _lazy_init_long_memory(self):
        """延迟初始化长期记忆"""
        if self.use_long_term and self.long_memory is None:
            self.long_memory = get_long_term_memory(session_id=self.session_id)
            if self.session_id is None:
                self.session_id = self.long_memory.session_id
    
    def set_long_term_enabled(self, enabled: bool):
        """动态开关长期记忆"""
        self.use_long_term = enabled
        if enabled and self.long_memory is None:
            self._lazy_init_long_memory()
        print(f"[LTM] 长期记忆已{'开启' if enabled else '关闭'}")
    
    def get_session_id(self) -> str:
        """获取当前会话ID"""
        if self.session_id is None and self.long_memory:
            return self.long_memory.session_id
        return self.session_id or "未初始化"
    
    def switch_session(self, session_id: str):
        """切换到指定会话"""
        self.session_id = session_id
        self.long_memory = None
        self.short_memory.clear()
        self.current_dialog = []
        self.trajectory = []
        self.saved_facts_cache = set()
        if self.use_reflection:
            self.reflection = ReflectionEngine(self.llm, session_id=session_id)
        if self.use_long_term:
            self._lazy_init_long_memory()
        print(f"[会话] 已切换到会话: {session_id}")
    
    def new_session(self):
        """创建新会话（重置所有记忆）"""
        self.session_id = None
        self.long_memory = None
        self.short_memory.clear()
        self.current_dialog = []
        self.trajectory = []
        self.saved_facts_cache = set()
        if self.use_reflection:
            self.reflection = ReflectionEngine(self.llm, session_id=None)
        if self.use_long_term:
            self._lazy_init_long_memory()
        print(f"[会话] 已创建新会话: {self.session_id}")
    
    def list_available_sessions(self) -> List[str]:
        """列出所有可用会话"""
        return list_sessions()
    
    # ==================== 重试机制 ====================
    
    def _exponential_backoff(self, attempt: int) -> float:
        """
        计算指数退避延迟
        
        公式: delay = base_delay * (multiplier ^ attempt) * jitter
        
        Args:
            attempt: 当前尝试次数（从0开始）
        
        Returns:
            延迟时间（秒）
        """
        delay = self.retry_base_delay * (self.retry_multiplier ** attempt)
        jitter = random.uniform(0.5, 1.5)  # 添加随机抖动避免重试风暴
        return delay * jitter
    
    def _is_retryable_error(self, result: str) -> bool:
        """
        判断错误是否可以重试
        
        可重试错误类型：
        - 网络超时
        - 服务器错误（5xx）
        - 临时不可用
        
        Args:
            result: 工具执行结果或错误消息
        
        Returns:
            是否可重试
        """
        retryable_patterns = [
            "超时", "网络", "连接", "暂时不可用", "server error",
            "500", "502", "503", "504", "timeout", "connection",
            "request failed", "service unavailable"
        ]
        result_lower = result.lower()
        return any(pattern.lower() in result_lower for pattern in retryable_patterns)
    
    def _retry_tool_call(self, tool_call: Dict) -> Tuple[str, bool, int]:
        """
        使用指数退避重试工具调用
        
        Args:
            tool_call: 工具调用字典，包含 name 和 arguments
        
        Returns:
            (result, success, retry_count)
        """
        tool_name = tool_call["name"]
        arguments = tool_call["arguments"]
        retry_count = 0
        
        for attempt in range(self.max_retries + 1):
            try:
                start_time = time.time()
                result = self._execute_tool(tool_call)
                execution_time = time.time() - start_time
                
                if "错误" in result or "失败" in result:
                    if attempt < self.max_retries and self._is_retryable_error(result):
                        delay = self._exponential_backoff(attempt)
                        retry_count += 1
                        print(f"⚠️ [{tool_name}] 调用失败（第{attempt+1}次），{delay:.2f}秒后重试...")
                        time.sleep(delay)
                        continue
                    else:
                        print(f"❌ [{tool_name}] 调用最终失败: {result[:50]}")
                        return result, False, retry_count
                
                print(f"✅ [{tool_name}] 调用成功（第{attempt+1}次，耗时{execution_time:.2f}s）")
                return result, True, retry_count
            
            except Exception as e:
                error_msg = str(e)
                if attempt < self.max_retries:
                    delay = self._exponential_backoff(attempt)
                    retry_count += 1
                    print(f"⚠️ [{tool_name}] 异常（第{attempt+1}次）: {error_msg[:50]}，{delay:.2f}秒后重试...")
                    time.sleep(delay)
                    continue
                else:
                    print(f"❌ [{tool_name}] 最终异常: {error_msg[:50]}")
                    return f"执行错误: {error_msg}", False, retry_count
        
        return "达到最大重试次数", False, retry_count
    
    # ==================== 无效JSON修正机制 ====================
    
    def _validate_tool_call(self, tool_call: Dict) -> Tuple[bool, str]:
        """
        验证工具调用格式是否正确
        
        验证内容：
        1. 检查必要字段（name, arguments）
        2. 检查工具是否存在于注册表
        3. 检查参数JSON格式是否有效
        4. 验证必需参数是否存在
        
        Args:
            tool_call: 工具调用字典
        
        Returns:
            (is_valid, error_message)
        """
        # 检查必要字段
        required_fields = ["name", "arguments"]
        for field in required_fields:
            if field not in tool_call:
                return False, f"缺少必要字段: {field}"
        
        # 检查工具是否存在
        tool_name = tool_call["name"]
        if tool_name not in self.tools._tools:
            return False, f"工具不存在: {tool_name}"
        
        # 解析参数
        arguments = tool_call["arguments"]
        args_dict = {}
        if isinstance(arguments, str):
            try:
                args_dict = json.loads(arguments)
            except json.JSONDecodeError as e:
                return False, f"无效的JSON参数: {str(e)}"
        elif isinstance(arguments, dict):
            args_dict = arguments
        else:
            return False, f"参数格式错误，应为字典或JSON字符串"
        
        # 验证必需参数
        tool_obj = self.tools._tools[tool_name]
        if tool_obj.schema:
            # schema可能是嵌套格式 {"type": "function", "function": {...}} 或直接包含parameters
            schema_data = tool_obj.schema.get("function", tool_obj.schema)
            params = schema_data.get("parameters", {})
            required_params = params.get("required", [])
            for param in required_params:
                if param not in args_dict:
                    return False, f"缺少必需参数: {param}"
        
        return True, "OK"
    
    def _repair_tool_call(self, tool_call: Dict, error_message: str) -> Optional[Dict]:
        """
        尝试修复无效的工具调用（通过LLM重新生成）
        
        当LLM生成无效的工具调用时：
        1. 将错误信息和原始调用发送给LLM
        2. 让LLM分析错误并生成正确的调用
        3. 返回修复后的工具调用
        
        Args:
            tool_call: 原始无效的工具调用
            error_message: 错误信息
        
        Returns:
            修复后的工具调用，或None（无法修复）
        """
        print(f"🔧 尝试修复无效工具调用: {error_message}")
        
        # 获取工具schema用于提示
        tool_schemas = self.tools.get_schemas()
        
        repair_prompt = f"""
你之前生成的工具调用有错误：

错误信息：{error_message}

原始工具调用：
{json.dumps(tool_call, ensure_ascii=False)}

请根据以下工具定义，修正工具调用参数：

工具列表：
{json.dumps(tool_schemas, ensure_ascii=False, indent=2)}

请只返回修复后的工具调用JSON，格式如下：
{{
  "name": "工具名称",
  "arguments": "{{参数JSON字符串}}"
}}
"""
        
        try:
            response = self.llm.chat_completion(
                messages=[
                    {"role": "system", "content": "你是一个工具调用修复助手。请分析错误并生成正确的工具调用。"},
                    {"role": "user", "content": repair_prompt}
                ],
                temperature=0.1  # 低温度保证确定性
            )
            
            content = response.get("content", "")
            if not content:
                print("❌ 修复失败：LLM返回空内容")
                return None
            
            # 尝试解析修复后的JSON
            try:
                fixed_call = json.loads(content.strip())
                print(f"✅ 修复成功：{fixed_call['name']}")
                return fixed_call
            except json.JSONDecodeError:
                print(f"❌ 修复失败：无法解析LLM返回的JSON: {content[:100]}")
                return None
        
        except Exception as e:
            print(f"❌ 修复过程异常: {str(e)}")
            return None
    
    # ==================== 主执行循环 ====================
    
    def run(self, query: str) -> str:
        """
        主执行循环
        
        流程：
        1. 解析用户命令（会话操作、记忆开关）
        2. 检索长期记忆
        3. 构建上下文消息
        4. 迭代调用LLM和工具
        5. 处理工具调用（验证→修复→重试→执行）
        6. 执行反思并保存长期记忆
        
        Args:
            query: 用户查询
        
        Returns:
            最终回答
        """
        # 检查用户是否想要开关长期记忆
        if self._check_memory_switch(query):
            return "已按您的要求调整长期记忆设置。"
        
        # 检查用户是否想要切换会话
        session_cmd = self._check_session_command(query)
        if session_cmd:
            return session_cmd
        
        # 记录用户输入
        self.short_memory.add_user(query)
        self.current_dialog.append({"role": "user", "content": query})
        
        # 检索长期记忆（带阈值过滤）
        long_term_facts = self._retrieve_long_term(query)
        
        # 构建消息上下文
        messages = self._build_messages(long_term_facts)
        
        final_answer = ""
        tools_used = []
        last_tool_calls = None
        
        for iteration in range(self.max_iterations):
            print(f"\n--- 第 {iteration+1} 轮 ---")
            
            # 调用LLM生成响应
            response = self.llm.chat_completion(
                messages=messages,
                tools=self.tools.get_schemas(),
                tool_choice="auto",
                temperature=0.2
            )
            
            # 处理API错误
            if "error" in response:
                final_answer = f"API错误: {response['error']}"
                break
            
            # 直接回答（没有工具调用）
            if response.get("finish_reason") == "stop" or not response.get("tool_calls"):
                final_answer = response.get("content", "")
                self.short_memory.add_assistant(content=final_answer)
                self.current_dialog.append({"role": "assistant", "content": final_answer})
                print(f"✅ 直接回答: {final_answer[:100]}...")
                break
            
            # 处理工具调用
            tool_calls = response["tool_calls"]
            tools_used = [tc["name"] for tc in tool_calls]
            last_tool_calls = tool_calls
            print(f"🔧 调用工具: {tools_used}")
            
            # 记录assistant消息
            assistant_msg = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"]
                        }
                    } for tc in tool_calls
                ]
            }
            messages.append(assistant_msg)
            self.short_memory.add_assistant(tool_calls=tool_calls)
            
            # 逐个执行工具调用
            for tc in tool_calls:
                start_time = time.time()
                
                # 1. 验证工具调用
                is_valid, validation_msg = self._validate_tool_call(tc)
                
                if not is_valid:
                    print(f"⚠️ 工具调用验证失败: {validation_msg}")
                    
                    # 2. 尝试修复
                    fixed_call = self._repair_tool_call(tc, validation_msg)
                    
                    if fixed_call:
                        tc = fixed_call
                    else:
                        # 无法修复，记录错误并继续
                        error_result = f"工具调用失败: {validation_msg}"
                        self._record_tool_result(messages, tc, error_result, False, iteration + 1)
                        continue
                
                # 3. 执行工具调用（带重试）
                result, success, retry_count = self._retry_tool_call(tc)
                execution_time = time.time() - start_time
                
                # 记录结果
                self._record_tool_result(
                    messages, tc, result, success, iteration + 1, 
                    retry_count, execution_time
                )
        
        else:
            # 达到最大迭代次数，进行总结
            if last_tool_calls:
                print("\n--- 最后总结 ---")
                response = self.llm.chat_completion(
                    messages=messages,
                    tools=self.tools.get_schemas(),
                    tool_choice="none",
                    temperature=0.2
                )
                if response.get("content"):
                    final_answer = response["content"]
                    print(f"✅ 总结回答: {final_answer[:100]}...")
                else:
                    final_answer = "达到最大迭代次数，任务未完成"
            else:
                final_answer = "达到最大迭代次数，任务未完成"
            
            self.short_memory.add_assistant(content=final_answer)
            self.current_dialog.append({"role": "assistant", "content": final_answer})
        
        # 记录会话
        self.short_memory.add_conversation({
            "user": query,
            "agent_response": final_answer,
            "tools_used": tools_used
        })
        
        # 执行反思（保存到长期记忆）
        if self.use_reflection and self._should_reflect(query, final_answer):
            self._do_reflection(final_answer)
        
        # 设置完成状态
        self._current_state = "completed"
        
        return final_answer
    
    def _record_tool_result(self, messages: List[Dict], tc: Dict, result: str, 
                           success: bool, step_num: int, retry_count: int = 0, 
                           execution_time: float = 0.0):
        """记录工具执行结果"""
        # 添加到消息列表
        tool_msg = {
            "role": "tool",
            "tool_call_id": tc["id"],
            "name": tc["name"],
            "content": str(result)
        }
        messages.append(tool_msg)
        
        # 添加到短期记忆
        self.short_memory.add_tool_result(tc["id"], tc["name"], str(result))
        
        # 解析参数
        try:
            action_input = json.loads(tc["arguments"]) if isinstance(tc["arguments"], str) else tc["arguments"]
        except:
            action_input = {}
        
        # 添加到执行轨迹
        self.trajectory.append(ToolExecutionStep(
            action=tc["name"],
            action_input=action_input,
            observation=str(result),
            step_num=step_num,
            success=success,
            error_message="" if success else str(result),
            retry_count=retry_count,
            execution_time=execution_time
        ))
        
        print(f"👁 [{tc['name']}]: {str(result)[:60]}...")
    
    # ==================== 辅助方法 ====================
    
    def _check_memory_switch(self, query: str) -> bool:
        """检查用户是否想要开关长期记忆"""
        query_lower = query.lower()
        if "关闭长期记忆" in query_lower or "关闭记忆" in query_lower:
            self.set_long_term_enabled(False)
            return True
        elif "开启长期记忆" in query_lower or "开启记忆" in query_lower:
            self.set_long_term_enabled(True)
            return True
        return False
    
    def _check_session_command(self, query: str) -> Optional[str]:
        """检查用户是否想要进行会话操作"""
        query_lower = query.lower()
        
        if "新会话" in query_lower or "新建会话" in query_lower or "重置会话" in query_lower:
            self.new_session()
            return f"已创建新会话，会话ID: {self.session_id}"
        
        if "切换会话" in query_lower:
            import re
            match = re.search(r'切换会话\s*(\S+)', query)
            if match:
                target_session = match.group(1)
                sessions = self.list_available_sessions()
                if target_session in sessions:
                    self.switch_session(target_session)
                    return f"已切换到会话: {target_session}"
                else:
                    return f"未找到会话: {target_session}。可用会话: {', '.join(sessions)}"
            else:
                sessions = self.list_available_sessions()
                return f"请指定要切换的会话ID。可用会话: {', '.join(sessions)}"
        
        if "查看会话" in query_lower or "会话列表" in query_lower:
            sessions = self.list_available_sessions()
            if sessions:
                return f"当前会话ID: {self.session_id}\n可用会话列表:\n" + "\n".join(sessions)
            else:
                return "暂无保存的会话"
        
        # 只有当用户明确询问会话信息时才响应，避免误匹配
        if "当前会话" in query_lower:
            # 检查是否只是询问会话信息（单独提问或作为主要意图）
            # 排除包含其他内容的情况，如"保存到当前会话"
            # 只有当"当前会话"是主要意图时才响应
            if query_lower == "当前会话" or \
               query_lower.startswith("当前会话") and len(query_lower) < 15 or \
               "当前会话是什么" in query_lower or \
               "当前会话id" in query_lower or \
               "当前会话id是" in query_lower:
                return f"当前会话ID: {self.session_id}"
        
        return None
    
    def _should_reflect(self, query: str, answer: str) -> bool:
        """
        判断是否需要进行反思（使用LLM自主决定）
        
        通过调用LLM分析对话内容，判断是否包含值得长期保存的信息。
        相比硬编码关键词，这种方式更灵活、更准确。
        
        性能优化：
        1. 最小长度阈值：总字符数<20直接返回False
        2. 关键词优先：匹配到关键词直接返回True，不调用LLM
        3. 缓存机制：相同对话内容直接返回缓存结果
        """
        # 最小长度检查：总字符数<20直接跳过（避免无意义的短对话）
        total_length = len(query) + len(answer)
        if total_length < 20:
            return False
        
        # 快速检查：排除简单问候和命令
        simple_patterns = ["你好", "hello", "hi", "谢谢", "再见", "谢谢", "q", "n", "ls", "help", "info"]
        query_lower = query.lower()
        for pattern in simple_patterns:
            if pattern in query_lower:
                return False
        
        # 优先检查明确的关键词（提高准确性，避免调用LLM）
        key_patterns = ["姓名", "名字", "叫", "出生", "生日", "年龄", "职业", "工作", 
                       "喜欢", "偏好", "擅长", "邮箱", "电话", "改名", "记忆"]
        for pattern in key_patterns:
            if pattern in query or pattern in answer:
                print(f"[反思判断] 匹配关键词 '{pattern}'，触发反思")
                return True
        
        # 缓存检查：避免重复判断相似对话
        cache_key = hash((query[:100], answer[:100]))
        if cache_key in self._reflect_decision_cache:
            cached_result = self._reflect_decision_cache[cache_key]
            print(f"[反思判断] 使用缓存结果: {cached_result}")
            return cached_result
        
        # 仅在关键词匹配失败且通过所有快速检查后，才使用LLM判断
        prompt = f"""分析以下对话是否包含值得长期保存的信息：

用户问题：{query}
Agent回答：{answer}

请判断这段对话是否包含以下类型的信息：
- 身份信息（姓名、生日、年龄、性别等）
- 职业信息（工作、职位、公司等）
- 偏好爱好（喜欢的事物、习惯等）
- 重要事实（联系方式、地址、关键日期等）
- 长期目标或计划

如果包含以上任何类型的信息，请回答 YES；否则回答 NO。

只需要回答 YES 或 NO，不要添加其他内容。
"""
        
        try:
            response = self.llm.generate(prompt)
            result = "YES" in response.upper()
            print(f"[反思判断] LLM判断结果: {result}")
            
            # 缓存结果（限制缓存大小，避免内存溢出）
            if len(self._reflect_decision_cache) < 100:
                self._reflect_decision_cache[cache_key] = result
            
            return result
        except Exception as e:
            print(f"[反思判断] LLM调用失败，回退到False: {e}")
            return False
    
    def _retrieve_long_term(self, query: str) -> List[Dict[str, Any]]:
        """检索相关长期记忆（基于向量语义匹配）"""
        if not self.use_long_term:
            return []
        
        self._lazy_init_long_memory()
        if self.long_memory is None:
            return []
        
        results = self.long_memory.retrieve(query, top_k=self.ltm_max_facts * 3)
        filtered = [r for r in results if r.get('score', 0) >= self.ltm_threshold]
        final_facts = self._detect_conflicts(filtered)
        
        facts = []
        for r in final_facts[:self.ltm_max_facts]:
            text = r['text'][:100]
            facts.append({
                "text": text, 
                "category": r.get('category', 'fact'), 
                "score": r['score']
            })
        
        if facts:
            fact_str = ", ".join([f"{f['text'][:50]} [{f['category']}]" for f in facts])
            print(f"[LTM] 检索到 {len(facts)} 条相关记忆: {fact_str}")
        
        return facts
    
    def _detect_conflicts(self, facts: List[Dict]) -> List[Dict]:
        """检测并移除冲突的事实"""
        if len(facts) < 2:
            return facts
        
        result = []
        seen_patterns = {}
        score_threshold = 0.1
        
        for fact in facts:
            text = fact['text']
            category = fact.get('category', 'fact')
            score = fact.get('score', 0)
            timestamp = fact.get('timestamp', '')
            
            key = None
            if '出生' in text or '生日' in text:
                import re
                name_match = re.search(r'([\u4e00-\u9fa5]{2,4})\s*', text)
                year_match = re.search(r'(\d{4})年', text)
                if name_match and year_match:
                    key = f"birth_{name_match.group(1)}"
            elif '年龄' in text:
                name_match = re.search(r'([\u4e00-\u9fa5]{2,4})\s*', text)
                if name_match:
                    key = f"age_{name_match.group(1)}"
            
            if key:
                if key in seen_patterns:
                    existing = seen_patterns[key]
                    existing_score = existing.get('score', 0)
                    existing_timestamp = existing.get('timestamp', '')
                    
                    should_replace = False
                    if abs(score - existing_score) > score_threshold:
                        should_replace = score > existing_score
                    else:
                        should_replace = timestamp > existing_timestamp
                    
                    if should_replace:
                        result = [f for f in result if f['text'] != existing['text']]
                        result.append(fact)
                        seen_patterns[key] = fact
                else:
                    seen_patterns[key] = fact
                    result.append(fact)
            else:
                result.append(fact)
        
        return result
    
    def _do_reflection(self, final_answer: str):
        """执行反思，提取并保存长期记忆"""
        if not self.use_reflection:
            return
        
        short_facts = self.short_memory.get_summary().split("；")
        
        result = self.reflection.reflect(
            conversation_history=self.current_dialog,
            current_facts=short_facts
        )
        
        if result.should_save and result.facts:
            for fact in result.facts:
                fact_key = fact['text'][:50]
                if fact_key not in self.saved_facts_cache:
                    self.saved_facts_cache.add(fact_key)
            
            print(f"\n[反思] 保存 {len(result.facts)} 条新的长期记忆")
            for fact in result.facts:
                print(f"  - {fact['text'][:50]} [{fact['category']}]")
    
    def _build_messages(self, long_term_facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """构建上下文消息"""
        short_facts = self.short_memory.get_summary()
        
        # 构建长期记忆部分（结构化格式，便于LLM理解）
        if long_term_facts:
            # 按类别分组显示记忆
            categories = {}
            for fact in long_term_facts:
                cat = fact.get('category', 'fact')
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(fact)
            
            long_term_parts = []
            for cat_name, cat_facts in categories.items():
                cat_label = {
                    'identity': '身份信息',
                    'fact': '客观事实',
                    'preference': '偏好爱好',
                    'relationship': '人际关系',
                    'general': '其他信息'
                }.get(cat_name, cat_name)
                
                fact_lines = [f"  - {f['text']} (相似度: {f['score']:.2f})" for f in cat_facts]
                if fact_lines:
                    long_term_parts.append(f"【{cat_label}】")
                    long_term_parts.extend(fact_lines)
            
            long_term_content = "\n".join(long_term_parts)
        else:
            long_term_content = "暂无长期记忆"
        
        # 获取可用工具列表（带描述）
        tool_descriptions = []
        for tool_name in self.tools.list_tools():
            tool_obj = self.tools.get_tool(tool_name)
            if tool_obj:
                desc = tool_obj.description
                tool_descriptions.append(f"- {tool_name}: {desc}")
        tools_content = "\n".join(tool_descriptions)
        
        # System Prompt（优化版）
        system_content = f"""你是一个具有长期记忆能力的智能助手。

=== 核心指令 ===
1. 【必须优先使用记忆】在回答任何问题前，先检查【用户记忆信息】中是否有相关内容
2. 【记忆匹配规则】如果用户问题与记忆中的信息相关（如询问身份、偏好、历史信息），必须使用记忆中的信息回答，不得编造
3. 【记忆缺失处理】如果记忆中没有相关信息，明确告知用户"我不记得了"或"我需要了解更多"，不要猜测

=== 可用工具 ===
{tools_content}

=== 用户记忆信息（重要！）===
{long_term_content}

=== 会话历史摘要 ===
{short_facts if short_facts else '无会话历史'}

=== 工具使用规则 ===
- 需要实时数据或搜索：使用 search_duckduckgo
- 需要数学计算或数据分析：使用 python_sandbox
- 需要保存结果：使用 save_file
- 需要读取文件：使用 read_file

=== 关键提醒 ===
- 如果用户问"我是谁？"、"我的名字是什么？"、"我多大了？"等问题，必须从【用户记忆信息】中查找答案
- 如果记忆中有关于用户的信息（如姓名、年龄、职业等），回答时要自然地引用这些信息
- 工具调用参数必须是有效的JSON格式
- 保持回答自然友好，像日常聊天一样

请开始回答用户的问题。"""

        messages = [{"role": "system", "content": system_content}]
        
        # 添加最近历史消息
        recent_history = self.short_memory.get_messages()[-4:]
        if recent_history:
            messages.extend(recent_history)
            print(f"[上下文] {len(recent_history)} 条消息")
        
        return messages
    
    def _execute_tool(self, tool_call: Dict) -> str:
        """执行单个工具调用"""
        name = tool_call["name"]
        try:
            # 解析参数
            args = tool_call["arguments"]
            if isinstance(args, str):
                args = json.loads(args)
            elif not isinstance(args, dict):
                return f"参数格式错误: 期望字典或JSON字符串，实际类型: {type(args).__name__}"
            
            # 执行工具
            if name in self.tools._tools:
                tool_obj = self.tools._tools[name]
                if hasattr(tool_obj, 'func') and callable(tool_obj.func):
                    result = tool_obj.func(**args)
                elif callable(tool_obj):
                    result = tool_obj(**args)
                else:
                    return f"错误: 工具 {name} 不可调用"
                return str(result)[:1000]
            return f"错误: 未找到工具 {name}"
        except json.JSONDecodeError as e:
            return f"JSON解析错误: {str(e)}"
        except TypeError as e:
            return f"参数类型错误: {str(e)} - 参数: {args}"
        except Exception as e:
            import traceback
            return f"执行错误: {str(e)}\n{traceback.format_exc()}"
    
    # ==================== 调试和统计 ====================
    
    def get_memory_debug(self) -> str:
        """获取记忆调试信息"""
        short = self.short_memory.get_full_summary()
        session_info = f"\n【会话ID】{self.session_id}"
        if self.use_long_term and self.long_memory:
            profile = self.long_memory.get_user_profile()
            long_summary = f"\n【长期记忆画像】\n身份: {str(profile['identity'])[:50]}...\n偏好: {str(profile['preference'])[:50]}...\n事实: {str(profile['fact'])[:50]}..."
        else:
            long_summary = "\n【长期记忆】已关闭"
        return session_info + short + long_summary
    
    def clear_short_memory(self):
        """清空短期记忆"""
        self.short_memory.clear()
        self.current_dialog = []
        self.trajectory = []
    
    def get_status(self) -> Dict[str, Any]:
        """获取当前Agent状态"""
        return {
            "state": self._current_state if hasattr(self, '_current_state') else "completed",
            "iteration": len(self.trajectory),
            "error_count": sum(1 for step in self.trajectory if not step.success),
            "tool_calls": len([step for step in self.trajectory if step.action != "总结"]),
            "memory_enabled": self.use_long_term,
            "reflection_enabled": self.use_reflection
        }
    
    def get_tool_call_stats(self) -> Dict[str, Any]:
        """获取工具调用统计信息"""
        stats = {
            "total_steps": len(self.trajectory),
            "successful_steps": sum(1 for step in self.trajectory if step.success),
            "failed_steps": sum(1 for step in self.trajectory if not step.success),
            "total_retries": sum(step.retry_count for step in self.trajectory),
            "total_execution_time": sum(step.execution_time for step in self.trajectory),
            "tools_used": list(set(step.action for step in self.trajectory)),
            "trajectory": [
                {
                    "step": step.step_num,
                    "tool": step.action,
                    "success": step.success,
                    "retry_count": step.retry_count,
                    "execution_time": step.execution_time,
                    "observation": step.observation[:50]
                }
                for step in self.trajectory
            ]
        }
        return stats


# 为了保持与Week 6架构的兼容性，添加ReActAgentV4别名
ReActAgentV4 = MultiToolAgent