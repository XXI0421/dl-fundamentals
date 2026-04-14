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
        llm_client: KimiClient,
        tool_registry,
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
    ):
        # LLM和工具
        self.llm = llm_client
        self.tools = tool_registry
        
        # 执行配置
        self.max_iterations = max_iterations
        
        # 记忆配置
        self.use_long_term = use_long_term
        self.use_reflection = use_reflection
        self.ltm_threshold = ltm_threshold
        self.ltm_max_facts = ltm_max_facts
        self.session_id = session_id
        
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
            self.reflection = ReflectionEngine(llm_client, session_id=session_id)
        
        # 执行轨迹
        self.trajectory: List[ToolExecutionStep] = []
        self.current_dialog: List[Dict] = []
        
        # 已保存的事实（用于去重）
        self.saved_facts_cache = set()
    
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
        
        # 尝试解析参数JSON
        arguments = tool_call["arguments"]
        if isinstance(arguments, str):
            try:
                json.loads(arguments)
            except json.JSONDecodeError as e:
                return False, f"无效的JSON参数: {str(e)}"
        
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
        
        if "当前会话" in query_lower:
            return f"当前会话ID: {self.session_id}"
        
        return None
    
    def _should_reflect(self, query: str, answer: str) -> bool:
        """判断是否需要进行反思"""
        key_patterns = ["出生", "生日", "年龄", "姓名", "职业", "工作", 
                       "喜欢", "偏好", "擅长", "邮箱", "电话"]
        
        for pattern in key_patterns:
            if pattern in query or pattern in answer:
                return True
        
        if "计算" in query or "统计" in query or "分析" in query:
            return True
        
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
        else:
            print(f"\n[反思] 无需保存")
    
    def _build_messages(self, long_term_facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """构建上下文消息"""
        short_facts = self.short_memory.get_summary()
        
        # 构建长期记忆部分
        if long_term_facts:
            high_confidence = [f for f in long_term_facts if f['score'] >= 0.4]
            if high_confidence:
                long_term_content = "\n".join([f"- {f['text']}" for f in high_confidence])
            else:
                long_term_content = "\n".join([f"- {f['text']}" for f in long_term_facts[:2]])
        else:
            long_term_content = "无"
        
        # 获取可用工具列表
        available_tools = ", ".join(self.tools.list_tools())
        
        # System Prompt
        system_content = f"""你是一个智能助手，擅长利用工具和记忆完成复杂任务。

【可用工具】
{available_tools}

【用户记忆信息】
{long_term_content}

【会话历史】
{short_facts if short_facts else '无'}

## 重要指令：
1. 如果需要实时信息或数据，使用 search_duckduckgo 或 python_sandbox 工具
2. 如果需要计算或数据分析，使用 python_sandbox 工具执行Python代码
3. 如果需要保存结果，使用 save_file 工具
4. 仔细阅读【用户记忆信息】，这是关于用户的重要个人信息
5. 当用户问"我是谁？"、"我的名字？"等问题时，必须从【用户记忆信息】中查找答案
6. 如果记忆中有相关信息，必须使用记忆中的信息回答
7. 工具调用失败时会自动重试，请确保参数格式正确（JSON格式）
8. 如果工具调用返回错误，分析错误原因并尝试修复后重新调用
9. 回答要自然、友好，像聊天一样"""

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
            args = json.loads(tool_call["arguments"]) if isinstance(tool_call["arguments"], str) else tool_call["arguments"]
            if name in self.tools._tools:
                tool_obj = self.tools._tools[name]
                result = tool_obj.func(**args) if hasattr(tool_obj, 'func') else tool_obj(**args)
                return str(result)[:1000]
            return f"错误: 未找到工具 {name}"
        except json.JSONDecodeError as e:
            return f"JSON解析错误: {str(e)}"
        except Exception as e:
            return f"执行错误: {str(e)}"
    
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