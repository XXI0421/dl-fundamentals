"""
ReAct Agent V3 - 集成短期记忆、长期记忆和反思引擎
优化版本：减少token消耗，提高准确性
支持会话隔离：每个会话有独立的长期记忆存储
"""
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from llm_client import KimiClient
from memory.short_term import ConversationSummaryMemory
from memory.long_term import get_long_term_memory, LongTermMemory, list_sessions
from memory.reflection import ReflectionEngine, ReflectionResult

@dataclass
class Step:
    action: str
    action_input: Dict[str, Any]
    observation: str
    step_num: int

class ReActAgentV3:
    """
    集成三种记忆的 ReAct Agent：
    - 短期记忆：会话内上下文（滑动窗口）
    - 长期记忆：跨会话持久存储（FAISS向量检索），支持会话隔离
    - 反思引擎：自动提取并保存长期记忆
    
    优化特性：
    1. 长期记忆可动态开关
    2. 可信度阈值过滤低质量匹配
    3. 事实冲突检测避免错误信息
    4. 限制记忆数量和长度减少token消耗
    5. 会话隔离：新会话重置长期记忆，旧会话可恢复
    """
    
    def __init__(
        self,
        llm_client: KimiClient,
        tool_registry,
        max_iterations: int = 5,
        memory_k: int = 3,
        use_long_term: bool = True,
        use_reflection: bool = True,
        ltm_threshold: float = 0.3,  # 可信度阈值
        ltm_max_facts: int = 3,       # 最大返回记忆数量
        session_id: Optional[str] = None  # 会话ID，为None时创建新会话
    ):
        self.llm = llm_client
        self.tools = tool_registry
        self.max_iterations = max_iterations
        self.use_long_term = use_long_term
        self.use_reflection = use_reflection
        self.ltm_threshold = ltm_threshold
        self.ltm_max_facts = ltm_max_facts
        self.session_id = session_id
        
        # 短期记忆（会话内）
        self.short_memory = ConversationSummaryMemory(k=memory_k)
        
        # 长期记忆（跨会话）- 延迟加载
        self.long_memory = None
        
        # 反思引擎
        if use_reflection:
            self.reflection = ReflectionEngine(llm_client, session_id=session_id)
        
        self.trajectory: List[Step] = []
        self.current_dialog: List[Dict] = []  # 用于反思的完整对话
        
        # 已保存的事实（用于去重）
        self.saved_facts_cache = set()
    
    def _lazy_init_long_memory(self):
        """延迟初始化长期记忆"""
        if self.use_long_term and self.long_memory is None:
            self.long_memory = get_long_term_memory(session_id=self.session_id)
            # 更新会话ID（如果是新创建的）
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
        # 重新初始化反思引擎以绑定到新会话
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
        # 重新初始化反思引擎（不指定session_id，会自动生成新的）
        if self.use_reflection:
            self.reflection = ReflectionEngine(self.llm, session_id=None)
        if self.use_long_term:
            self._lazy_init_long_memory()
        print(f"[会话] 已创建新会话: {self.session_id}")
    
    def list_available_sessions(self) -> List[str]:
        """列出所有可用会话"""
        return list_sessions()
    
    def run(self, query: str) -> str:
        """主循环"""
        # 检查用户是否想要开关长期记忆
        if self._check_memory_switch(query):
            return "已按您的要求调整长期记忆设置。"
        
        # 检查用户是否想要切换会话
        session_cmd = self._check_session_command(query)
        if session_cmd:
            return session_cmd
        
        self.short_memory.add_user(query)
        self.current_dialog.append({"role": "user", "content": query})
        
        # 检索长期记忆（带阈值过滤）
        long_term_facts = self._retrieve_long_term(query)
        
        messages = self._build_messages(long_term_facts)
        
        final_answer = ""
        tools_used = []
        last_tool_calls = None
        
        for i in range(self.max_iterations):
            print(f"\n--- 第 {i+1} 轮 ---")
            
            response = self.llm.chat_completion(
                messages=messages,
                tools=self.tools.get_schemas(),
                tool_choice="auto",
                temperature=0.2
            )
            
            if "error" in response:
                final_answer = f"API错误: {response['error']}"
                break
            
            if response.get("finish_reason") == "stop" or not response.get("tool_calls"):
                final_answer = response.get("content", "")
                self.short_memory.add_assistant(content=final_answer)
                self.current_dialog.append({"role": "assistant", "content": final_answer})
                print(f"✅ 直接回答: {final_answer[:100]}...")
                break
            
            tool_calls = response["tool_calls"]
            tools_used = [tc["name"] for tc in tool_calls]
            last_tool_calls = tool_calls
            print(f"🔧 调用: {tools_used}")
            
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
            
            for tc in tool_calls:
                result = self._execute_tool(tc)
                
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": tc["name"],
                    "content": str(result)
                }
                messages.append(tool_msg)
                self.short_memory.add_tool_result(tc["id"], tc["name"], str(result))
                
                print(f"👁 {tc['name']}: {str(result)[:60]}...")
                
                self.trajectory.append(Step(
                    action=tc["name"],
                    action_input=json.loads(tc["arguments"]),
                    observation=str(result),
                    step_num=i+1
                ))
        
        else:
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
        
        self.short_memory.add_conversation({
            "user": query,
            "agent_response": final_answer,
            "tools_used": tools_used
        })
        
        # 执行反思（保存到长期记忆）- 优化：只在有新信息时保存
        if self.use_reflection and self._should_reflect(query, final_answer):
            self._do_reflection(final_answer)
        
        return final_answer
    
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
            # 尝试提取会话ID
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
        """判断是否需要进行反思（优化：减少不必要的反思）"""
        # 只在包含关键信息类型时进行反思
        key_patterns = ["出生", "生日", "年龄", "姓名", "职业", "工作", 
                       "喜欢", "偏好", "擅长", "邮箱", "电话"]
        
        for pattern in key_patterns:
            if pattern in query or pattern in answer:
                return True
        
        # 计算相关的问题也需要保存
        if "计算" in query or "统计" in query or "分析" in query:
            return True
        
        return False
    
    def _retrieve_long_term(self, query: str) -> List[Dict[str, Any]]:
        """检索相关长期记忆（基于向量语义匹配）
        
        现代RAG系统的工作方式：
        1. 将用户查询转换为向量
        2. 在向量数据库中进行余弦相似度检索
        3. 返回最相关的记忆作为上下文
        4. LLM自动利用这些上下文生成回答
        
        关键词：向量嵌入、余弦相似度、语义检索、上下文注入
        """
        if not self.use_long_term:
            return []
        
        self._lazy_init_long_memory()
        if self.long_memory is None:
            return []
        
        # 核心：向量语义检索（已经在long_memory.retrieve中完成）
        # 向量检索会自动计算查询与记忆的语义相似度
        results = self.long_memory.retrieve(query, top_k=self.ltm_max_facts * 3)
        
        # 应用阈值过滤（只保留相似度足够高的记忆）
        # 这里的分数是相似度（0-1），而非距离
        filtered = [r for r in results if r.get('score', 0) >= self.ltm_threshold]
        
        # 冲突检测：移除矛盾的事实
        final_facts = self._detect_conflicts(filtered)
        
        # 限制数量并格式化
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
        """检测并移除冲突的事实（结合时间因素）"""
        if len(facts) < 2:
            return facts
        
        # 冲突检测：检查是否有相同类型但不同值的事实
        # 例如："张三出生于1995年" 和 "张三出生于1996年"
        result = []
        seen_patterns = {}
        score_threshold = 0.1  # 分数差异阈值
        
        for fact in facts:
            text = fact['text']
            category = fact.get('category', 'fact')
            score = fact.get('score', 0)
            timestamp = fact.get('timestamp', '')
            
            # 提取关键字（姓名、出生、年龄等）
            key = None
            if '出生' in text or '生日' in text:
                # 找到名字和年份
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
                    # 冲突：综合考虑分数和时间因素
                    existing = seen_patterns[key]
                    existing_score = existing.get('score', 0)
                    existing_timestamp = existing.get('timestamp', '')
                    
                    should_replace = False
                    
                    # 如果分数差异超过阈值，选择分数高的
                    if abs(score - existing_score) > score_threshold:
                        should_replace = score > existing_score
                    else:
                        # 分数相近时，选择时间最新的
                        should_replace = timestamp > existing_timestamp
                    
                    if should_replace:
                        # 移除旧的，添加新的
                        result = [f for f in result if f['text'] != existing['text']]
                        result.append(fact)
                        seen_patterns[key] = fact
                else:
                    seen_patterns[key] = fact
                    result.append(fact)
            else:
                result.append(fact)
        
        return result
    
    def _get_relevant_categories(self, query: str) -> List[str]:
        """根据查询内容确定需要检索的记忆类别"""
        query_lower = query.lower()
        
        # 定义查询类型到类别的映射
        category_rules = {
            # 身份相关查询
            'identity': ['名字', '姓名', '叫什么', '多大', '年龄', '出生', '生日', '哪里人'],
            # 偏好相关查询
            'preference': ['喜欢', '爱好', '偏好', '擅长', '讨厌', '想要'],
            # 工作相关查询
            'work': ['工作', '职业', '公司', '职位', '上班'],
            # 数据相关查询
            'data': ['数据', '销售额', '统计', '分析', '计算', '图表', 'csv', 'excel'],
            # 知识相关查询
            'knowledge': ['什么是', '定义', '解释', '原理', '知识', '科普'],
        }
        
        relevant = []
        for category, keywords in category_rules.items():
            for keyword in keywords:
                if keyword in query_lower:
                    relevant.append(category)
                    break
        
        return relevant
    
    def _filter_by_keywords(self, query: str, facts: List[Dict]) -> List[Dict]:
        """语义过滤：现代RAG系统的简化实现
        
        在真正的LLM系统中，这个函数几乎不需要存在，因为：
        1. 向量检索已经完成了精确的语义匹配
        2. LLM会自动根据上下文判断哪些记忆相关
        3. 只有在特殊业务场景下才需要额外过滤
        
        这里保留空实现保持兼容性，实际过滤已在_retrieve_long_term中完成
        """
        return facts
    
    def _do_reflection(self, final_answer: str):
        """执行反思，提取并保存长期记忆（带去重）"""
        if not self.use_reflection:
            return
        
        short_facts = self.short_memory.get_summary().split("；")
        
        result = self.reflection.reflect(
            conversation_history=self.current_dialog,
            current_facts=short_facts
        )
        
        if result.should_save and result.facts:
            # 更新缓存：ReflectionEngine 已经保存了事实
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
        """构建上下文（精简版，减少token消耗）"""
        short_facts = self.short_memory.get_summary()
        
        # 构建长期记忆部分（更精简的格式）
        if long_term_facts:
            # 只保留高可信度的事实
            high_confidence = [f for f in long_term_facts if f['score'] >= 0.4]
            if high_confidence:
                long_term_content = "\n".join([f"- {f['text']}" for f in high_confidence])
            else:
                long_term_content = "\n".join([f"- {f['text']}" for f in long_term_facts[:2]])
        else:
            long_term_content = "无"
        
        # 更明确的system prompt，告诉模型如何使用记忆
        system_content = f"""你是一个智能助手，擅长利用记忆回答用户问题。

【用户记忆信息】
{long_term_content}

【会话历史】
{short_facts if short_facts else '无'}

## 重要指令：
1. 仔细阅读【用户记忆信息】，这是关于用户的重要个人信息
2. 当用户问"我是谁？"、"我的名字？"、"我的生日？"等问题时，必须从【用户记忆信息】中查找答案
3. 如果记忆中有相关信息，必须使用记忆中的信息回答，不能说不知道
4. 如果记忆中没有相关信息，可以说"我还不知道"或询问用户
5. 回答要自然、友好，像聊天一样"""

        messages = [{"role": "system", "content": system_content}]
        
        # 限制历史消息数量（保留足够上下文）
        recent_history = self.short_memory.get_messages()[-4:]  # 保留最近4条消息
        if recent_history:
            messages.extend(recent_history)
            print(f"[上下文] {len(recent_history)} 条消息")
        
        return messages
    
    def _execute_tool(self, tool_call: Dict) -> str:
        name = tool_call["name"]
        try:
            args = json.loads(tool_call["arguments"])
            if name in self.tools._tools:
                tool_obj = self.tools._tools[name]
                result = tool_obj.func(**args) if hasattr(tool_obj, 'func') else tool_obj(**args)
                return str(result)[:1000]
            return f"错误: 未找到工具 {name}"
        except Exception as e:
            return f"执行错误: {str(e)}"
    
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
        """清空短期记忆（新会话开始）"""
        self.short_memory.clear()
        self.current_dialog = []
        self.trajectory = []
