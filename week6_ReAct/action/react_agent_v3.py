"""
ReAct Agent V3 - 集成短期记忆、长期记忆和反思引擎
优化版本：减少token消耗，提高准确性
"""
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from llm_client import KimiClient
from memory.short_term import ConversationSummaryMemory
from memory.long_term import get_long_term_memory, LongTermMemory
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
    - 长期记忆：跨会话持久存储（FAISS向量检索）
    - 反思引擎：自动提取并保存长期记忆
    
    优化特性：
    1. 长期记忆可动态开关
    2. 可信度阈值过滤低质量匹配
    3. 事实冲突检测避免错误信息
    4. 限制记忆数量和长度减少token消耗
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
        ltm_max_facts: int = 3       # 最大返回记忆数量
    ):
        self.llm = llm_client
        self.tools = tool_registry
        self.max_iterations = max_iterations
        self.use_long_term = use_long_term
        self.use_reflection = use_reflection
        self.ltm_threshold = ltm_threshold
        self.ltm_max_facts = ltm_max_facts
        
        # 短期记忆（会话内）
        self.short_memory = ConversationSummaryMemory(k=memory_k)
        
        # 长期记忆（跨会话）- 延迟加载
        self.long_memory = None
        
        # 反思引擎
        if use_reflection:
            self.reflection = ReflectionEngine(llm_client)
        
        self.trajectory: List[Step] = []
        self.current_dialog: List[Dict] = []  # 用于反思的完整对话
        
        # 已保存的事实（用于去重）
        self.saved_facts_cache = set()
    
    def _lazy_init_long_memory(self):
        """延迟初始化长期记忆"""
        if self.use_long_term and self.long_memory is None:
            self.long_memory = get_long_term_memory()
    
    def set_long_term_enabled(self, enabled: bool):
        """动态开关长期记忆"""
        self.use_long_term = enabled
        if enabled and self.long_memory is None:
            self._lazy_init_long_memory()
        print(f"[LTM] 长期记忆已{'开启' if enabled else '关闭'}")
    
    def run(self, query: str) -> str:
        """主循环"""
        # 检查用户是否想要开关长期记忆
        if self._check_memory_switch(query):
            return "已按您的要求调整长期记忆设置。"
        
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
        """检索相关长期记忆（带关键词过滤、阈值过滤和冲突检测）"""
        if not self.use_long_term:
            return []
        
        self._lazy_init_long_memory()
        if self.long_memory is None:
            return []
        
        # 关键词前置过滤：根据查询内容确定需要检索的记忆类别
        relevant_categories = self._get_relevant_categories(query)
        
        results = self.long_memory.retrieve(query, top_k=self.ltm_max_facts * 3)
        
        # 应用关键词过滤：只保留与查询相关的记忆
        filtered = self._filter_by_keywords(query, results)
        
        # 应用类别过滤
        if relevant_categories:
            filtered = [r for r in filtered if r.get('category', 'fact') in relevant_categories]
        
        # 应用可信度阈值过滤（提高阈值，只保留高相关结果）
        filtered = [r for r in filtered if r.get('score', 0) >= self.ltm_threshold]
        
        # 冲突检测：移除矛盾的事实
        final_facts = self._detect_conflicts(filtered)
        
        # 限制数量并格式化
        facts = []
        for r in final_facts[:self.ltm_max_facts]:
            text = r['text'][:100]  # 截断过长内容
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
        """关键词过滤：只保留与查询相关的记忆"""
        if not facts:
            return []
        
        query_lower = query.lower()
        
        # 提取查询中的关键词
        query_keywords = []
        
        # 数字相关
        import re
        numbers = re.findall(r'\d+', query)
        query_keywords.extend(numbers)
        
        # 年份
        years = re.findall(r'\d{4}', query)
        query_keywords.extend(years)
        
        # 名词关键词
        noun_keywords = ['csv', '图表', 'matplotlib', '销售额', '数据', '表格', '绘制', 
                        '名字', '姓名', '年龄', '生日', '工作', '公司', '喜欢', '爱好']
        for keyword in noun_keywords:
            if keyword in query_lower:
                query_keywords.append(keyword)
        
        # 如果查询没有特殊关键词，不过滤（保持向后兼容）
        if not query_keywords:
            return facts
        
        # 过滤：只保留包含至少一个查询关键词的事实
        filtered = []
        for fact in facts:
            fact_text = fact.get('text', '').lower()
            matched = False
            
            # 检查是否有匹配的关键词
            for keyword in query_keywords:
                if keyword.lower() in fact_text:
                    matched = True
                    break
            
            # 如果没有匹配且是身份类记忆，跳过（避免无关的身份信息干扰）
            if matched or fact.get('category') != 'identity':
                filtered.append(fact)
        
        return filtered
    
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
        
        # 精简的system prompt
        system_content = f"""你是一个智能助手。

【已知信息】
{long_term_content}

【会话历史】
{short_facts if short_facts else '无'}

规则：
1. 优先使用已知信息回答
2. 简单计算直接心算
3. 回答简洁"""

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
        if self.use_long_term and self.long_memory:
            profile = self.long_memory.get_user_profile()
            long_summary = f"\n【长期记忆画像】\n偏好: {profile['preference'][:50]}...\n事实: {profile['fact'][:50]}..."
        else:
            long_summary = "\n【长期记忆】已关闭"
        return short + long_summary
    
    def clear_short_memory(self):
        """清空短期记忆（新会话开始）"""
        self.short_memory.clear()
        self.current_dialog = []
        self.trajectory = []