"""
MemoryEnabledAgent - 集成长期记忆的Agent
验证Week 7核心约束：记忆继承与跨Agent共享
"""
from dummy_agent import DummyAgent
from memory_system import get_long_term_memory, ConversationSummaryMemory
from message_bus import MessageBus, Message


class MemoryEnabledAgent(DummyAgent):
    """
    具备记忆能力的Agent（Week 7标准Agent雏形）
    
    继承ReActAgentV4的设计思想（通过交接文档还原）：
    1. 长期记忆：全局单例（共享）
    2. 短期记忆：实例独立（隔离）
    3. SOP集成：完成节点后自动写入长期记忆
    """
    
    def __init__(self, agent_id: str, bus: MessageBus, role: str = "Worker"):
        super().__init__(agent_id, bus, auto_reply=False)  # 先不自动回复，手动控制
        self.role = role
        
        # 关键：获取全局单例（强制共享）
        self.long_memory = get_long_term_memory()
        
        # 短期记忆：独立实例（每个Agent私有）
        self.short_memory = ConversationSummaryMemory(k=3)
        
        # 加载历史用户画像（根据交接文档要求）
        profile = self.long_memory.get_user_profile()
        if profile["preferences"]:
            print(f"[{self.agent_id}] 加载用户画像: {profile['preferences'][:2]}")
    
    def process_with_memory(self, msg: Message):
        """
        带记忆的完整处理流程（模拟ReActAgentV4.run()）
        
        流程：
        1. 接收消息 -> 存入短期记忆
        2. 检索相关长期记忆 -> 注入上下文
        3. 生成响应
        4. 如果是SOP产出 -> 写入长期记忆
        """
        # 1. 短期记忆：记录接收
        self.short_memory.add_message(f"from_{msg['sender']}", msg['content'])
        
        # 2. 长期记忆：检索相关事实（语义检索）
        related_facts = self.long_memory.retrieve(msg['content'], top_k=2)
        context = self._build_context(related_facts)
        
        print(f"\n[{self.agent_id}] 🤔 思考过程:")
        print(f"   短期记忆上下文: {len(self.short_memory.messages)} 条消息")
        print(f"   检索到相关事实: {len(related_facts)} 条")
        for f in related_facts:
            print(f"      - [{f['category']}] {f['text'][:30]}...")
        
        # 3. 生成响应（模拟LLM处理）
        response = self._generate_response(msg, context)
        
        # 4. 如果是SOP节点完成，写入长期记忆（交接文档关键要求）
        if msg.get('control', {}).get('sop_node'):
            output_key = msg['control']['output_key']
            self._save_sop_progress(output_key, response)
        
        return response
    
    def _build_context(self, facts: list) -> str:
        """构建Prompt上下文（长期记忆注入）"""
        if not facts:
            return ""
        return "【相关背景知识】\n" + "\n".join([f"- {f['text']}" for f in facts])
    
    def _generate_response(self, msg: Message, context: str) -> str:
        """模拟LLM响应生成（实际应为ReAct循环）"""
        if context:
            return f"[基于{len(context)}字符背景知识] 处理: {msg['content'][:20]}..."
        return f"[{self.role}] 收到并处理: {msg['content'][:20]}..."
    
    def _save_sop_progress(self, output_key: str, content: str):
        """
        SOP进度保存（跨Agent共享的关键机制）
        
        根据交接文档：Agent完成产出后，必须保存到长期记忆
        使下游Agent（甚至重启后）能检索到
        """
        fact_text = f"{self.role}完成了{output_key}，产出摘要：{content[:50]}"
        self.long_memory.add_fact(
            text=fact_text,
            category="sop_progress",
            importance=0.8,
            agent_id=self.agent_id
        )
        print(f"[{self.agent_id}] 💾 已保存SOP进度到长期记忆: {output_key}")
    
    def check_shared_memory(self):
        """调试接口：查看当前全局记忆状态"""
        all_facts = self.long_memory.get_all_facts()
        print(f"\n[{self.agent_id}] 🔍 全局记忆状态检查:")
        print(f"   总事实数: {len(all_facts)}")
        for f in all_facts:
            print(f"   [{f['agent_id']}] {f['category']}: {f['text'][:40]}...")
