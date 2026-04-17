"""
Week 7 专业Agent实现 - 基于ReActAgentV4的特化继承
角色：产品经理、系统架构师、软件工程师

核心特性：
1. 基于Week 6 ReActAgentV4完整架构继承
2. MetaGPT风格角色定义（role/goal/backstory）
3. 工具集差异化配置
4. 共享长期记忆系统
5. SOP流程节点实现
"""
from react_agent_v4 import ReActAgentV4
from tools.base import ToolRegistry
from message_bus import MessageBus
from typing import Dict, Any


class ProductManagerAgent(ReActAgentV4):
    """
    产品经理Agent（Week 7角色示例）
    
    职责：理解用户需求，撰写PRD文档
    工具：检索工具（查设计模式）、文件工具（写PRD）
    产出：game_design.md
    """
    
    def __init__(self, llm_client=None, bus: MessageBus = None):
        self.bus = bus
        
        super().__init__(
            llm_client=llm_client,
            tool_registry=self._get_pm_tools(),
            max_iterations=5,
            memory_k=3,
            use_long_term=True,
            use_reflection=True,
            
            role="产品经理",
            goal="根据用户需求编写清晰、可执行的PRD文档。"
                 "始终考虑游戏设计最佳实践和用户体验。",
            backstory="你是一位拥有10年行业经验的资深游戏产品经理。"
                     "擅长将模糊的想法转化为具体的游戏机制。"
                     "偏好简洁、结构化的文档风格。"
        )
        
        print(f"[{self.role}] 初始化完成，专业工具: {self.tools.list_tools()}")
    
    def _get_pm_tools(self) -> ToolRegistry:
        """产品经理专属工具集"""
        registry = ToolRegistry()
        
        def retriever_tool(query: str) -> str:
            """检索游戏设计模式文档"""
            patterns = {
                "RPG": "核心循环：探索-战斗-升级-剧情推进",
                "roguelike": "永久死亡+随机地图+回合制",
                "idle": "离线收益+进度保存+简单交互"
            }
            for key in patterns:
                if key in query.lower():
                    return f"[设计模式] {key}: {patterns[key]}"
            return "[设计模式] 未找到特定模式，建议使用通用街机风格。"
        
        registry.register(
            name="retriever_tool",
            func=retriever_tool,
            description="检索游戏设计模式和最佳实践",
            schema={
                "name": "retriever_tool",
                "description": "检索游戏设计模式和最佳实践",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "要研究的游戏类型或机制"}
                    },
                    "required": ["query"]
                }
            }
        )
        
        def file_tool(filename: str, content: str) -> str:
            """写入PRD文档到磁盘"""
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(f"# 游戏设计文档\n\n{content}")
                return f"PRD文档已成功保存到 {filename}"
            except Exception as e:
                return f"保存文件时出错: {e}"
        
        registry.register(
            name="file_tool",
            func=file_tool,
            description="将PRD文档写入文件系统",
            schema={
                "name": "file_tool",
                "description": "将PRD文档写入文件系统",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string", "default": "game_design.md"},
                        "content": {"type": "string", "description": "PRD内容（markdown格式）"}
                    },
                    "required": ["content"]
                }
            }
        )
        
        return registry
    
    def write_prd(self, requirement: str) -> str:
        """
        SOP节点：撰写PRD（产品经理的核心任务）
        
        自动保存到长期记忆，供下游Agent检索
        """
        print(f"\n[{self.role}] 📝 执行SOP节点：撰写PRD")
        
        result = self.run(f"编写PRD文档：{requirement}")
        
        self._lazy_init_long_memory()
        if self.long_memory:
            self.long_memory.add_fact(
                text=f"PRD已完成: game_design.md 基于 '{requirement[:40]}...'",
                category="sop_progress",
                importance=0.9
            )
        
        if self.bus:
            self.bus.broadcast(
                self.role, 
                f"PRD完成: game_design.md",
                control={"sop_node": "pm_complete", "output_file": "game_design.md"}
            )
        
        return result


class SystemArchitectAgent(ReActAgentV4):
    """
    系统架构师Agent（Week 7角色示例）
    
    职责：基于PRD设计技术方案
    工具：Python沙盒（画架构图）、文件工具（写tech_spec）
    输入：game_design.md（从长期记忆或文件读取）
    产出：tech_spec.json
    """
    
    def __init__(self, llm_client=None, bus: MessageBus = None):
        self.bus = bus
        
        super().__init__(
            llm_client=llm_client,
            tool_registry=self._get_architect_tools(),
            max_iterations=5,
            memory_k=3,
            use_long_term=True,
            use_reflection=True,
            
            role="系统架构师",
            goal="为游戏设计健壮、可扩展的技术架构。"
                 "偏好经过验证的模式（MVC、ECS）和清晰的模块分离。",
            backstory="你是一位资深游戏开发者，曾发布过20+款游戏。"
                     "你对过度设计持怀疑态度，但要求架构清晰。"
                     "你通过图表和结构化JSON规格进行沟通。"
        )
    
    def _get_architect_tools(self) -> ToolRegistry:
        """架构师专属工具集"""
        registry = ToolRegistry()
        
        def python_sandbox(code: str) -> str:
            """执行Python代码，用于生成架构图或验证技术可行性"""
            if "matplotlib" in code or "diagram" in code:
                return "[架构图已生成] 包含GameObject、Player、Enemy等类层次结构..."
            return f"[代码执行结果] 已处理 {len(code)} 个字符"
        
        registry.register(
            name="python_sandbox",
            func=python_sandbox,
            description="执行Python代码生成图表或验证逻辑",
            schema={
                "name": "python_sandbox",
                "description": "执行Python代码生成图表或验证逻辑",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "要执行的Python代码"}
                    },
                    "required": ["code"]
                }
            }
        )
        
        def read_prd() -> str:
            """读取ProductManager产出的PRD"""
            try:
                with open("game_design.md", 'r', encoding='utf-8') as f:
                    return f.read()
            except FileNotFoundError:
                return "错误：未找到game_design.md。请确保产品经理已完成任务。"
        
        registry.register(
            name="read_prd",
            func=read_prd,
            description="读取产品经理产出的PRD文档",
            schema={
                "name": "read_prd",
                "description": "读取产品经理产出的PRD文档",
                "parameters": {"type": "object", "properties": {}}
            }
        )
        
        def write_tech_spec(json_content: str) -> str:
            """写入tech_spec.json"""
            try:
                with open("tech_spec.json", 'w', encoding='utf-8') as f:
                    f.write(json_content)
                return "tech_spec.json已成功保存"
            except Exception as e:
                return f"错误: {e}"
        
        registry.register(
            name="write_tech_spec",
            func=write_tech_spec,
            description="写入技术规范JSON文件",
            schema={
                "name": "write_tech_spec",
                "description": "写入技术规范JSON文件",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "json_content": {"type": "string", "description": "JSON格式的技术规范"}
                    },
                    "required": ["json_content"]
                }
            }
        )
        
        return registry
    
    def design_architecture(self, prd_content: str = None) -> str:
        """
        SOP节点：技术架构设计
        
        自动从长期记忆检索上游产出（PRD）如果未提供
        """
        print(f"\n[{self.role}] 🏗️ 执行SOP节点：架构设计")
        
        if not prd_content:
            self._lazy_init_long_memory()
            if self.long_memory:
                facts = self.long_memory.retrieve("PRD game_design", top_k=2)
                if facts:
                    prd_content = facts[0]["text"]
                    print(f"[{self.role}] 从长期记忆检索到上游产出: {prd_content[:50]}...")
            
            # 如果仍然没有获取到PRD内容，使用工具读取
            if not prd_content:
                prd_content = self.tools.execute("read_prd", {})
                print(f"[{self.role}] 从文件读取PRD: {prd_content[:50]}...")
        
        # 如果仍然没有PRD内容，使用默认值
        if not prd_content:
            prd_content = "设计一个Roguelike地牢探险游戏"
            print(f"[{self.role}] 使用默认PRD内容")
        
        task = f"为以下需求设计技术架构: {prd_content[:100]}..."
        result = self.run(task)
        
        self._lazy_init_long_memory()
        if self.long_memory:
            self.long_memory.add_fact(
                text=f"架构设计完成: tech_spec.json 基于PRD",
                category="sop_progress",
                importance=0.9
            )
        
        return result


class SoftwareEngineerAgent(ReActAgentV4):
    """
    软件工程师Agent（Coder）
    
    职责：基于tech_spec实现代码
    工具：Python沙盒（写Pygame代码）、文件工具
    特殊行为：当检测到[IMPOSSIBLE]时，通过MessageBus发送Re-planning信号
    """
    
    def __init__(self, llm_client=None, bus: MessageBus = None):
        self.bus = bus
        
        super().__init__(
            llm_client=llm_client,
            tool_registry=self._get_coder_tools(),
            max_iterations=7,
            memory_k=3,
            use_long_term=True,
            use_reflection=True,
            
            role="软件工程师",
            goal="根据技术规范实现干净、高效的游戏代码。"
                 "遵循PEP8规范，添加注释，处理边缘情况。",
            backstory="你是一位务实的程序员，重视可运行的代码胜过完美的架构。"
                     "你精通Pygame。当发现不可行的需求时，"
                     "你会立即用[IMPOSSIBLE]标签标记它们。"
        )
    
    def _get_coder_tools(self) -> ToolRegistry:
        """工程师专属工具集"""
        registry = ToolRegistry()
        
        def python_sandbox(code: str, mode: str = "execute") -> str:
            """执行或测试Python代码"""
            if "pygame" in code.lower():
                return "[Pygame代码已执行] 窗口已初始化, FPS: 60"
            return "[代码执行结果] 成功"
        
        registry.register(
            name="python_sandbox",
            func=python_sandbox,
            description="编写并执行Python/Pygame代码",
            schema={
                "name": "python_sandbox",
                "description": "编写并执行Python/Pygame代码",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string"},
                        "mode": {"type": "string", "enum": ["execute", "test"], "default": "execute"}
                    },
                    "required": ["code"]
                }
            }
        )
        
        def check_feasibility(requirement: str) -> Dict:
            """检查需求技术可行性"""
            impossible_keywords = ["3d", "online multiplayer", "vr", "blockchain"]
            for kw in impossible_keywords:
                if kw in requirement.lower():
                    return {
                        "feasible": False,
                        "reason": f"{kw} 在当前技术栈中不支持（仅支持Pygame 2D）",
                        "suggest": "使用2D替代方案"
                    }
            return {"feasible": True}
        
        registry.register(
            name="check_feasibility",
            func=check_feasibility,
            description="检查需求在技术上是否可行",
            schema={
                "name": "check_feasibility",
                "description": "检查需求在技术上是否可行",
                "parameters": {
                    "type": "object",
                    "properties": {"requirement": {"type": "string"}},
                    "required": ["requirement"]
                }
            }
        )
        
        return registry
    
    def implement_code(self, tech_spec: str = None) -> str:
        """
        SOP节点：代码实现
        
        特殊逻辑：如果检测到不可行，发送[IMPOSSIBLE]信号到MessageBus
        """
        print(f"\n[{self.role}] 💻 执行SOP节点：代码实现")
        
        if not tech_spec:
            self._lazy_init_long_memory()
            if self.long_memory:
                facts = self.long_memory.retrieve("tech_spec architecture", top_k=2)
                if facts:
                    tech_spec = facts[0]["text"]
        
        check = self.tools.execute("check_feasibility", {"requirement": tech_spec if tech_spec else ""})
        if isinstance(check, str):
            try:
                import json
                check = json.loads(check)
            except:
                check = {"feasible": "feasible" in check.lower() or "true" in check.lower()}
        
        if isinstance(check, dict) and not check.get("feasible", True):
            if self.bus:
                self.bus.send(
                    self.role,
                    "产品经理",
                    f"[IMPOSSIBLE] 无法实现: {check.get('reason', '')}",
                    control={
                        "signal": "IMPOSSIBLE",
                        "reason": check.get("reason", ""),
                        "suggest": check.get("suggest", ""),
                        "blocking_node": "coder_implementation"
                    }
                )
            return f"[IMPOSSIBLE] {check.get('reason', '未知原因')}"
        
        result = self.run(f"基于技术规范实现游戏: {tech_spec[:100]}..." if tech_spec else "实现游戏")
        
        self._lazy_init_long_memory()
        if self.long_memory:
            self.long_memory.add_fact(
                text=f"代码实现完成: main.py 已生成",
                category="sop_progress",
                importance=0.9
            )
        
        return result
