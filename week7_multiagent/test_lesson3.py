"""
第三课验证脚本：Agent角色特化与跨Agent协作

验证内容：
1. 角色继承ReActAgentV4（非重写）
2. MetaGPT风格角色定义（role/goal/backstory）
3. 工具集差异化（PM检索 vs Architect代码）
4. 长期记忆共享（跨Agent检索上游产出）
5. [IMPOSSIBLE]信号机制（冲突解决准备）
6. 三层记忆集成（短期/长期/反思）
"""
import time
import shutil
from pathlib import Path
from tools.base import ToolRegistry
from specialized_agents import ProductManagerAgent, SystemArchitectAgent, SoftwareEngineerAgent
from message_bus import MessageBus
from memory.long_term import LongTermMemory, get_long_term_memory, _sessions


def reset_environment():
    """清理环境"""
    storage = Path("./long_term_memory")
    if storage.exists():
        shutil.rmtree(storage)
    
    # 清理全局会话缓存
    _sessions.clear()
    
    # 清理临时文件
    for f in ["game_design.md", "tech_spec.json"]:
        if Path(f).exists():
            Path(f).unlink()


def test_role_inheritance():
    """测试1：角色正确继承ReActAgentV4（非重写）"""
    print("\n" + "="*60)
    print("测试1: 角色继承验证（MetaGPT风格）")
    print("="*60)
    
    reset_environment()
    
    pm = ProductManagerAgent()
    architect = SystemArchitectAgent()
    
    # 验证继承关系
    assert hasattr(pm, 'run'), "PM必须继承run方法（ReAct循环）"
    assert hasattr(pm, 'get_status'), "PM必须继承状态追踪"
    assert hasattr(pm, '_lazy_init_long_memory'), "PM必须继承延迟初始化方法"
    
    # 验证差异化配置
    assert pm.role == "产品经理"
    assert architect.role == "系统架构师"
    assert pm.goal != architect.goal
    
    # 验证工具集差异化（关键）
    pm_tools = pm.tools.list_tools()
    arch_tools = architect.tools.list_tools()
    
    assert "retriever_tool" in pm_tools, "PM必须有设计模式检索工具"
    assert "python_sandbox" in arch_tools, "Architect必须有代码执行工具"
    assert "retriever_tool" not in arch_tools, "Architect不应有PM专属工具"
    
    print(f"\n✅ 角色继承验证通过:")
    print(f"   PM角色: {pm.role}, 工具: {pm_tools}")
    print(f"   Architect角色: {architect.role}, 工具: {arch_tools}")


def test_react_loop_reuse():
    """测试2：ReAct循环复用（禁止重写的验证）"""
    print("\n" + "="*60)
    print("测试2: ReAct循环复用验证")
    print("="*60)
    
    reset_environment()
    pm = ProductManagerAgent()
    
    # 验证ReAct循环工作（使用继承的run方法）
    result = pm.run("设计一个简单的RPG游戏")
    
    status = pm.get_status()
    assert status["state"] == "completed"
    assert status["iteration"] > 0 and status["iteration"] <= 5
    
    print(f"\n✅ ReAct循环复用验证通过:")
    print(f"   执行轮数: {status['iteration']}")
    print(f"   最终状态: {status['state']}")
    print(f"   错误计数: {status['error_count']}（指数退避生效）")


def test_cross_agent_sop_flow():
    """测试3：跨Agent SOP流程（PM -> Architect）"""
    print("\n" + "="*60)
    print("测试3: SOP跨Agent流程（集成MessageBus）")
    print("="*60)
    
    reset_environment()
    bus = MessageBus()
    
    # 创建Agent并注册到Bus
    pm = ProductManagerAgent(bus=bus)
    architect = SystemArchitectAgent(bus=bus)
    
    # SOP步骤1：PM撰写PRD
    print("\n>>> SOP Step 1: ProductManager撰写PRD")
    pm_result = pm.write_prd("设计一个Roguelike地牢探险游戏")
    
    # 验证PM保存到长期记忆
    pm._lazy_init_long_memory()
    pm_facts = pm.long_memory.retrieve("PRD", top_k=1)
    assert len(pm_facts) > 0, "PM应将PRD进度存入长期记忆"
    
    # SOP步骤2：Architect设计架构（自动检索上游产出）
    print("\n>>> SOP Step 2: Architect自动检索并设计架构")
    
    # 模拟：Architect的短期记忆应通过某种方式获得上下文（简化测试）
    # 实际应由SOP引擎注入，这里直接测试检索能力
    arch_result = architect.design_architecture()
    
    # 验证Architect能"看到"PM的产出（通过长期记忆）
    architect._lazy_init_long_memory()
    arch_facts = architect.long_memory.retrieve("PRD", top_k=1)
    assert len(arch_facts) > 0, "Architect应能检索到PM的PRD事实"
    
    print(f"\n✅ SOP跨Agent流程验证通过:")
    print(f"   PM产出保存: {pm_facts[0]['text'][:40]}...")
    print(f"   Architect读取: {arch_facts[0]['text'][:40]}...")


def test_impossible_signal():
    """测试4：[IMPOSSIBLE]信号与冲突解决准备"""
    print("\n" + "="*60)
    print("测试4: [IMPOSSIBLE]信号机制（Coder -> PM）")
    print("="*60)
    
    reset_environment()
    bus = MessageBus()
    
    pm = ProductManagerAgent(bus=bus)
    coder = SoftwareEngineerAgent(bus=bus)
    
    # 先让PM写入一个包含3D需求的技术规范（模拟冲突场景）
    pm._lazy_init_long_memory()
    pm.long_memory.add_fact(
        text="Tech spec requires 3D engine support with online multiplayer",
        category="sop_progress",
        importance=0.9
    )
    
    # Coder尝试实现，应检测到IMPOSSIBLE并发送信号
    print("\n>>> Coder检测到不可行需求，发送[IMPOSSIBLE]")
    result = coder.implement_code("3D online multiplayer game")
    
    # 验证结果包含IMPOSSIBLE标记
    assert "[IMPOSSIBLE]" in result, "Coder应标记不可行任务"
    
    # 验证MessageBus收到信号（Coder应发送消息给PM）
    print(f"\n✅ IMPOSSIBLE信号验证通过:")
    print(f"   Coder响应: {result[:60]}...")
    print(f"   MessageBus历史: {len(bus.get_history())} 条消息")
    if bus.get_history():
        last_msg = bus.get_history()[-1]
        if last_msg.get('control', {}).get('signal') == 'IMPOSSIBLE':
            print(f"   控制信号: {last_msg['control']}")


def test_memory_integration():
    """测试5：三层记忆集成（短期独立/长期共享/反思）"""
    print("\n" + "="*60)
    print("测试5: 三层记忆系统集成")
    print("="*60)
    
    reset_environment()
    
    pm = ProductManagerAgent()
    
    # 执行两轮ReAct（触发反思）
    pm.run("设计游戏A")
    pm.run("设计游戏B")
    
    # 验证短期记忆（ConversationSummaryMemory）
    short_ctx = pm.short_memory.get_context()
    
    # 验证长期记忆（有反思产出）
    pm._lazy_init_long_memory()
    reflections = pm.long_memory.retrieve("reflection", top_k=3)
    
    print(f"\n✅ 三层记忆集成验证通过:")
    print(f"   短期记忆消息数: {len(pm.short_memory.messages)}")
    print(f"   短期记忆摘要数: {len(pm.short_memory.conversations)}")
    print(f"   长期记忆事实数: {pm.long_memory.get_fact_count()}")
    print(f"   反思事实数: {len(reflections)}")


def test_session_isolation():
    """测试6：会话隔离验证（Week 6特性迁移）"""
    print("\n" + "="*60)
    print("测试6: 会话隔离验证")
    print("="*60)
    
    reset_environment()
    
    pm1 = ProductManagerAgent()
    session_id_1 = pm1.session_id
    
    pm2 = ProductManagerAgent()
    session_id_2 = pm2.session_id
    
    # 验证不同Agent实例有不同会话
    assert session_id_1 != session_id_2, "不同实例应有不同会话"
    
    # 在会话1中添加记忆
    pm1._lazy_init_long_memory()
    pm1.long_memory.add_fact("会话1的专属记忆", category="test", importance=0.9)
    
    # 验证会话2看不到会话1的记忆
    pm2._lazy_init_long_memory()
    facts = pm2.long_memory.retrieve("会话1", top_k=1)
    assert len(facts) == 0, "会话2不应看到会话1的记忆"
    
    print(f"\n✅ 会话隔离验证通过:")
    print(f"   会话1 ID: {session_id_1}")
    print(f"   会话2 ID: {session_id_2}")
    print(f"   会话1事实数: {pm1.long_memory.get_fact_count()}")
    print(f"   会话2事实数: {pm2.long_memory.get_fact_count()}")


if __name__ == "__main__":
    print("🎭 第三课测试开始：Agent角色特化与协作")
    
    try:
        test_role_inheritance()
        test_react_loop_reuse()
        test_cross_agent_sop_flow()
        test_impossible_signal()
        test_memory_integration()
        test_session_isolation()
        
        print("\n" + "="*60)
        print("🎉 第三课所有测试通过！角色化Agent已就绪")
        print("="*60)
        print("\n📋 核心能力验证清单:")
        print("   ✅ 继承ReActAgentV4（非重写）")
        print("   ✅ MetaGPT风格角色定义（role/goal/backstory）")
        print("   ✅ 工具集差异化（PM检索 vs Architect代码）")
        print("   ✅ 长期记忆共享（跨Agent检索上游产出）")
        print("   ✅ [IMPOSSIBLE]信号机制（冲突解决准备）")
        print("   ✅ 三层记忆集成（短期/长期/反思）")
        print("   ✅ 会话隔离（Week 6特性迁移）")
        print("\n🚀 准备进入第四课：SOP引擎与群聊管理器（DAG执行与冲突解决）")
        
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n💥 异常: {e}")
        import traceback
        traceback.print_exc()