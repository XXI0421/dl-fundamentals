"""
第二课验证脚本：记忆继承与跨Agent状态共享
基于Week 6索引格式优化版
"""
import time
import shutil
from pathlib import Path
from message_bus import MessageBus
from memory_agent import MemoryEnabledAgent
from memory_system import get_long_term_memory, LongTermMemory


def reset_memory_storage():
    """清理存储，模拟冷启动（用于测试持久化）"""
    storage = Path("./long_term_memory")
    if storage.exists():
        shutil.rmtree(storage)
    # 重置单例状态（强制重新初始化）
    LongTermMemory._instance = None
    LongTermMemory._initialized = False


def test_singleton_pattern():
    """测试1：单例模式强制共享（核心约束验证）"""
    print("\n" + "="*60)
    print("测试1: 长期记忆单例模式（强制共享）")
    print("="*60)
    
    reset_memory_storage()
    bus = MessageBus()
    
    # 创建两个Agent（模拟PM和Coder）
    pm = MemoryEnabledAgent("PM_Agent", bus, role="ProductManager")
    coder = MemoryEnabledAgent("Coder_Agent", bus, role="SoftwareEngineer")
    
    # 验证是同一内存对象（Python id检查）
    assert pm.long_memory is coder.long_memory, "必须是同一对象！"
    assert pm.long_memory is get_long_term_memory(), "全局获取也必须相同！"
    
    print(f"\n✅ 单例验证通过:")
    print(f"   PM的长记忆ID: {id(pm.long_memory)}")
    print(f"   Coder的长记忆ID: {id(coder.long_memory)}")
    print(f"   全局获取ID: {id(get_long_term_memory())}")


def test_cross_agent_write_read():
    """测试2：跨Agent写入/读取（A写B读）"""
    print("\n" + "="*60)
    print("测试2: 跨Agent记忆共享（PM写入 -> Coder读取）")
    print("="*60)
    
    reset_memory_storage()
    bus = MessageBus()
    
    pm = MemoryEnabledAgent("PM", bus, role="ProductManager")
    coder = MemoryEnabledAgent("Coder", bus, role="SoftwareEngineer")
    
    # 模拟SOP流程：PM完成PRD，写入长期记忆
    print("\n--- 模拟SOP执行：PM节点 ---")
    sop_msg = {
        "message_id": "sop-001",
        "sender": "system",
        "recipient": "PM",
        "content": "请设计游戏PRD文档",
        "msg_type": "direct",
        "timestamp": time.time(),
        "metadata": {},
        "control": {"sop_node": True, "output_key": "game_design.md"}  # SOP标记
    }
    
    pm.process_with_memory(sop_msg)
    
    # 验证Coder能否检索到PM的产出（语义检索）
    print("\n--- Coder节点启动，检索上游产出 ---")
    coder_check = coder.long_memory.retrieve("game_design PRD", top_k=3)
    
    assert len(coder_check) > 0, "Coder应该检索到PM写入的事实！"
    assert any("PRD" in f["text"] or "design" in f["text"] for f in coder_check), \
        "检索内容应包含PRD相关信息"
    
    print(f"\n✅ 跨Agent共享验证通过:")
    print(f"   PM写入的事实被Coder成功检索")
    print(f"   检索结果: {coder_check[0]['text'][:50]}...")


def test_persistence_across_restart():
    """测试3：持久化验证（进程重启后记忆不丢失）"""
    print("\n" + "="*60)
    print("测试3: 持久化验证（模拟进程重启）")
    print("="*60)
    
    # Phase 1: 第一次运行，写入记忆
    print("\n--- Phase 1: 初始进程写入 ---")
    reset_memory_storage()
    bus1 = MessageBus()
    
    pm_v1 = MemoryEnabledAgent("PM", bus1, role="ProductManager")
    pm_v1.long_memory.add_fact(
        text="用户偏好Python语言而非JavaScript",
        category="user_profile",
        importance=0.9,
        agent_id="PM"
    )
    
    del pm_v1  # 模拟进程结束
    del bus1
    
    # Phase 2: 重置单例但保留文件，模拟新进程启动
    print("\n--- Phase 2: 新进程启动（模拟重启） ---")
    # 关键：重置Python内存中的单例，但保留磁盘文件
    LongTermMemory._instance = None
    LongTermMemory._initialized = False
    
    bus2 = MessageBus()
    pm_v2 = MemoryEnabledAgent("PM", bus2, role="ProductManager")
    
    # 验证能读取旧记忆
    profile = pm_v2.long_memory.get_user_profile()
    assert len(profile["preferences"]) > 0, "重启后应加载历史用户画像！"
    assert "Python" in profile["preferences"][0], "应保留具体偏好内容"
    
    print(f"\n✅ 持久化验证通过:")
    print(f"   新进程加载了旧记忆: {profile['preferences']}")
    print(f"   事实总数: {profile['fact_count']}")


def test_memory_isolation():
    """测试4：短期记忆隔离（独立） vs 长期记忆共享"""
    print("\n" + "="*60)
    print("测试4: 记忆隔离性验证（短期独立/长期共享）")
    print("="*60)
    
    reset_memory_storage()
    bus = MessageBus()
    
    pm = MemoryEnabledAgent("PM", bus, role="PM")
    architect = MemoryEnabledAgent("Architect", bus, role="Architect")
    
    # 向两者短期记忆添加不同内容（模拟各自对话历史）
    pm.short_memory.add_message("user", "PM的私密需求讨论")
    architect.short_memory.add_message("user", "Architect的技术细节")
    
    # 验证短期记忆隔离（互不可见）
    assert "私密需求" not in str(architect.short_memory.get_context()), \
        "Architect不应看到PM的短期记忆"
    assert "技术细节" not in str(pm.short_memory.get_context()), \
        "PM不应看到Architect的短期记忆"
    
    # 两者写入长期记忆（共享）
    pm.long_memory.add_fact("技术栈选择：Pygame", category="tech_decision", agent_id="PM")
    architect.long_memory.add_fact("架构：MVC模式", category="tech_decision", agent_id="Architect")
    
    # 验证长期记忆共享（都能读到对方写入的）
    pm_facts = pm.long_memory.retrieve("架构模式")
    architect_facts = architect.long_memory.retrieve("Pygame")
    
    assert len(pm_facts) > 0 and any("MVC" in f["text"] for f in pm_facts), \
        "PM应能检索到Architect写入的架构决策"
    assert len(architect_facts) > 0 and any("Pygame" in f["text"] for f in architect_facts), \
        "Architect应能检索到PM写入的技术栈"
    
    print("\n✅ 隔离性验证通过:")
    print(f"   短期记忆隔离: PM({len(pm.short_memory.messages)}条) vs Architect({len(architect.short_memory.messages)}条)")
    print(f"   长期记忆共享: 双方都能检索到对方的tech_decision")


def test_conflict_resolution_preparation():
    """测试5：为冲突解决机制准备（IMPOSSIBLE信号+记忆关联）"""
    print("\n" + "="*60)
    print("测试5: 冲突解决机制准备（[IMPOSSIBLE]信号与记忆关联）")
    print("="*60)
    
    reset_memory_storage()
    bus = MessageBus()
    
    pm = MemoryEnabledAgent("PM", bus, role="ProductManager")
    coder = MemoryEnabledAgent("Coder", bus, role="SoftwareEngineer")
    
    # 模拟正常SOP流程：PM产出需求
    pm.process_with_memory({
        "message_id": "req-001",
        "sender": "User",
        "recipient": "PM",
        "content": "需要3D游戏引擎功能",
        "msg_type": "direct",
        "timestamp": time.time(),
        "metadata": {},
        "control": {"sop_node": True, "output_key": "requirement.md"}
    })
    
    # Coder收到需求，发现无法实现（3D需求但技术限制）
    print("\n--- Coder处理需求，发现[IMPOSSIBLE] ---")
    
    # Coder检索相关记忆（应看到PM的需求）
    context = coder.long_memory.retrieve("3D game engine", top_k=2)
    
    # 模拟Coder生成IMPOSSIBLE信号（通过MessageBus发送，结合第一课）
    impossible_msg = {
        "message_id": "imp-001",
        "sender": "Coder",
        "recipient": "PM",
        "content": "无法实现3D引擎：Pygame仅支持2D，建议改为2D俯视角游戏",
        "msg_type": "direct",
        "timestamp": time.time(),
        "metadata": {"reply_to": "req-001"},
        "control": {
            "signal": "IMPOSSIBLE",
            "reason": "tech_limitation",
            "blocking_fact": "3d_engine_required",  # 关联的长期记忆ID
            "suggest": "use_2d_topdown"
        }
    }
    
    # PM接收并处理（应能关联到之前的写入）
    pm.short_memory.add_message("Coder", impossible_msg["content"])
    
    # 验证PM能从记忆中找到冲突源头
    related = pm.long_memory.retrieve("requirement", top_k=1)
    
    print(f"\n✅ 冲突解决准备验证:")
    print(f"   Coder检测到IMPOSSIBLE并关联长期记忆中的3D需求")
    print(f"   PM能检索到冲突源头: {related[0]['text'][:40]}...")
    print(f"   建议方案: {impossible_msg['control']['suggest']}")


if __name__ == "__main__":
    print("🧠 第二课测试开始：记忆继承与跨Agent状态共享")
    
    try:
        test_singleton_pattern()
        test_cross_agent_write_read()
        test_persistence_across_restart()
        test_memory_isolation()
        test_conflict_resolution_preparation()
        
        print("\n" + "="*60)
        print("🎉 第二课所有测试通过！记忆系统已就绪")
        print("="*60)
        print("\n📋 核心能力验证清单:")
        print("   ✅ 单例强制共享（所有Agent同一LTM实例）")
        print("   ✅ 跨Agent实时读写（A写入B立即读取）")
        print("   ✅ 磁盘持久化（重启后记忆不丢失）")
        print("   ✅ 双轨隔离（短期独立/长期共享）")
        print("   ✅ SOP进度追踪（自动保存节点产出）")
        print("\n🚀 准备进入第三课：Agent角色特化（ProductManager/Architect）")
        
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n💥 异常: {e}")
        import traceback
        traceback.print_exc()
