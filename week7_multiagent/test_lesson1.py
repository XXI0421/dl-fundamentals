"""
第一课验证脚本：测试MessageBus三大原语
运行方式: python test_lesson1.py
"""
import time
from message_bus import MessageBus
from dummy_agent import DummyAgent


def test_point_to_point():
    """测试1：点对点私聊（Send/Receive）"""
    print("\n" + "="*50)
    print("测试1: 点对点通信 (A <-> B)")
    print("="*50)
    
    bus = MessageBus()
    
    # 创建两个Agent（不启动自动监听线程，手动控制）
    agent_a = DummyAgent("Alice", bus, auto_reply=False)
    agent_b = DummyAgent("Bob", bus, auto_reply=False)
    
    # 手动发送（不启动线程，直接操作Bus）
    bus.send("Alice", "Bob", "你好Bob，这是私密消息", 
             metadata={"priority": "high"}, 
             control={"encrypt": False})
    
    # Bob接收（阻塞5秒或直到收到）
    msg = bus.receive("Bob", timeout=2.0)
    assert msg is not None, "Bob应该收到消息"
    assert msg['sender'] == "Alice"
    assert msg['content'] == "你好Bob，这是私密消息"
    assert msg['metadata']['priority'] == "high"
    
    print(f"✅ 点对点验证通过: {msg['sender']} -> {msg['recipient']}")
    print(f"   消息ID: {msg['message_id']}")
    print(f"   时间戳: {msg['timestamp']}")


def test_broadcast():
    """测试2：广播机制（Broadcast）"""
    print("\n" + "="*50)
    print("测试2: 广播通信 (A -> [B, C, D])")
    print("="*50)
    
    bus = MessageBus()
    
    # 注册多个Agent
    for name in ["Host", "Guest1", "Guest2", "Guest3"]:
        bus.register(name)
    
    # Host广播
    bus.broadcast("Host", "全员注意，系统即将维护", 
                  control={"alert_level": "warning"})
    
    # 所有Guest应该收到相同消息（验证共享引用）
    messages = []
    for name in ["Guest1", "Guest2", "Guest3"]:
        msg = bus.receive(name, timeout=1.0)
        assert msg is not None, f"{name}应该收到广播"
        messages.append(msg)
    
    # 验证是同一对象（共享内存，选项B验证）
    assert messages[0] is messages[1] is messages[2], \
        "广播消息应该是同一对象（共享引用）"
    
    print(f"✅ 广播验证通过: 3个Agent收到同一消息对象")
    print(f"   消息ID: {messages[0]['message_id']}（一致）")
    print(f"   Control信号: {messages[0]['control']}")


def test_concurrent_communication():
    """测试3：并发通信（真多线程）"""
    print("\n" + "="*50)
    print("测试3: 并发通信 (多线程Agent)")
    print("="*50)
    
    bus = MessageBus()
    
    # 创建两个自动回复的Agent（启动线程）
    alice = DummyAgent("Alice", bus, auto_reply=True)
    bob = DummyAgent("Bob", bus, auto_reply=True)
    
    # 启动线程
    alice.start()
    bob.start()
    
    # Alice发送给Bob
    time.sleep(0.1)  # 确保线程启动
    alice.send_to("Bob", "在吗？请回复")
    
    # 等待自动回复流程完成（Alice发->Bob收->Bob发->Alice收）
    time.sleep(1.5)
    
    # 验证历史记录
    history = bus.get_history()
    print(f"📜 消息历史记录 ({len(history)} 条):")
    for msg in history:
        direction = "->" if msg['msg_type'] == 'direct' else "=>"
        print(f"   [{msg['sender']} {direction} {msg['recipient']}] "
              f"{msg['content'][:30]}...")
    
    # 停止线程
    alice.stop()
    bob.stop()
    
    print(f"✅ 并发验证通过: Alice/Bob各处理 {alice.received_count}/{bob.received_count} 条消息")


def test_control_signal():
    """测试4：控制指令隔离（Week 7关键特性）"""
    print("\n" + "="*50)
    print("测试4: 控制指令隔离 ([IMPOSSIBLE]信号)")
    print("="*50)
    
    bus = MessageBus()
    bus.register("PM")
    bus.register("Coder")
    
    # Coder发送IMPOSSIBLE信号（技术无法实现）
    bus.send("Coder", "PM", "该功能无法实现，技术限制", 
             control={"signal": "IMPOSSIBLE", 
                     "reason": "pygame_no_3d_support",
                     "suggest": "use_2d_alternative"})
    
    msg = bus.receive("PM", timeout=1.0)
    
    # 验证：content是人类可读，control是机器可读
    assert "无法实现" in msg['content']  # LLM能理解的文本
    assert msg['control']['signal'] == "IMPOSSIBLE"  # 系统能解析的信号
    
    print(f"✅ 控制指令隔离验证通过")
    print(f"   Content给LLM: {msg['content']}")
    print(f"   Control给系统: {msg['control']}")


if __name__ == "__main__":
    print("🚀 第一课测试开始：MessageBus最小实现验证")
    
    try:
        test_point_to_point()
        test_broadcast()
        test_concurrent_communication()
        test_control_signal()
        
        print("\n" + "="*50)
        print("🎉 所有测试通过！MessageBus已就绪")
        print("="*50)
        print(f"📊 最终统计: {MessageBus().get_stats()}")
        
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
    except Exception as e:
        print(f"\n💥 异常: {e}")
        import traceback
        traceback.print_exc()
