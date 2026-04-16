"""
DummyAgent - 用于测试MessageBus的替身对象
无LLM能力，仅验证通信契约
"""
import threading
import time
from typing import Callable, Optional
from message_bus import MessageBus, Message


class DummyAgent(threading.Thread):
    """
    最小Agent实现（继承Thread实现真并发）
    
    行为模式：
    - 启动后进入监听循环（阻塞receive）
    - 收到消息后执行on_message回调
    - 支持主动send/broadcast
    """
    
    def __init__(self, agent_id: str, bus: MessageBus, 
                 auto_reply: bool = True,
                 handler: Optional[Callable[[Message], str]] = None):
        super().__init__(name=f"Agent-{agent_id}")
        self.agent_id = agent_id
        self.bus = bus
        self.auto_reply = auto_reply
        self.handler = handler or self._default_handler
        
        # 注册到总线
        self.bus.register(agent_id)
        
        # 状态追踪
        self.received_count = 0
        self.running = False
        self.daemon = True  # 主线程退出时自动终止
    
    def _default_handler(self, msg: Message) -> str:
        """默认消息处理器：简单回声"""
        return f"Echo from {self.agent_id}: 已收到 '{msg['content'][:20]}...'"
    
    def send_to(self, recipient: str, content: str, control: dict = None):
        """便捷方法：发送点对点消息"""
        self.bus.send(self.agent_id, recipient, content, control=control)
    
    def broadcast(self, content: str, control: dict = None):
        """便捷方法：广播消息"""
        self.bus.broadcast(self.agent_id, content, control=control)
    
    def run(self):
        """
        主循环：阻塞接收 -> 处理 -> 回复（如果auto_reply=True）
        
        注意：这是真并发线程，与主线程并行运行
        """
        self.running = True
        print(f"[{self.agent_id}] 启动监听...")
        
        while self.running:
            # 阻塞接收（永久等待，直到收到消息或stop）
            msg = self.bus.receive(self.agent_id, timeout=None)
            
            if msg is None:
                continue
            
            self.received_count += 1
            print(f"[{self.agent_id}] 📥 收到来自 {msg['sender']} 的消息: "
                  f"{msg['content'][:40]}... | "
                  f"Control: {msg.get('control')}")
            
            # 处理消息
            response = self.handler(msg)
            
            # 自动回复（仅针对direct消息，避免广播风暴）
            if self.auto_reply and msg['msg_type'] == 'direct':
                # 延迟模拟处理时间
                time.sleep(0.5)
                self.send_to(msg['sender'], response)
    
    def stop(self):
        """优雅停止监听"""
        self.running = False
        # 发送一个空消息解除阻塞（可选优化）
        print(f"[{self.agent_id}] 停止监听，共处理 {self.received_count} 条消息")
