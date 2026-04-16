"""
MessageBus 最小实现 - 第一课
基于多线程队列的Agent通信基础设施
"""
import uuid
import time
from typing import TypedDict, Literal, Optional, Dict, List
from queue import Queue, Empty
from threading import Lock


class Message(TypedDict):
    """
    结构化消息信封（选项B设计）
    
    为什么分离content和control？
    - content: 给LLM看的自然语言（用户查询、Agent响应）
    - control: 给系统看的控制指令（[IMPOSSIBLE]、[HUMAN]等）
    避免LLM误解析控制标记，同时支持人机混合协议
    """
    message_id: str           # UUID v4，全局唯一标识
    sender: str              # 发送者agent_id（Bus自动填充，防伪造）
    recipient: Optional[str]  # None表示广播；str表示单播
    content: str             # 业务内容（人类可读）
    msg_type: Literal["direct", "broadcast", "system"]  # 路由类型标记
    timestamp: float         # Unix时间戳（高精度，便于排序）
    metadata: Dict           # 扩展元数据（优先级、线程ID等）
    control: Optional[Dict]  # 控制指令（信号量、跳转目标等）


class MessageBus:
    """
    中心化消息总线（Mediated Communication）
    
    职责：
    1. 消息路由（send:单播, broadcast:多播）
    2. 队列管理（为每个Agent创建独立Inbox）
    3. 历史记录（追加式日志，支持审计与重放）
    """
    
    def __init__(self):
        # 每个Agent的独立收件箱（线程安全队列）
        self._inboxes: Dict[str, Queue] = {}
        # 全局消息历史（追加模式，LangGraph风格）
        self._history: List[Message] = []
        # 保护_history的锁（Queue线程安全，但List需要额外保护）
        self._history_lock = Lock()
        # 统计指标
        self._stats = {"sent": 0, "broadcast": 0, "delivered": 0}
    
    def register(self, agent_id: str) -> None:
        """
        注册Agent到总线，创建其专属Inbox
        
        必须在Agent启动时调用，幂等设计（重复注册不报错）
        """
        if agent_id not in self._inboxes:
            self._inboxes[agent_id] = Queue()
            print(f"[Bus] 注册Agent: {agent_id}")
    
    def send(self, sender: str, recipient: str, content: str, 
             metadata: Dict = None, control: Dict = None) -> Message:
        """
        单播发送（Point-to-Point）
        
        Args:
            sender: 发送者ID（需已注册）
            recipient: 接收者ID（需已注册）
            content: 消息内容
            metadata: 业务元数据（如优先级、reply_to）
            control: 控制指令（如IMPOSSIBLE信号）
            
        Returns:
            创建的消息对象（已入队）
            
        Raises:
            ValueError: 如果recipient未注册
        """
        if recipient not in self._inboxes:
            raise ValueError(f"接收者 '{recipient}' 未注册，消息无法送达")
        
        msg = self._create_message(sender, recipient, "direct", 
                                   content, metadata, control)
        
        # 放入目标Inbox（线程安全操作）
        self._inboxes[recipient].put(msg)
        
        # 追加到全局历史（审计追踪）
        with self._history_lock:
            self._history.append(msg)
        
        self._stats["sent"] += 1
        self._stats["delivered"] += 1
        
        print(f"[Bus] {sender} -> {recipient}: {content[:30]}...")
        return msg
    
    def broadcast(self, sender: str, content: str, 
                  metadata: Dict = None, control: Dict = None) -> Message:
        """
        广播发送（One-to-Many，群聊模式）
        
        实现策略：
        - 同一Message对象被放入所有Inbox（共享引用，内存高效）
        - recipient设为None标记为广播
        """
        if not self._inboxes:
            print("[Bus] 警告：广播时无已注册Agent")
            return None
        
        # 创建广播消息（recipient=None）
        msg = self._create_message(sender, None, "broadcast", 
                                   content, metadata, control)
        
        # 复制消息引用到所有Agent的Inbox（包括发送者自己，模拟群聊可见性）
        for agent_id, inbox in self._inboxes.items():
            inbox.put(msg)
        
        # 追加历史
        with self._history_lock:
            self._history.append(msg)
        
        self._stats["broadcast"] += 1
        self._stats["delivered"] += len(self._inboxes)
        
        print(f"[Bus] {sender} -> [ALL({len(self._inboxes)})]: {content[:30]}...")
        return msg
    
    def receive(self, agent_id: str, timeout: Optional[float] = None) -> Optional[Message]:
        """
        接收消息（阻塞/非阻塞模式）
        
        Args:
            agent_id: 接收者ID
            timeout: None表示永久阻塞；0表示非阻塞；>0表示超时秒数
            
        Returns:
            Message对象 或 None（超时/无消息）
        """
        if agent_id not in self._inboxes:
            raise ValueError(f"Agent '{agent_id}' 未注册，无法接收")
        
        inbox = self._inboxes[agent_id]
        
        try:
            # Queue.get是线程安全的阻塞操作
            msg = inbox.get(block=True, timeout=timeout)
            return msg
        except Empty:
            return None
    
    def get_history(self) -> List[Message]:
        """获取全局消息历史（不可变副本，防止外部修改）"""
        with self._history_lock:
            return self._history.copy()
    
    def get_stats(self) -> Dict:
        """获取投递统计"""
        return self._stats.copy()
    
    def _create_message(self, sender: str, recipient: Optional[str], 
                       msg_type: str, content: str, 
                       metadata: Dict = None, control: Dict = None) -> Message:
        """工厂方法：创建标准化消息结构"""
        return {
            "message_id": str(uuid.uuid4()),
            "sender": sender,
            "recipient": recipient,
            "content": content,
            "msg_type": msg_type,  # direct | broadcast | system
            "timestamp": time.time(),
            "metadata": metadata or {},
            "control": control
        }
