"""
消息总线模块 - 实现Agent间通信基础设施
"""
import uuid
import time
from typing import TypedDict, Literal, Optional, Dict, List
from queue import Queue, Empty
from threading import Lock

class Message(TypedDict):
    """结构化消息信封"""
    message_id: str
    sender: str
    recipient: Optional[str]
    content: str
    msg_type: Literal["direct", "broadcast", "system"]
    timestamp: float
    metadata: Dict
    control: Optional[Dict]

class MessageBus:
    """
    中心化消息总线
    
    职责：
    1. 消息路由（单播/广播）
    2. 队列管理（为每个Agent创建独立Inbox）
    3. 历史记录（支持审计与重放）
    """
    
    def __init__(self):
        self._inboxes: Dict[str, Queue] = {}
        self._history: List[Message] = []
        self._history_lock = Lock()
        self._stats = {"sent": 0, "broadcast": 0, "delivered": 0}
    
    def register(self, agent_id: str) -> None:
        """注册Agent到总线"""
        if agent_id not in self._inboxes:
            self._inboxes[agent_id] = Queue()
    
    def send(self, sender: str, recipient: str, content: str, 
             metadata: Dict = None, control: Dict = None) -> Message:
        """单播发送"""
        if recipient not in self._inboxes:
            raise ValueError(f"接收者 '{recipient}' 未注册")
        
        msg = self._create_message(sender, recipient, "direct", content, metadata, control)
        self._inboxes[recipient].put(msg)
        
        with self._history_lock:
            self._history.append(msg)
        
        self._stats["sent"] += 1
        self._stats["delivered"] += 1
        
        return msg
    
    def broadcast(self, sender: str, content: str, 
                  metadata: Dict = None, control: Dict = None) -> Message:
        """广播发送"""
        if not self._inboxes:
            return None
        
        msg = self._create_message(sender, None, "broadcast", content, metadata, control)
        
        for agent_id, inbox in self._inboxes.items():
            inbox.put(msg)
        
        with self._history_lock:
            self._history.append(msg)
        
        self._stats["broadcast"] += 1
        self._stats["delivered"] += len(self._inboxes)
        
        return msg
    
    def receive(self, agent_id: str, timeout: Optional[float] = None) -> Optional[Message]:
        """接收消息"""
        if agent_id not in self._inboxes:
            raise ValueError(f"Agent '{agent_id}' 未注册")
        
        inbox = self._inboxes[agent_id]
        
        try:
            msg = inbox.get(block=True, timeout=timeout)
            return msg
        except Empty:
            return None
    
    def get_history(self) -> List[Message]:
        """获取全局消息历史"""
        with self._history_lock:
            return self._history.copy()
    
    def get_stats(self) -> Dict:
        """获取投递统计"""
        return self._stats.copy()
    
    def _create_message(self, sender: str, recipient: Optional[str], 
                       msg_type: str, content: str, 
                       metadata: Dict = None, control: Dict = None) -> Message:
        """创建标准化消息结构"""
        return {
            "message_id": str(uuid.uuid4()),
            "sender": sender,
            "recipient": recipient,
            "content": content,
            "msg_type": msg_type,
            "timestamp": time.time(),
            "metadata": metadata or {},
            "control": control
        }
