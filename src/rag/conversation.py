"""
Conversation Manager — 多轮对话支持
管理 conversation_id、消息历史、上下文注入
"""
import uuid
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# 注入到 RAG 上下文的最大历史轮次
MAX_HISTORY_TURNS = 5


@dataclass
class Message:
    """单条对话消息"""
    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: float = 0.0


@dataclass
class Conversation:
    """一次对话会话"""
    id: str
    messages: list = field(default_factory=list)
    profile: dict = field(default_factory=dict)  # 用户画像
    created_at: float = 0.0

    def add_message(self, role: str, content: str):
        self.messages.append(Message(role=role, content=content))

    def get_context(self, max_turns: int = MAX_HISTORY_TURNS) -> str:
        """将最近 max_turns 轮对话格式化为上下文文本"""
        recent = self.messages[-(max_turns * 2):]
        lines = ["## 对话历史"]
        for msg in recent:
            prefix = "用户" if msg.role == "user" else "系统"
            lines.append(f"{prefix}: {msg.content[:200]}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "messages": [{"role": m.role, "content": m.content}
                         for m in self.messages],
            "profile": self.profile,
        }


class ConversationManager:
    """对话管理器 — 内存存储，支持创建/追加/查询/清理"""

    def __init__(self):
        self._conversations: dict[str, Conversation] = {}

    def create(self, profile: Optional[dict] = None) -> Conversation:
        """创建新对话"""
        conv = Conversation(
            id=str(uuid.uuid4())[:8],
            profile=profile or {},
        )
        self._conversations[conv.id] = conv
        logger.info("创建新对话: %s", conv.id)
        return conv

    def get(self, conv_id: str) -> Optional[Conversation]:
        """获取指定对话"""
        return self._conversations.get(conv_id)

    def add_message(self, conv_id: str, role: str, content: str) -> Optional[Conversation]:
        """向对话追加消息"""
        conv = self.get(conv_id)
        if not conv:
            logger.warning("对话 %s 不存在，创建新对话", conv_id)
            conv = self.create()
            # 将消息加到新对话
        conv.add_message(role, content)
        return conv

    def get_context(self, conv_id: str, max_turns: int = MAX_HISTORY_TURNS) -> str:
        """获取指定对话的上下文"""
        conv = self.get(conv_id)
        if not conv:
            return ""
        return conv.get_context(max_turns)

    def update_profile(self, conv_id: str, profile: dict):
        """更新用户画像"""
        conv = self.get(conv_id)
        if conv:
            conv.profile.update(profile)

    def clear(self, conv_id: str) -> bool:
        """清除指定对话"""
        if conv_id in self._conversations:
            del self._conversations[conv_id]
            logger.info("清除对话: %s", conv_id)
            return True
        return False

    def clear_all(self):
        """清除所有对话"""
        count = len(self._conversations)
        self._conversations.clear()
        logger.info("清除全部 %d 个对话", count)


# 全局单例
conversation_manager = ConversationManager()
