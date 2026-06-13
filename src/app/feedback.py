"""
Feedback Handler — 用户反馈闭环
存储用户对回答的 👍/👎 评价和纠错文本
"""
import json
import uuid
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

FEEDBACK_FILE = Path(__file__).parent.parent.parent / "data" / "feedback.jsonl"


class Feedback:
    """一条用户反馈"""
    def __init__(self, answer_id: str, rating: str,
                 correction: str = "",
                 query: str = "",
                 metadata: Optional[dict] = None):
        self.id = str(uuid.uuid4())[:12]
        self.answer_id = answer_id
        self.rating = rating  # "up" | "down" | "correct"
        self.correction = correction
        self.query = query
        self.metadata = metadata or {}
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "answer_id": self.answer_id,
            "rating": self.rating,
            "correction": self.correction,
            "query": self.query,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


class FeedbackHandler:
    """反馈处理器 — 存储到 JSONL 文件"""

    def __init__(self, filepath: Optional[Path] = None):
        self.filepath = filepath or FEEDBACK_FILE
        # 确保目录存在
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def save_feedback(self, feedback: Feedback) -> bool:
        """保存一条反馈到 JSONL"""
        try:
            with open(self.filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(feedback.to_dict(), ensure_ascii=False) + "\n")
            logger.info("反馈已保存: %s (%s)", feedback.id, feedback.rating)
            return True
        except Exception as e:
            logger.error("保存反馈失败: %s", e)
            return False

    def get_stats(self) -> dict:
        """获取反馈统计"""
        up = down = correct = 0
        try:
            if not self.filepath.exists():
                return {"total": 0, "up": 0, "down": 0, "correct": 0}
            with open(self.filepath, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        r = data.get("rating", "")
                        if r == "up": up += 1
                        elif r == "down": down += 1
                        elif r == "correct": correct += 1
        except Exception as e:
            logger.error("读取反馈统计失败: %s", e)
        return {"total": up + down + correct, "up": up, "down": down, "correct": correct}

    def get_recent(self, limit: int = 20) -> list:
        """获取最近的反馈"""
        results = []
        try:
            if not self.filepath.exists():
                return results
            with open(self.filepath, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        results.append(json.loads(line))
            return results[-limit:]
        except Exception as e:
            logger.error("读取反馈失败: %s", e)
            return results


# 全局单例
feedback_handler = FeedbackHandler()
