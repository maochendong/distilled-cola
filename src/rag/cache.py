"""
SemanticCache — 语义问答缓存
基于 embedding 余弦相似度，命中阈值以上直接返回缓存结果
"""

from __future__ import annotations

import time
import logging
from typing import Optional

import numpy as np

from src.knowledge_base.embedder import Embedder

logger = logging.getLogger(__name__)


class SemanticCache:
    """语义缓存：相同/相似问题直接返回缓存，避免重复 LLM 调用。"""

    def __init__(self, threshold: float = 0.92, ttl: int = 3600, embedder: Optional[Embedder] = None):
        self.threshold = threshold
        self.ttl = ttl  # 缓存有效期（秒）
        self.embedder = embedder or Embedder()
        self._store: dict[str, dict] = {}       # key → cached result
        self._embeddings: dict[str, list[float]] = {}  # key → query embedding
        self._timestamps: dict[str, float] = {}  # key → created_at
        self.hits = 0
        self.misses = 0

    def _cosine_sim(self, a: list[float], b: list[float]) -> float:
        """余弦相似度。"""
        a_arr = np.array(a, dtype=np.float32)
        b_arr = np.array(b, dtype=np.float32)
        dot = float(np.dot(a_arr, b_arr))
        norm = float(np.linalg.norm(a_arr) * np.linalg.norm(b_arr) + 1e-10)
        return dot / norm

    def _is_expired(self, key: str) -> bool:
        """检查缓存是否过期。"""
        if self.ttl <= 0:
            return False
        return time.time() - self._timestamps.get(key, 0) > self.ttl

    def _evict_expired(self):
        """清理过期缓存。"""
        expired = [k for k in self._store if self._is_expired(k)]
        for k in expired:
            del self._store[k]
            self._embeddings.pop(k, None)
            self._timestamps.pop(k, None)
        if expired:
            logger.debug("清除 %d 条过期缓存", len(expired))

    def get(self, query: str) -> Optional[dict]:
        """查找语义最相似的缓存结果。

        Args:
            query: 用户问题

        Returns:
            缓存的结果 dict，或 None（未命中）
        """
        if not self.embedder.is_available():
            self.misses += 1
            return None

        self._evict_expired()
        if not self._store:
            self.misses += 1
            return None

        q_emb = self.embedder.embed_single(query)
        if not q_emb:
            self.misses += 1
            return None

        best_key, best_sim = None, 0.0
        for key, cached_emb in self._embeddings.items():
            sim = self._cosine_sim(q_emb, cached_emb)
            if sim > best_sim:
                best_sim = sim
                best_key = key

        if best_key and best_sim >= self.threshold:
            self.hits += 1
            result = dict(self._store[best_key])
            result["cached"] = True
            result["cache_similarity"] = round(best_sim, 4)
            logger.info("缓存命中 key=%s sim=%.4f", best_key, best_sim)
            return result

        self.misses += 1
        return None

    def set(self, query: str, result: dict):
        """写入缓存。

        Args:
            query: 用户问题（用于生成 embedding）
            result: 问答结果 dict
        """
        if not self.embedder.is_available():
            return

        q_emb = self.embedder.embed_single(query)
        if not q_emb:
            return

        key = f"q_{hash(query) % 10**10:010x}"
        # 只存可序列化的内容，不存 embedding（单独存）
        cache_entry = {k: v for k, v in result.items() if k != "answer"}
        cache_entry["answer"] = result.get("answer", "")
        self._store[key] = cache_entry
        self._embeddings[key] = q_emb
        self._timestamps[key] = time.time()
        logger.debug("缓存写入 key=%s", key)

    def invalidate(self, answer_id: str = None):
        """使缓存失效。"""
        if answer_id:
            to_del = [k for k, v in self._store.items()
                      if v.get("answer_id") == answer_id]
            for k in to_del:
                del self._store[k]
                self._embeddings.pop(k, None)
                self._timestamps.pop(k, None)
        else:
            self._store.clear()
            self._embeddings.clear()
            self._timestamps.clear()
        logger.info("缓存已清除")

    @property
    def stats(self) -> dict:
        """缓存统计。"""
        return {
            "size": len(self._store),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / (self.hits + self.misses + 1e-10), 4),
            "threshold": self.threshold,
            "ttl": self.ttl,
        }
