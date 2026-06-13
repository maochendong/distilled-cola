"""Retriever — 混合检索引擎 (T-005)"""
import logging

logger = logging.getLogger(__name__)


class Retriever:
    """混合检索器 — 语义 + BM25 + RRF，向后兼容旧接口"""

    def __init__(self):
        self.hybrid = None
        self._last_search_status = {"semantic": True, "keyword": True}

    def _ensure_hybrid(self):
        if self.hybrid is None:
            from src.knowledge_base.hybrid_retriever import HybridRetriever
            self.hybrid = HybridRetriever()

    def ensure_bm25_index(self):
        self._ensure_hybrid()
        self.hybrid.build_bm25_index()

    @property
    def last_search_status(self) -> dict:
        return dict(self._last_search_status)

    def retrieve(self, query: str, top_k: int = 5) -> list:
        self._ensure_hybrid()
        try:
            hits = self.hybrid.retrieve(query, top_k=top_k)
            self._last_search_status["semantic"] = True
            self._last_search_status["keyword"] = True
            if not hits:
                self._last_search_status["semantic"] = False
            return hits
        except Exception as e:
            self._last_search_status["semantic"] = False
            self._last_search_status["keyword"] = False
            logger.error("检索失败: %s", e)
            return []
