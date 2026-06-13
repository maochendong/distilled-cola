"""
Cross-encoder Reranker — 精排系统
非阻塞设计：模型不可用时可 graceful degradation + 日志告警
"""
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class Reranker:
    """交叉编码器精排 — 惰性初始化，支持无损降级"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None
        self._backend = None
        self._initialized = False

    def _model_cached(self) -> bool:
        """检查 HuggingFace 模型是否已缓存"""
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        model_path = os.path.join(
            cache_dir, f"models--{self.model_name.replace('/', '--')}"
        )
        return os.path.exists(model_path)

    def _lazy_init(self) -> bool:
        """惰性初始化 — 首次调用时尝试加载模型"""
        if self._model is not None:
            return True
        if self._initialized:
            return False
        self._initialized = True

        # 检查模型是否已缓存
        if not self._model_cached():
            logger.warning(
                "Cross-encoder 模型 %s 未缓存，跳过精排。"
                "如需启用精排请先下载模型。", self.model_name
            )
            return False

        # 1) FlagReranker (官方接口)
        try:
            from FlagEmbedding import FlagReranker as FR
            self._model = FR(self.model_name, use_fp16=True)
            self._backend = "flag"
            logger.info("Reranker 使用 FlagReranker 后端 (FP16)")
            return True
        except ImportError:
            pass

        # 2) sentence-transformers (退路)
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, max_length=512)
            self._backend = "ce"
            logger.info("Reranker 使用 CrossEncoder (sentence-transformers) 后端")
            return True
        except ImportError:
            pass

        logger.warning(
            "Cross-encoder 后端不可用 (FlagReranker/CrossEncoder 均未安装)，"
            "跳过精排阶段，使用 RRF 原始分数排序。"
        )
        return False

    def rerank(self, query: str, candidates: list,
               top_k: int = 5) -> list:
        """对候选结果进行精排"""
        if not candidates:
            if self._lazy_init():
                logger.info("Reranker 可用但无候选项，跳过精排")
            return candidates

        if not self._lazy_init():
            logger.info("Reranker 不可用，跳过精排，使用原始排序")
            for c in candidates:
                c["rerank_score"] = c.get("score", 0.0)
            return candidates[:top_k]

        pairs = [(query, c.get("text", "")) for c in candidates]

        try:
            if self._backend == "flag":
                scores = [self._model.compute_score(p) for p in pairs]
            elif self._backend == "ce":
                scores = self._model.predict(pairs).tolist()
            else:
                scores = [c.get("score", 0.0) for c in candidates]
        except Exception as e:
            logger.warning("Cross-encoder 打分失败 (%s)，使用原始排序", e)
            for c in candidates:
                c["rerank_score"] = c.get("score", 0.0)
            return candidates[:top_k]

        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)

        candidates.sort(key=lambda x: x.get("rerank_score", 0.0),
                        reverse=True)
        return candidates[:top_k]

    @property
    def is_available(self) -> bool:
        """检查 Reranker 是否可用（不触发加载）"""
        return self._model is not None
