"""Cross-encoder 重排序器 — 对混合检索结果做精排，提升 top-k 精度。

使用 BGE Reranker v2 系列模型（支持中文），在语义+BM25 粗排基础上做 pair-wise 精排。
输出 rerank_score 覆盖原有的 RRF score，确保最相关的知识片段排在前面。

模型选择（按优先级）:
  1. BAAI/bge-reranker-v2-m3 — 推荐，多语言，精度最高（需 FlagEmbedding）
  2. BAAI/bge-reranker-base  — 退路，与 sentence-transformers 原生兼容
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder 精排器，对 (query, doc) pair 打分后重排。

    延迟初始化模型，首次调用 rerank() 时自动加载可用后端。
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        self.model_name = model_name
        self._model: Any = None
        self._backend: str | None = None  # "flag" | "ce" | None

    def rerank(
        self, query: str, candidates: list[dict], top_k: int = 5,
    ) -> list[dict]:
        """对候选项进行 cross-encoder 重排。

        Args:
            query: 用户原始查询
            candidates: 待重排的候选列表（每项含 text 字段）
            top_k: 返回 top-k 条

        Returns:
            按 cross-encoder 分数降序排列的结果（每项增加 rerank_score）
        """
        if not candidates:
            return []
        if not self._lazy_init():
            # 无可用精排模型，直接按原有 score 截断返回
            logger.warning("无可用 cross-encoder 模型，跳过精排")
            for c in candidates:
                c["rerank_score"] = c.get("score", 0.0)
            return candidates[:top_k]

        pairs = [(query, c.get("text", "")) for c in candidates]

        try:
            if self._backend == "flag":
                scores = [self._model.compute_score(p) for p in pairs]
            elif self._backend == "ce":
                scores = self._model.predict(pairs).tolist()  # type: ignore[union-attr]
            else:
                scores = [c.get("score", 0.0) for c in candidates]
        except Exception as e:
            logger.warning("Cross-encoder 打分失败 (%s)，使用原始排序", e)
            for c in candidates:
                c["rerank_score"] = c.get("score", 0.0)
            return candidates[:top_k]

        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s) if isinstance(s, (int, float)) else float(s[0] if isinstance(s, (list, tuple)) else 0.0)

        candidates.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        return candidates[:top_k]

    def _lazy_init(self) -> bool:
        if self._model is not None:
            return True

        # 确保从可访问的 HuggingFace 源下载
        os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")

        # 快速检查模型文件是否已缓存。未缓存时不阻塞下载，直接跳过精排。
        if not self._model_cached():
            logger.info("精排模型未缓存 (%s)，跳过精排。需要时运行: python -c "
                        "\"from huggingface_hub import snapshot_download; "
                        "snapshot_download('%s')\"", self.model_name, self.model_name)
            self._model = None
            self._backend = None
            return False

        # 1) FlagReranker (BGE v2 m3 官方接口)
        try:
            from FlagEmbedding import FlagReranker  # type: ignore[import-untyped]
            self._model = FlagReranker(self.model_name, use_fp16=True)
            self._backend = "flag"
            logger.info("精排后端: FlagReranker(%s)", self.model_name)
            return True
        except ImportError:
            pass
        except Exception as e:
            logger.warning("FlagReranker 加载失败: %s，尝试退路", e)

        # 2) sentence-transformers CrossEncoder (退路)
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, max_length=512)
            self._backend = "ce"
            logger.info("精排后端: CrossEncoder(%s)", self.model_name)
            return True
        except Exception as e:
            logger.warning("CrossEncoder 加载失败: %s", e)

        self._model = None
        self._backend = None
        return False

    @staticmethod
    def _model_cached(model_name: str = "BAAI/bge-reranker-v2-m3") -> bool:
        """检查模型文件是否已在 HF 缓存中。

        先看 safetensors/bin 权重文件，再看 blobs 目录中是否有大文件。
        BGE Reranker v2-m3 使用 PyTorch 格式，权重存储在 blobs 中。
        """
        import os
        from pathlib import Path
        cache_dir = Path(os.path.expanduser(
            os.environ.get("HF_HOME", "~/.cache/huggingface/hub")
        ))
        model_dir = cache_dir / f"models--{model_name.replace('/', '--')}"
        if not model_dir.exists():
            return False
        # 检查是否有 safetensors 或 bin 权重文件
        for pattern in ("model.safetensors", "*.bin"):
            for f in model_dir.rglob(pattern):
                if f.stat().st_size > 1_000_000:
                    return True
        # 检查 blobs 目录中是否有大文件（PyTorch 格式缓存）
        blobs_dir = model_dir / "blobs"
        if blobs_dir.exists():
            total = sum(f.stat().st_size for f in blobs_dir.iterdir() if f.is_file())
            if total > 50_000_000:  # >50MB
                return True
        return False
