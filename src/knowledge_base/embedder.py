"""文本嵌入模块 — 将文本转为向量表示。

默认使用本地 sentence-transformers 模型（BGE-M3 多语言）。
若设置了 OPENAI_API_KEY，也可用 OpenAI Embedding API。
注意：DeepSeek 不提供 Embedding API。
"""

from __future__ import annotations

from src.config import config


EMBED_DIM = 1024  # bge-m3 输出维度


class Embedder:
    """文本嵌入生成器，优先本地模型，OpenAI 为备选。"""

    def __init__(self) -> None:
        self._local_model = None
        self._openai_client = None

    def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        """批量生成文本向量。"""
        # 如果有 OpenAI key，优先用 OpenAI（质量更高）
        if config.openai_api_key:
            return self._embed_openai(texts, model or config.embedding_model)
        # 否则用本地模型
        return self._embed_local(texts)

    def embed_single(self, text: str) -> list[float]:
        """生成单段文本向量。"""
        return self.embed([text])[0]

    def _embed_openai(self, texts: list[str], model: str) -> list[list[float]]:
        """使用 OpenAI Embedding API。"""
        if not self._openai_client:
            from openai import OpenAI
            self._openai_client = OpenAI(api_key=config.openai_api_key)
        resp = self._openai_client.embeddings.create(input=texts, model=model)
        return [d.embedding for d in resp.data]

    def _embed_local(self, texts: list[str]) -> list[list[float]]:
        """使用本地 sentence-transformers 模型。"""
        import os
        if "HF_ENDPOINT" not in os.environ:
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
            if not self._local_model:
                self._local_model = SentenceTransformer("BAAI/bge-m3")
            return self._local_model.encode(texts, normalize_embeddings=True).tolist()  # type: ignore[no-any-return]
        except ImportError:
            msg = (
                "没有可用的嵌入模型。请设置 OPENAI_API_KEY 或安装 sentence-transformers:\n"
                "  pip install sentence-transformers"
            )
            raise RuntimeError(msg) from None
