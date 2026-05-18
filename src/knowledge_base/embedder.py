"""文本嵌入模块 — 将文本转为向量表示。

默认使用本地 sentence-transformers 模型（BGE-M3 多语言）。
若设置了 OPENAI_API_KEY，也可用 OpenAI Embedding API。
注意：DeepSeek 不提供 Embedding API。
"""

from __future__ import annotations

from src.config import config


EMBED_DIM = 1024  # bge-m3 输出维度


class Embedder:
    """文本嵌入生成器，优先 OpenAI API，退路使用 bge-m3 本地模型。"""

    def __init__(self) -> None:
        self._local_model = None
        self._openai_client = None
        self._available: bool | None = None  # cache availability after first check

    def is_available(self) -> bool:
        """检查是否有可用的嵌入后端。"""
        if self._available is not None:
            return self._available
        if config.openai_api_key:
            self._available = True
            return True
        try:
            self._check_local_model()
            self._available = True
        except (ImportError, Exception):
            self._available = False
        return self._available

    def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        """批量生成文本向量。"""
        if config.openai_api_key:
            return self._embed_openai(texts, model or config.embedding_model)
        # 否则用本地模型
        self._check_local_model()
        return self._embed_local(texts)

    def embed_single(self, text: str) -> list[float]:
        """生成单段文本向量。"""
        return self.embed([text])[0]

    def _check_local_model(self) -> None:
        """检查本地模型是否可加载，惰性初始化。"""
        if self._local_model is not None:
            return
        import os
        if "HF_ENDPOINT" not in os.environ:
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
        self._local_model = SentenceTransformer(
            "BAAI/bge-m3",
            device="cpu",
            model_kwargs={"low_cpu_mem_usage": False},
        )

    def _embed_openai(self, texts: list[str], model: str) -> list[list[float]]:
        """使用 OpenAI Embedding API。"""
        if not self._openai_client:
            from openai import OpenAI
            self._openai_client = OpenAI(api_key=config.openai_api_key)
        resp = self._openai_client.embeddings.create(input=texts, model=model)
        return [d.embedding for d in resp.data]

    def _embed_local(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """使用本地 sentence-transformers 模型，分 batch 避免 OOM。"""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_emb = self._local_model.encode(batch, normalize_embeddings=True).tolist()
            all_embeddings.extend(batch_emb)
            if len(texts) > batch_size:
                print(f"    嵌入进度: {min(i+batch_size, len(texts))}/{len(texts)}")
        return all_embeddings  # type: ignore[no-any-return]
