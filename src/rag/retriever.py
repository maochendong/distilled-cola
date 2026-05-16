"""检索器 — 封装混合检索和推理链检索。"""

from __future__ import annotations

from src.config import config
from src.knowledge_base.hybrid_retriever import HybridRetriever


class Retriever:
    """检索器：将用户问题 → 混合检索 → 上下文文本。"""

    def __init__(self) -> None:
        self.hybrid = HybridRetriever()

    def ensure_bm25_index(self) -> None:
        """确保 BM25 索引已构建。"""
        self.hybrid.build_bm25_index()

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """混合检索知识 + 推理链。"""
        return self.hybrid.retrieve(query, top_k=top_k)

    def get_context(self, query: str, top_k: int | None = None) -> str:
        """返回拼接后的知识上下文。"""
        hits = self.retrieve(query, top_k=top_k)
        parts = []
        for i, h in enumerate(hits, 1):
            source = ""
            if "metadata" in h and h["metadata"]:
                source = h["metadata"].get("source", h["metadata"].get("title", ""))
            elif "trigger" in h:
                source = f"推理链: {h['trigger'][:60]}"

            tag_info = ""
            if "metadata" in h and h["metadata"]:
                tags = []
                if h["metadata"].get("logic_tags"):
                    tags.append(f"逻辑: {h['metadata']['logic_tags']}")
                if h["metadata"].get("areas"):
                    tags.append(f"板块: {h['metadata']['areas']}")
                if tags:
                    tag_info = f"  [{', '.join(tags)}]"

            parts.append(f"[{i}] (来源: {source}){tag_info}\n{h['text']}")

        return "\n\n---\n\n".join(parts) if parts else ""

    def get_reasoning_chains(self, query: str, top_k: int = 3) -> str:
        """返回推理链文本，作为 few-shot 思维范例。"""
        hits = self.hybrid._reasoning_search(query, top_k=top_k)
        parts = []
        for i, h in enumerate(hits, 1):
            parts.append(f"### 推理链 {i}\n{h['text']}")
        return "\n\n".join(parts) if parts else ""
