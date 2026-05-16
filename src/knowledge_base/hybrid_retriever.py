"""混合检索器 — BM25 关键词检索 + 语义向量检索，RRF 融合排序。

确保「前滩2024价格走势」「大宁学区房」这类精确查询能同时命中。
"""

from __future__ import annotations

import math
import re

from src.config import config
from src.knowledge_base.embedder import Embedder
from src.knowledge_base.reasoning_index import ReasoningIndex
from src.knowledge_base.vector_store import KnowledgeIndex


def tokenize(text: str) -> list[str]:
    """中文分词，支持中英文混合 tokenization。"""
    try:
        import jieba
        return list(jieba.cut(text))
    except ImportError:
        pass
    # fallback: 逐字分割中文 + 英文词组保留
    tokens: list[str] = []
    current_eng = ""
    for ch in text:
        if re.match(r"[a-zA-Z0-9]", ch):
            current_eng += ch.lower()
        else:
            if current_eng:
                tokens.append(current_eng)
                current_eng = ""
            if ch.strip():
                tokens.append(ch)
    if current_eng:
        tokens.append(current_eng)
    return tokens


class HybridRetriever:
    """混合检索器：语义 + BM25 + 推理链检索。"""

    def __init__(self) -> None:
        self.embedder = Embedder()
        self.knowledge_index = KnowledgeIndex()
        self.reasoning_index = ReasoningIndex()
        self._bm25: BM25Okapi | None = None
        self._bm25_corpus: list[dict] | None = None

    def build_bm25_index(self) -> None:
        """从知识索引构建 BM25 倒排索引。"""
        from rank_bm25 import BM25Okapi

        all_data = self.knowledge_index.collection.get(include=["documents", "metadatas"])
        corpus_texts = []
        self._bm25_corpus = []
        for i in range(len(all_data["ids"])):
            corpus_texts.append(tokenize(all_data["documents"][i]))
            self._bm25_corpus.append({
                "id": all_data["ids"][i],
                "text": all_data["documents"][i],
                "metadata": all_data["metadatas"][i],
            })
        if corpus_texts:
            self._bm25 = BM25Okapi(corpus_texts)
            print(f"  🔍 BM25 索引构建完成: {len(corpus_texts)} 篇文档")

    def _semantic_search(self, query: str, top_k: int) -> list[dict]:
        """语义向量检索。"""
        query_emb = self.embedder.embed_single(query)
        return self.knowledge_index.search(query_emb, top_k=top_k)

    def _keyword_search(self, query: str, top_k: int) -> list[dict]:
        """BM25 关键词检索。"""
        if not self._bm25 or not self._bm25_corpus:
            return []
        tokenized = tokenize(query)
        scores = self._bm25.get_scores(tokenized)

        scored = [
            {"id": self._bm25_corpus[i]["id"], "text": self._bm25_corpus[i]["text"],
             "metadata": self._bm25_corpus[i]["metadata"], "bm25_score": scores[i]}
            for i in range(len(scores))
        ]
        scored.sort(key=lambda x: x["bm25_score"], reverse=True)
        return [s for s in scored if s["bm25_score"] > 0][:top_k]

    def _reasoning_search(self, query: str, top_k: int) -> list[dict]:
        """推理链检索。"""
        query_emb = self.embedder.embed_single(query)
        return self.reasoning_index.search(query_emb, top_k=top_k)

    def retrieve(
        self, query: str, top_k: int | None = None,
        include_reasoning: bool = True,
    ) -> list[dict]:
        """混合检索：语义 + BM25 + 推理链，RRF 融合。

        Returns:
            按融合分数降序排列的检索结果
        """
        k = top_k or config.top_k

        # 各检索器结果
        semantic_hits = self._semantic_search(query, k * 2)
        keyword_hits = self._keyword_search(query, k * 2)

        # RRF 融合
        rrf_scores: dict[str, dict] = {}

        for rank, hit in enumerate(semantic_hits):
            doc_id = hit["id"]
            rrf_scores.setdefault(doc_id, {**hit, "score": 0.0})
            rrf_scores[doc_id]["score"] += 1.0 / (60 + rank)  # RRF

        for rank, hit in enumerate(keyword_hits):
            doc_id = hit["id"]
            if doc_id in rrf_scores:
                rrf_scores[doc_id]["score"] += 1.0 / (60 + rank)
            else:
                rrf_scores[doc_id] = {**hit, "score": 1.0 / (60 + rank)}

        fused = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)[:k]

        # 追加推理链
        if include_reasoning:
            reasoning_hits = self._reasoning_search(query, max(1, k // 2))
            if reasoning_hits:
                fused.extend(reasoning_hits)

        return fused

    def retrieve_knowledge_only(self, query: str, top_k: int | None = None) -> list[dict]:
        """仅检索知识（不包含推理链）。"""
        return self.retrieve(query, top_k=top_k, include_reasoning=False)
