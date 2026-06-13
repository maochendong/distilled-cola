"""混合检索器 — BM25 关键词检索 + 语义向量检索，RRF 融合排序 + Cross-encoder 精排。

确保「前滩2024价格走势」「大宁学区房」这类精确查询能同时命中，
且精排后 top-k 的 query-doc 相关度更高，间接提升生成质量与置信度。
"""

from __future__ import annotations

import math
import re

from src.config import config
from src.knowledge_base.embedder import Embedder
from src.knowledge_base.reasoning_index import ReasoningIndex
from src.knowledge_base.reranker import Reranker
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
        self.reranker = Reranker()
        self._bm25: BM25Okapi | None = None
        self._bm25_corpus: list[dict] | None = None
        self._known_areas: list[str] | None = None
        self._embed_cache: dict[str, list[float]] = {}
        self._embed_available: bool | None = None  # checked lazily

    def build_bm25_index(self) -> None:
        """从知识索引构建 BM25 倒排索引。"""
        from rank_bm25 import BM25Okapi

        if self._bm25 is not None:
            return  # already built
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

    def _get_embedding(self, query: str) -> list[float] | None:
        """获取查询向量，带缓存。嵌入不可用时返回 None。"""
        if query in self._embed_cache:
            return self._embed_cache[query]
        if self._embed_available is None:
            self._embed_available = self.embedder.is_available()
        if not self._embed_available:
            return None
        emb = self.embedder.embed_single(query)
        self._embed_cache[query] = emb
        return emb

    def _semantic_search(self, query: str, top_k: int) -> list[dict]:
        """语义向量检索。"""
        query_emb = self._get_embedding(query)
        if query_emb is None:
            return []
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

    def _extract_query_entities(self, query: str) -> dict:
        """从查询中提取板块和逻辑标签，用于推理链排序加分。

        策略：
        1. 完整子串匹配已知板块名（最长优先，避免短名误匹配长名）
        2. 同板块扩展：如果匹配板块是另一已知板块的子串，自动扩展
           （"前滩"→"前滩南""前滩九宫格"等）
        3. 逻辑标签用关键词规则映射（"vs""对比"→"板块对比"）
        """
        # 惰性加载已知板块列表
        if self._known_areas is None:
            all_data = self.knowledge_index.collection.get(include=["metadatas"])
            areas = set()
            for m in all_data["metadatas"]:
                if m.get("areas"):
                    for a in m["areas"].split(","):
                        a = a.strip()
                        if len(a) >= 2:
                            areas.add(a)
            self._known_areas = sorted(areas, key=lambda x: (-len(x), x))

        matched_areas = []
        remaining = query

        # 板块匹配：最长子串优先（精确完整匹配，不用前缀截断）
        for area in self._known_areas:
            if area in remaining:
                matched_areas.append(area)
                remaining = remaining.replace(area, "", 1)

        # 同板块扩展：如果匹配板块名是另一已知板块的子串，自动包含
        # 例如 "前滩" → "前滩南"、"前滩九宫格"、"前滩太古里"
        expanded = set(matched_areas)
        for base in matched_areas:
            related = [a for a in self._known_areas
                       if base in a and a != base]
            expanded.update(related[:3])

        # 逻辑标签：关键词规则映射（不用子串匹配，用户不会说"板块对比"）
        logic_rules = [
            ("板块对比", ["vs", "对比", "怎么选", "还是", "比较"]),
            ("学区分析", ["学区", "学校", "教育", "上学"]),
            ("时机判断", ["时机", "时候", "现在", "还能", "抄底", "高位"]),
            ("政策解读", ["政策", "新规", "调控", "贷款", "利率", "认房不认贷", "限购"]),
            ("供需分析", ["供应", "供需", "库存", "去化", "挂牌"]),
            ("倒挂判断", ["倒挂", "划算", "溢价"]),
            ("风险提示", ["风险", "危险", "谨慎"]),
            ("流动性评估", ["流动性", "出手", "成交", "流通", "变现"]),
            ("规划利好", ["规划", "利好", "发展", "潜力"]),
        ]
        matched_logic = []
        for tag, keywords in logic_rules:
            if any(kw in query for kw in keywords):
                matched_logic.append(tag)

        return {"areas": list(expanded)[:4], "logic_tags": matched_logic[:2]}

    def _reasoning_search(self, query: str, top_k: int) -> list[dict]:
        """推理链检索（元数据排序加分）。

        提取查询中的板块/逻辑标签后，对候选链做排序加分而非硬过滤：
        - 匹配板块的链 score +0.15
        - 匹配逻辑标签的链 score +0.10
        - 多项匹配可叠加，上限 +0.30
        保证不丢任何候选，同时让语义相关 + 标签匹配的链排到前面。
        """
        entities = self._extract_query_entities(query)
        query_emb = self._get_embedding(query)
        if query_emb is None:
            return []

        # 拉较大候选集供排序加分
        candidates = self.reasoning_index.search(query_emb, top_k=50)

        matched_areas = entities.get("areas", [])
        matched_logic = entities.get("logic_tags", [])

        if not matched_areas and not matched_logic:
            # 无实体可匹配，直接返回 top-k
            return candidates[:top_k]

        # 排序加分
        for c in candidates:
            boost = 0.0
            chain_areas = c.get("areas", "") or ""
            chain_tags = c.get("logic_tags", "") or ""

            # 板块匹配加分
            if matched_areas and any(a and a in chain_areas for a in matched_areas):
                boost += 0.15

            # 逻辑标签匹配加分
            if matched_logic and any(t and t in chain_tags for t in matched_logic):
                boost += 0.10

            c["score"] = c.get("score", 0.0) + boost

        # 按加分后排序列取 top-k
        candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return candidates[:top_k]

    def _structured_search(self, query: str) -> list[dict]:
        """结构化数据查询（SQLite 楼盘/价格/学区数据）。"""
        try:
            from src.data.db import search_district, get_district_stats, find_listings
        except ImportError:
            return []

        # 从查询中提取板块名
        entities = self._extract_query_entities(query)
        matched_areas = entities.get("areas", [])
        if not matched_areas:
            return []

        hits = []
        for area in matched_areas:
            # 板块统计
            stats = get_district_stats(area)
            if stats and stats.get("avg_unit_price"):
                hits.append({
                    "id": f"struct_{area}",
                    "text": (
                        f"【{area}板块实时数据】均价 {stats['avg_unit_price']:.0f} 元/㎡, "
                        f"平均总价 {stats['avg_total_price']:.0f} 万元, "
                        f"在售 {stats['listing_count']} 套, "
                        f"户型面积 {stats['avg_size']:.0f} ㎡, "
                        f"价格区间 {stats['min_price']:.0f}-{stats['max_price']:.0f} 万元"
                    ),
                    "metadata": {
                        "source": f"structured_data/{area}",
                        "areas": area,
                        "logic_tags": "行情数据",
                    },
                    "score": 0.5 + (stats["listing_count"] or 0) * 0.01,
                    "type": "structured",
                })

            # 具体房源
            listings = find_listings(district_name=area, limit=5)
            if listings:
                lines = [f"【{area}板块在售房源】"]
                for l in listings:
                    lines.append(
                        f"- {l['property']} {l['layout']} {l['size_sqm']:.0f}㎡ "
                        f"{l['total_price']:.0f}万 ({l['unit_price']:.0f}元/㎡)"
                    )
                hits.append({
                    "id": f"listings_{area}",
                    "text": "\n".join(lines),
                    "metadata": {
                        "source": f"structured_data/{area}",
                        "areas": area,
                        "logic_tags": "在售房源",
                    },
                    "score": 0.4,
                    "type": "structured",
                })

        return hits

    def retrieve(
        self, query: str, top_k: int | None = None,
        include_reasoning: bool = True,
    ) -> list[dict]:
        """混合检索：语义 + BM25 + 推理链 + 结构化数据，RRF 融合。

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

        # RRF 融合后取 top_k * 4 作为精排候选池
        fused = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)[:k * 4]

        # Cross-encoder 精排: 对候选池 (query, doc) pair 重新打分
        reranked = self.reranker.rerank(query, fused, top_k=k)

        # 追加推理链
        if include_reasoning:
            reasoning_hits = self._reasoning_search(query, max(1, k // 2))
            if reasoning_hits:
                reranked.extend(reasoning_hits)

        # 追加结构化数据（板块价格/房源）
        structured_hits = self._structured_search(query)
        if structured_hits:
            reranked.extend(structured_hits)

        return reranked

    def retrieve_knowledge_only(self, query: str, top_k: int | None = None) -> list[dict]:
        """仅检索知识（不包含推理链）。"""
        return self.retrieve(query, top_k=top_k, include_reasoning=False)
