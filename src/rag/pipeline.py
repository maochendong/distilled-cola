"""RAG Pipeline — 四步分析流程的端到端问答流水线。

流程：
  1. 识别关键变量（从用户问题提取核心因子）
  2. 调用历史框架（混合检索知识 + 推理链）
  3. 代入当前数据（静态知识 + 实时行情）
  4. 给出个性化建议（生成 + 自检 + 溯源）
"""

from __future__ import annotations

from src.config import config
from src.rag.generator import Generator
from src.rag.reasoning import ReasoningValidator, format_sources
from src.rag.retriever import Retriever
from src.rag.web_search import web_search, format_web_results


class RAGPipeline:
    """端到端的 RAG 问答流水线（上海房产版）。"""

    def __init__(self) -> None:
        self.retriever = Retriever()
        self.generator = Generator()
        self.validator = ReasoningValidator()
        self._web = web_search()

    def _format_context(self, hits: list[dict]) -> str:
        """从已检索的结果中格式化上下文文本。"""
        parts = []
        for i, h in enumerate(hits, 1):
            # 知识块和推理链有不同的结构
            if "metadata" in h and h["metadata"]:
                source = h["metadata"].get("source", h["metadata"].get("title", ""))
                tags = []
                if h["metadata"].get("logic_tags"):
                    tags.append(f"逻辑: {h['metadata']['logic_tags']}")
                if h["metadata"].get("areas"):
                    tags.append(f"板块: {h['metadata']['areas']}")
                tag_info = f"  [{', '.join(tags)}]" if tags else ""
            elif "trigger" in h:
                source = f"推理链: {h['trigger'][:60]}"
                tag_info = ""
            else:
                source = ""
                tag_info = ""
            parts.append(f"[{i}] (来源: {source}){tag_info}\n{h['text']}")
        return "\n\n---\n\n".join(parts) if parts else ""

    def _get_reasoning_chains(self, hits: list[dict]) -> str:
        """从已检索的结果中提取推理链文本。"""
        chains = [h for h in hits if "trigger" in h]
        parts = []
        for i, h in enumerate(chains, 1):
            parts.append(f"### 推理链 {i}\n{h['text']}")
        return "\n\n".join(parts) if parts else ""

    def _needs_real_time(self, query: str) -> bool:
        """判断是否需要查询实时数据。

        包含成交量、价格、政策、挂牌等动态指标的查询触发实时搜索。
        同时覆盖常见的房产问题模式（值得买、对比、分析等）。
        """
        keywords = [
            "成交量", "成交价", "挂牌", "均价", "走势", "行情",
            "最新", "2025", "2026", "2027",
            "政策", "新政", "利率", "限购", "贷款", "首付",
            "涨", "跌", "涨幅", "跌幅", "环比", "同比",
            "多少", "价格", "多少钱", "预算",
            "值得买", "值得", "怎么样", "怎么看", "分析",
            "对比", "vs", "怎么选", "推荐", "建议",
            "数据", "成交", "网签", "交易量", "套数",
        ]
        return any(kw in query for kw in keywords)

    def ask(self, query: str, top_k: int | None = None) -> dict:
        """完整问答流程：检索 → 实时搜索 → 生成 → 自检。

        Args:
            query: 用户问题（如「800万预算前滩vs大宁怎么选？」）
            top_k: 检索的知识块数量

        Returns:
            {
                "query": 原始问题,
                "answer": 四步结构分析,
                "sources": 参考来源列表,
                "confidence": 置信度评分,
                "reasoning_chains_used": 使用的推理链数,
                "web_search_used": 是否使用了实时搜索,
            }
        """
        k = top_k or config.top_k

        # 静态 RAG 检索（博主知识 + 推理链）
        self.retriever.ensure_bm25_index()
        hits = self.retriever.retrieve(query, top_k=k)
        context = self._format_context(hits)
        reasoning_chains = self._get_reasoning_chains(hits)

        # 实时网络搜索（当查询涉及动态数据时自动触发）
        web_context = ""
        web_sources: list[dict] = []
        if self._needs_real_time(query) and self._web.available:
            try:
                # 强制限定上海 + 房产领域，避免结果漂移到其他城市
                sh_query = query if "上海" in query else f"上海 {query}"
                web_results = self._web.search(
                    sh_query, max_results=5,
                    domain="home", zone="cn",
                    freshness="year",
                )
                # 如果领域搜索无结果，降级为通用搜索
                if not web_results:
                    web_results = self._web.search(
                        sh_query, max_results=5,
                        zone="cn", freshness="month",
                    )
                if web_results:
                    web_context = format_web_results(web_results)
                    web_sources = [
                        {"source": r.url, "title": r.title, "snippet": r.snippet[:120]}
                        for r in web_results
                    ]
            except Exception:
                pass  # 网络搜索失败不影响主流程

        # 生成
        answer = self.generator.generate(
            query,
            context=context,
            reasoning_chains=reasoning_chains,
            web_context=web_context,
        )

        # 来源合并（静态 + 实时）
        sources = [
            {
                "id": h.get("id", ""),
                "source": (
                    h["metadata"].get("source", "")
                    if "metadata" in h and h["metadata"]
                    else h.get("trigger", "")[:60]
                ),
                "score": round(float(h.get("score", 0)), 4),
                "snippet": h.get("text", "")[:120] + "..." if len(h.get("text", "")) > 120 else h.get("text", ""),
                "type": "static",
            }
            for h in hits
        ] + [
            {**s, "type": "web", "score": 1.0}
            for s in web_sources
        ]

        validation = self.validator.validate(query, answer, sources=sources)

        # 如果质量不达标且存在推理链，重试一次
        if self.validator.needs_refinement(validation) and reasoning_chains:
            answer = self.generator.generate(
                query,
                context=context,
                reasoning_chains=reasoning_chains,
                web_context=web_context,
            )
            validation = self.validator.validate(query, answer, sources=sources)

        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "confidence": validation.get("confidence", 0.5),
            "reasoning_chains_used": len(reasoning_chains.split("推理链")) - 1 if reasoning_chains else 0,
            "web_search_used": bool(web_context),
        }
