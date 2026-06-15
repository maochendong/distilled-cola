"""RAG Pipeline — 四步分析流程的端到端问答流水线 (v2.1)

流程：
  1. 识别关键变量（从用户问题提取核心因子）
  2. 调用历史框架（混合检索知识 + 推理链）
  3. 代入当前数据（静态知识 + 实时行情）
  4. 给出个性化建议（生成 + 自检 + 溯源）

v2.1: +多轮对话 +流式输出 +指数退避自检 +非阻塞精排
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Generator

from src.config import config
from src.rag.generator import Generator
from src.rag.retriever import Retriever
from src.rag.web_search import web_search, format_web_results
from src.rag.profile_router import ProfileRouter

logger = logging.getLogger(__name__)


class RAGPipeline:
    """端到端的 RAG 问答流水线（上海房产版）。"""

    def __init__(self, retriever=None, generator=None,
                 validator=None, reranker=None,
                 conversation_manager=None):
        self.retriever = retriever or Retriever()
        self.generator = generator or Generator()
        # 验证器 — 指数退避 + 可配置阈值
        from src.rag.validator import ReasoningValidator
        self.validator = validator or ReasoningValidator(
            client=self.generator.client,
            teacher_model=getattr(config, 'teacher_model', 'deepseek-v4-pro'),
        )
        # 非阻塞精排
        from src.rag.reranker import Reranker
        self.reranker = reranker or Reranker()
        # 实时搜索
        self._web = web_search()
        # 多轮对话
        from src.rag.conversation import conversation_manager as cm
        self.conversation_manager = conversation_manager or cm
        # 语义缓存
        from src.rag.cache import SemanticCache
        self.cache = SemanticCache()
        self._answer_counter = 0

    def _next_answer_id(self) -> str:
        self._answer_counter += 1
        return f"ans-{int(time.time())}-{self._answer_counter}"

    # ── 格式化 ──

    def _format_context(self, hits: list[dict]) -> str:
        """从已检索的结果中格式化上下文文本。"""
        parts = []
        for i, h in enumerate(hits, 1):
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

    # ── 实时搜索判断 ──

    def _needs_real_time(self, query: str) -> bool:
        """判断是否需要查询实时数据。"""
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

    def _inject_search_warning(self, hits: list) -> str:
        if not hits:
            return ("\n\n> ⚠️ **检索提示**：未在知识库中找到直接相关的信息。"
                    "以下回答基于模型自身知识，可能不够准确，请谨慎参考。\n")
        return ""

    # ── 核心方法 ──

    def ask(self, query: str, top_k: Optional[int] = None,
            conv_id: Optional[str] = None,
            system_prompt: Optional[str] = None,
            response_format: Optional[dict] = None,
            mode: str = "detailed",
            profile_id: Optional[str] = None) -> dict:
        """完整问答流程：检索 → 实时搜索 → 生成 → 自检。"""
        k = top_k or config.top_k
        answer_id = self._next_answer_id()

        # 语义缓存检查
        cached = self.cache.get(query)
        if cached:
            cached["cached"] = True
            return cached

        # 用户画像路由 (T-015)
        if profile_id:
            query = ProfileRouter.route_query(query, profile_id)

        # 无 conv_id 时自动创建会话，确保多轮对话生效
        if self.conversation_manager and not conv_id:
            conv = self.conversation_manager.create()
            conv_id = conv.id

        # 对话上下文 (T-001)
        conversation_context = ""
        if self.conversation_manager and conv_id:
            conversation_context = self.conversation_manager.get_context(conv_id)

        # 静态 RAG 检索（博主知识 + 推理链）
        self.retriever.ensure_bm25_index()
        hits = self.retriever.retrieve(query, top_k=k)

        # 精排 (T-006)
        if self.reranker and hits:
            try:
                hits = self.reranker.rerank(query, hits, top_k=k)
            except Exception as e:
                logger.warning("精排失败: %s", e)

        context = self._format_context(hits)
        reasoning_chains = self._get_reasoning_chains(hits)

        # 实时网络搜索（当查询涉及动态数据时自动触发）
        web_context = ""
        web_sources: list[dict] = []
        if self._needs_real_time(query) and self._web.available:
            try:
                sh_query = query if "上海" in query else f"上海 {query}"
                results = self._web.search(
                    sh_query, max_results=5,
                    domain="home", zone="cn", freshness="year",
                )
                # 领域搜索无结果时降级为通用搜索
                if not results:
                    results = self._web.search(
                        sh_query, max_results=5,
                        zone="cn", freshness="month",
                    )
                if results:
                    web_context = format_web_results(results)
                    web_sources = [
                        {"source": r.url, "title": r.title, "snippet": r.snippet[:120]}
                        for r in results
                    ]
            except Exception:
                pass

        # 生成
        answer = self.generator.generate(
            query,
            context=context,
            reasoning_chains=reasoning_chains,
            system_prompt=system_prompt,
            conversation_context=conversation_context,
            web_context=web_context,
            mode=mode,
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

        # 自检未通过则重试一次 (T-004)
        if self.validator.needs_refinement(validation) and reasoning_chains:
            answer = self.generator.generate(
                query,
                context=context,
                reasoning_chains=reasoning_chains,
                system_prompt=system_prompt,
                conversation_context=conversation_context,
                web_context=web_context,
                mode=mode,
            )
            validation = self.validator.validate(query, answer, sources=sources)

        search_warning = self._inject_search_warning(hits)
        if search_warning:
            answer += search_warning

        # 保存对话 (T-001)
        if self.conversation_manager and conv_id:
            self.conversation_manager.add_message(conv_id, "user", query)
            self.conversation_manager.add_message(conv_id, "assistant", answer)

        result = {
            "answer_id": answer_id,
            "query": query,
            "answer": answer,
            "sources": sources,
            "confidence": getattr(validation, 'confidence', 0.5),
            "failed_checks": getattr(validation, 'failed_checks', []),
            "reasoning_chains_used": len([h for h in hits if "trigger" in h]),
            "web_search_used": bool(web_context),
            "conv_id": conv_id,
        }

        # 写入语义缓存
        self.cache.set(query, result)

        return result

    def ask_stream(self, query: str, top_k: Optional[int] = None,
                   conv_id: Optional[str] = None,
                   system_prompt: Optional[str] = None,
                   mode: str = "detailed",
                   profile_id: Optional[str] = None) -> Generator[dict, None, None]:
        """流式问答 (T-002) — SSE 逐 token 输出"""
        k = top_k or config.top_k

        # 用户画像路由 (T-015)
        if profile_id:
            query = ProfileRouter.route_query(query, profile_id)

        # 语义缓存检查
        cached = self.cache.get(query)
        if cached:
            yield {"type": "token", "content": cached.get("answer", "")}
            yield {
                "type": "done",
                "answer_id": self._next_answer_id(),
                "confidence": cached.get("confidence", 1.0),
                "sources": cached.get("sources", []),
                "reasoning_chains_used": cached.get("reasoning_chains_used", 0),
                "web_search_used": cached.get("web_search_used", False),
                "conv_id": conv_id,
                "cached": True,
            }
            return

        # 无 conv_id 时自动创建会话，确保多轮对话生效
        if self.conversation_manager and not conv_id:
            conv = self.conversation_manager.create()
            conv_id = conv.id

        conversation_context = ""
        if self.conversation_manager and conv_id:
            conversation_context = self.conversation_manager.get_context(conv_id)

        # 检索
        self.retriever.ensure_bm25_index()
        hits = self.retriever.retrieve(query, top_k=k)

        # 精排
        if self.reranker and hits:
            try:
                hits = self.reranker.rerank(query, hits, top_k=k)
            except Exception as e:
                logger.warning("精排失败: %s", e)

        context = self._format_context(hits)
        reasoning_chains = self._get_reasoning_chains(hits)

        # 实时搜索
        web_context = ""
        if self._needs_real_time(query) and self._web.available:
            try:
                sh_query = query if "上海" in query else f"上海 {query}"
                results = self._web.search(
                    sh_query, max_results=5,
                    domain="home", zone="cn", freshness="year",
                )
                if not results:
                    results = self._web.search(
                        sh_query, max_results=5,
                        zone="cn", freshness="month",
                    )
                if results:
                    web_context = format_web_results(results)
            except Exception:
                pass

        search_warning = self._inject_search_warning(hits)

        # 流式生成
        full = ""
        try:
            for chunk in self.generator.generate_stream(
                query=query, context=context,
                reasoning_chains=reasoning_chains,
                system_prompt=system_prompt,
                conversation_context=conversation_context,
                web_context=web_context,
                mode=mode,
            ):
                full += chunk
                yield {"type": "token", "content": chunk}
        except Exception as e:
            logger.error("流式生成失败: %s", e)
            yield {"type": "error", "content": f"⚠️ 生成错误: {e}"}

        if search_warning:
            yield {"type": "warning", "content": search_warning}

        # 验证
        v = self.validator.validate(query, full)
        confidence = v.confidence if self.validator else 1.0

        # 构建来源列表 (与 ask() 一致)
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
        ]
        reasoning_chains_used = len([h for h in hits if "trigger" in h])

        if self.conversation_manager and conv_id:
            self.conversation_manager.add_message(conv_id, "user", query)
            self.conversation_manager.add_message(conv_id, "assistant", full)

        yield {
            "type": "done",
            "answer_id": self._next_answer_id(),
            "confidence": confidence,
            "conv_id": conv_id,
            "sources": sources,
            "reasoning_chains_used": reasoning_chains_used,
            "web_search_used": bool(web_context),
        }
