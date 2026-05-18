"""RAG Pipeline — 四步分析流程的端到端问答流水线。

流程：
  1. 识别关键变量（从用户问题提取核心因子）
  2. 调用历史框架（混合检索知识 + 推理链）
  3. 代入当前数据（组装上下文）
  4. 给出个性化建议（生成 + 自检 + 溯源）
"""

from __future__ import annotations

from src.config import config
from src.rag.generator import Generator
from src.rag.reasoning import ReasoningValidator, format_sources
from src.rag.retriever import Retriever


class RAGPipeline:
    """端到端的 RAG 问答流水线（上海房产版）。"""

    def __init__(self) -> None:
        self.retriever = Retriever()
        self.generator = Generator()
        self.validator = ReasoningValidator()

    def ask(self, query: str, top_k: int | None = None) -> dict:
        """完整问答流程：检索 → 生成 → 自检。

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
            }
        """
        k = top_k or config.top_k

        # Step 1+2: 混合检索知识 + 推理链
        self.retriever.ensure_bm25_index()
        hits = self.retriever.retrieve(query, top_k=k)
        context = self.retriever.get_context(query, top_k=k)
        reasoning_chains = self.retriever.get_reasoning_chains(query, top_k=max(1, k // 2))

        # Step 3+4: 生成 + 自检
        answer = self.generator.generate(query, context=context, reasoning_chains=reasoning_chains)

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
            }
            for h in hits
        ]

        validation = self.validator.validate(query, answer, sources=sources)

        # 如果质量不达标且存在推理链，重试一次
        if self.validator.needs_refinement(validation) and reasoning_chains:
            answer = self.generator.generate(query, context=context, reasoning_chains=reasoning_chains)
            validation = self.validator.validate(query, answer, sources=sources)

        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "confidence": validation.get("confidence", 0.5),
            "reasoning_chains_used": len(reasoning_chains.split("推理链")) - 1 if reasoning_chains else 0,
        }
