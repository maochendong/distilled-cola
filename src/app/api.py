"""FastAPI 服务 — 上海房产分析专家 RESTful API。"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.rag.pipeline import RAGPipeline

app = FastAPI(
    title="蒸馏小可乐 API",
    description="上海房产分析专家 — 基于博主知识蒸馏的四步分析系统",
    version="0.1.0",
)

pipe = RAGPipeline()


class AskRequest(BaseModel):
    query: str
    top_k: int = 5


class AskResponse(BaseModel):
    query: str
    answer: str
    sources: list[dict]
    confidence: float
    reasoning_chains_used: int


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": "0.1.0", "domain": "上海房产分析"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> dict:
    """向蒸馏后的上海房产专家提问。返回四步结构分析。"""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")
    return pipe.ask(req.query, top_k=req.top_k)


@app.get("/stats")
def stats() -> dict:
    """知识库统计。"""
    from src.knowledge_base.vector_store import KnowledgeIndex
    from src.knowledge_base.reasoning_index import ReasoningIndex
    kb = KnowledgeIndex()
    ri = ReasoningIndex()
    return {"knowledge": kb.stats(), "reasoning_chains": ri.stats()}
