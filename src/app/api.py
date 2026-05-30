"""FastAPI 服务 — 上海房产分析专家 RESTful API。"""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel

from src.rag.pipeline import RAGPipeline

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="蒸馏小可乐 API",
    description="上海房产分析专家 — 基于博主知识蒸馏的四步分析系统",
    version="0.1.0",
)

# 挂载静态目录，提供 logo 等静态资源
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

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
    web_search_used: bool = False


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Web UI 首页。"""
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return "<h1>蒸馏小可乐</h1><p>页面构建中...</p>"


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": "0.1.0", "domain": "上海房产分析"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> dict:
    """向蒸馏后的上海房产专家提问。返回四步结构分析。"""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")
    t0 = time.time()
    result = pipe.ask(req.query, top_k=req.top_k)
    elapsed = time.time() - t0
    logger.info(
        "📝 ask | query={} | confidence={:.2f} | chains={} | web={} | "
        "sources={} | answer_len={} |耗时={:.1f}s".format(
            req.query[:50],
            result.get("confidence", 0),
            result.get("reasoning_chains_used", 0),
            result.get("web_search_used", False),
            len(result.get("sources", [])),
            len(result.get("answer", "")),
            elapsed,
        )
    )
    return result


@app.get("/stats")
def stats() -> dict:
    """知识库统计。"""
    from src.knowledge_base.vector_store import KnowledgeIndex
    from src.knowledge_base.reasoning_index import ReasoningIndex
    kb = KnowledgeIndex()
    ri = ReasoningIndex()
    return {"knowledge": kb.stats(), "reasoning_chains": ri.stats()}
