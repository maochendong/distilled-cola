"""FastAPI 服务 — 上海房产分析专家 RESTful API。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import json
import asyncio

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

from src.rag.pipeline import RAGPipeline
from src.app.feedback import feedback_handler, Feedback

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
    conv_id: Optional[str] = None
    mode: str = "detailed"


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
    result = pipe.ask(req.query, top_k=req.top_k, conv_id=req.conv_id, mode=req.mode)
    elapsed = time.time() - t0
    print(
        "[ask] query={} | confidence={:.2f} | chains={} | web={} | "
        "sources={} | answer_len={} | 耗时={:.1f}s".format(
            req.query[:80],
            result.get("confidence", 0),
            result.get("reasoning_chains_used", 0),
            result.get("web_search_used", False),
            len(result.get("sources", [])),
            len(result.get("answer", "")),
            elapsed,
        ),
        flush=True,
    )
    return result


@app.get("/ask/stream")
def ask_stream(query: str, top_k: int = 5, conv_id: Optional[str] = None, mode: str = "detailed"):
    """流式问答 — SSE (Server-Sent Events)，支持多轮对话"""
    def event_stream():
        for chunk in pipe.ask_stream(query=query, top_k=top_k, conv_id=conv_id, mode=mode):
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            if chunk.get("type") == "done":
                break
    return StreamingResponse(event_stream(), media_type="text/event-stream")


class FeedbackRequest(BaseModel):
    answer_id: str
    rating: str  # "up" | "down" | "correct"
    correction: str = ""
    query: Optional[str] = ""


@app.post("/feedback")
def feedback(req: FeedbackRequest) -> dict:
    """提交用户反馈。"""
    fb = Feedback(
        answer_id=req.answer_id,
        rating=req.rating,
        correction=req.correction,
        query=req.query,
    )
    ok = feedback_handler.save_feedback(fb)
    return {"ok": ok, "id": fb.id}


@app.get("/districts")
def districts() -> dict:
    """获取知识库中所有已知板块列表。"""
    from src.knowledge_base.vector_store import KnowledgeIndex
    kb = KnowledgeIndex()
    all_data = kb.collection.get(include=["metadatas"])
    areas = set()
    for m in all_data["metadatas"]:
        if m.get("areas"):
            for a in m["areas"].split(","):
                a = a.strip()
                if len(a) >= 2:
                    areas.add(a)
    return {"districts": sorted(areas)}


@app.get("/data/districts")
def data_districts() -> dict:
    """结构化数据中的板块列表及统计。"""
    from src.data.db import init_db, query
    init_db()
    districts = query("""
        SELECT d.name, d.area, d.ring_road,
               COUNT(DISTINCT p.id) as properties,
               COUNT(l.id) as listings,
               ROUND(AVG(l.unit_price), 0) as avg_price
        FROM districts d
        LEFT JOIN properties p ON p.district_id = d.id
        LEFT JOIN listings l ON l.property_id = p.id AND l.status = '在售'
        GROUP BY d.id
        ORDER BY d.name
    """)
    return {"districts": districts}


@app.get("/data/listings")
def data_listings(
    district: str = None, min_price: float = 0, max_price: float = 0,
    layout: str = None, limit: int = 20,
) -> dict:
    """查询结构化房源数据。"""
    from src.data.db import init_db, find_listings
    init_db()
    results = find_listings(
        district_name=district,
        min_price=min_price, max_price=max_price,
        layout=layout, limit=limit,
    )
    return {"listings": results, "count": len(results)}


@app.get("/data/stats")
def data_stats() -> dict:
    """结构化数据统计。"""
    from src.data.db import init_db, query
    init_db()
    districts = query("SELECT COUNT(*) as c FROM districts")[0]["c"]
    properties = query("SELECT COUNT(*) as c FROM properties")[0]["c"]
    listings = query("SELECT COUNT(*) as c FROM listings")[0]["c"]
    return {"districts": districts, "properties": properties, "listings": listings}


@app.get("/stats")
def stats() -> dict:
    """知识库统计。"""
    from src.knowledge_base.vector_store import KnowledgeIndex
    from src.knowledge_base.reasoning_index import ReasoningIndex
    kb = KnowledgeIndex()
    ri = ReasoningIndex()
    return {"knowledge": kb.stats(), "reasoning_chains": ri.stats()}
