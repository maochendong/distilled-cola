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

QUESTIONNAIRE_TEMPLATES = {
    "firsthome": [
        {"id": "workplace", "type": "text", "label": "上班地点在哪？", "placeholder": "如：张江、漕河泾、陆家嘴..."},
        {"id": "commute_tolerance", "type": "radio", "label": "通勤能忍受多久？", "options": ["15分钟", "30分钟", "1小时", "1小时以上"]},
        {"id": "down_payment", "type": "number", "label": "首付能凑多少？（万）", "placeholder": "如：100"},
        {"id": "holding_period", "type": "radio", "label": "几年内打算换房？", "options": ["3年以内", "5年左右", "10年以上", "不确定"]},
    ],
    "herhome": [
        {"id": "living_alone", "type": "radio", "label": "是否独居？", "options": ["是", "否"]},
        {"id": "night_frequency", "type": "radio", "label": "夜间出入频率？", "options": ["每天", "偶尔", "很少"]},
        {"id": "safety_concern", "type": "text", "label": "最担心什么安全问题？", "placeholder": "如：小区照明、监控、门禁..."},
    ],
    "family": [
        {"id": "children_ages", "type": "text", "label": "孩子几岁？", "placeholder": "如：3岁、5岁和8岁"},
        {"id": "need_school", "type": "radio", "label": "是否需要换学区？", "options": ["是", "否", "不确定"]},
        {"id": "current_home", "type": "radio", "label": "现在房子卖还是租？", "options": ["卖掉", "出租", "还没买"]},
        {"id": "commute_locations", "type": "text", "label": "夫妻通勤地点？", "placeholder": "如：南京西路 & 张江"},
    ],
    "golden": [
        {"id": "chronic_condition", "type": "radio", "label": "是否有慢性病？", "options": ["是", "否"]},
        {"id": "hospital_visits", "type": "radio", "label": "去三甲医院频率？", "options": ["每周", "每月", "偶尔"]},
        {"id": "living_alone", "type": "radio", "label": "是否独居？", "options": ["是", "否（与子女同住）"]},
        {"id": "children_proximity", "type": "radio", "label": "子女住得近吗？", "options": ["同小区", "同区", "跨区"]},
    ],
}

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
    profile_id: Optional[str] = None


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
    result = pipe.ask(req.query, top_k=req.top_k, conv_id=req.conv_id, mode=req.mode, profile_id=req.profile_id)
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
def ask_stream(query: str, top_k: int = 5, conv_id: Optional[str] = None, mode: str = "detailed", profile_id: Optional[str] = None):
    """流式问答 — SSE (Server-Sent Events)，支持多轮对话"""
    def event_stream():
        for chunk in pipe.ask_stream(query=query, top_k=top_k, conv_id=conv_id, mode=mode, profile_id=profile_id):
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            if chunk.get("type") == "done":
                break
    return StreamingResponse(event_stream(), media_type="text/event-stream")


class ProfileRequest(BaseModel):
    profile_id: Optional[str] = None
    identity_type: str = "general"
    identity_label: str = ""
    questionnaire: dict = {}


@app.post("/profile")
def create_or_update_profile(req: ProfileRequest) -> dict:
    """创建或更新用户画像。"""
    from src.data.db import init_db
    from src.data.user_profiles import create_profile, update_profile, get_profile
    init_db()
    if req.profile_id:
        existing = get_profile(req.profile_id)
        if existing:
            updated = update_profile(
                req.profile_id,
                identity_type=req.identity_type,
                identity_label=req.identity_label,
                questionnaire=req.questionnaire,
            )
            return {"profile_id": req.profile_id, "identity_type": req.identity_type}
        # profile_id not found, create new
    profile = create_profile(
        identity_type=req.identity_type,
        identity_label=req.identity_label,
        questionnaire=req.questionnaire,
    )
    return {"profile_id": profile["profile_id"], "identity_type": profile["identity_type"]}


@app.get("/profile/{profile_id}")
def api_get_profile(profile_id: str) -> dict:
    """获取用户画像。"""
    from src.data.db import init_db
    from src.data.user_profiles import get_profile
    init_db()
    profile = get_profile(profile_id)
    if not profile:
        return {"error": "profile not found"}
    return profile


class QuestionnaireRequest(BaseModel):
    profile_id: str
    identity_type: str
    answers: dict = {}


@app.post("/questionnaire")
def submit_questionnaire(req: QuestionnaireRequest) -> dict:
    """提交需求深度问卷答案。"""
    from src.data.db import init_db
    from src.data.user_profiles import update_profile
    init_db()
    profile = update_profile(req.profile_id, questionnaire=req.answers)
    if not profile:
        return {"error": "profile not found"}
    return {"profile_id": req.profile_id, "questionnaire_count": len(req.answers)}


@app.get("/questionnaire/templates")
def get_questionnaire_templates() -> dict:
    """获取所有身份的问卷模板。"""
    return {"templates": QUESTIONNAIRE_TEMPLATES}


@app.get("/data/price_trend")
def price_trend(district: str = None, months: int = 12) -> dict:
    """获取板块价格趋势(按月聚合)。"""
    from src.data.db import init_db, query
    init_db()
    params: list = []
    where = ""
    if district:
        districts = [d.strip() for d in district.split(",") if d.strip()]
        if districts:
            placeholders = ",".join(["?" for _ in districts])
            where = f"AND d.name IN ({placeholders})"
            params.extend(districts)
    rows = query(f"""
        SELECT d.name as district,
               strftime('%Y-%m', t.trans_date) as month,
               ROUND(AVG(t.unit_price), 0) as avg_price,
               COUNT(t.id) as transaction_count
        FROM transactions t
        JOIN properties p ON t.property_id = p.id
        JOIN districts d ON p.district_id = d.id
        WHERE t.trans_date >= date('now', '-{months + 1} months')
        {where}
        GROUP BY d.name, month
        ORDER BY month
    """, tuple(params))
    return {"trends": rows}


@app.get("/data/district_scores")
def api_district_scores(district: str = None) -> dict:
    """获取板块评分。"""
    from src.data.db import init_db, query
    init_db()
    if district:
        rows = query("""
            SELECT d.name, s.score_price, s.score_school, s.score_commute,
                   s.score_appreciation, s.score_new_supply, s.score_total
            FROM district_scores s
            JOIN districts d ON s.district_id = d.id
            WHERE d.name = ?
        """, (district,))
    else:
        rows = query("""
            SELECT d.name, s.score_price, s.score_school, s.score_commute,
                   s.score_appreciation, s.score_new_supply, s.score_total
            FROM district_scores s
            JOIN districts d ON s.district_id = d.id
            ORDER BY s.score_total DESC
        """)
    return {"scores": rows}


@app.post("/data/compute_scores")
def compute_district_scores() -> dict:
    """计算并存储所有板块评分。"""
    from src.data.scorers.district_scorer import compute_and_store_all
    from src.data.db import init_db
    init_db()
    count = compute_and_store_all()
    return {"computed": count}


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


SUGGEST_TEMPLATES = {
    "前滩": ["前滩 vs 徐汇滨江怎么选？", "前滩二手房成交价", "前滩新房认筹情况"],
    "大宁": ["大宁学区房推荐", "大宁房价走势", "大宁 vs 不夜城怎么选"],
    "唐镇": ["唐镇房价现在多少", "唐镇二手房值得买吗", "唐镇 vs 周浦怎么选"],
    "徐汇滨江": ["徐汇滨江房价", "徐汇滨江新房供应", "徐汇滨江 vs 前滩"],
    "泗泾": ["泗泾房价现在多少", "泗泾到漕河泾通勤时间", "泗泾新房"],
    "江桥": ["江桥房价", "江桥北虹桥规划", "江桥 vs 南翔"],
    "周浦": ["周浦房价", "周浦18号线", "周浦二手房推荐"],
    "张江": ["张江板块分析", "张江通勤圈买房", "张江周边板块推荐"],
}


@app.get("/suggest")
def suggest(q: str = "", limit: int = 6) -> dict:
    """搜索建议和自动补全。"""
    results = []
    if not q.strip():
        return {"suggestions": results}

    from src.data.db import init_db, query
    init_db()

    # 1. 匹配板块名
    districts = query("SELECT name FROM districts WHERE name LIKE ? LIMIT 3", (f"%{q}%",))
    for d in districts:
        results.append({
            "type": "district",
            "text": d["name"],
            "label": f"📍 板块 · {d['name']}",
        })

    # 2. 匹配模板问题
    for keyword, questions in SUGGEST_TEMPLATES.items():
        if keyword in q or q in keyword:
            for question in questions[:2]:
                results.append({"type": "question", "text": question, "label": f"❓ {question}"})

    # 3. 通用建议
    if len(results) < limit:
        general = [
            f"预算{q}万在上海能买哪？",
            f"{q}板块怎么样？",
            f"{q}值得买吗？",
        ]
        for g in general:
            if len(results) < limit:
                results.append({"type": "suggestion", "text": g, "label": f"💡 {g}"})

    return {"suggestions": results[:limit]}


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
