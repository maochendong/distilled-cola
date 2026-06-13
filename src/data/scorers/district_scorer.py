"""
District Scorer — 板块评分系统
多因子加权评分模型
"""

from __future__ import annotations

import logging
import math

from src.data.db import get_db, query, execute

logger = logging.getLogger(__name__)

# 各维度权重
WEIGHTS = {
    "price": 0.30,
    "school": 0.20,
    "commute": 0.15,
    "appreciation": 0.30,
    "new_supply": 0.05,
}


def _normalize(value: float, min_v: float, max_v: float) -> float:
    """归一化到 0-10。"""
    if max_v <= min_v:
        return 5.0
    return min(10.0, max(0.0, (value - min_v) / (max_v - min_v) * 10))


def _inverse_normalize(value: float, min_v: float, max_v: float) -> float:
    """反向归一化(越低分越高)。"""
    return 10 - _normalize(value, min_v, max_v)


def calculate_scores(district_name: str = None) -> list[dict]:
    """计算板块评分。"""
    db = get_db()
    results = []

    # 获取所有板块的统计数据
    rows = query("""
        SELECT d.id, d.name,
               AVG(l.unit_price) as avg_price,
               COUNT(l.id) as listing_count,
               COUNT(DISTINCT p.id) as property_count,
               COUNT(DISTINCT ms.id) as metro_count
        FROM districts d
        LEFT JOIN properties p ON p.district_id = d.id
        LEFT JOIN listings l ON l.property_id = p.id AND l.status = '在售'
        LEFT JOIN property_metro pm ON pm.property_id = p.id
        LEFT JOIN metro_stations ms ON ms.id = pm.station_id
        GROUP BY d.id
    """)

    if not rows:
        return results

    # 找极值用于归一化
    prices = [r["avg_price"] or 0 for r in rows]
    prices = [p for p in prices if p > 0]
    min_price = min(prices) if prices else 0
    max_price = max(prices) if prices else 1

    for r in rows:
        if district_name and r["name"] != district_name:
            continue

        avg_price = r["avg_price"] or 0

        # 各维度评分
        score_price = _inverse_normalize(avg_price, min_price, max_price) if avg_price > 0 else 5.0
        score_school = min(10.0, query("""
            SELECT COUNT(*) * 2 as sc FROM property_school ps
            JOIN properties p ON ps.property_id = p.id
            JOIN schools s ON ps.school_id = s.id
            WHERE p.district_id = ? AND s.tier = '一梯队'
        """, (r["id"],))[0]["sc"] or 0)
        score_commute = min(10.0, (r["metro_count"] or 0) * 3.0)
        score_appreciation = 5.0  # 暂无历史数据，默认中值
        score_new_supply = min(10.0, math.log((r["listing_count"] or 0) + 1) * 3)

        total = (
            score_price * WEIGHTS["price"]
            + score_school * WEIGHTS["school"]
            + score_commute * WEIGHTS["commute"]
            + score_appreciation * WEIGHTS["appreciation"]
            + score_new_supply * WEIGHTS["new_supply"]
        )

        results.append({
            "name": r["name"],
            "score_price": round(score_price, 1),
            "score_school": round(score_school, 1),
            "score_commute": round(score_commute, 1),
            "score_appreciation": round(score_appreciation, 1),
            "score_new_supply": round(score_new_supply, 1),
            "score_total": round(total, 2),
        })

    return results


def compute_and_store_all() -> int:
    """计算所有板块评分并写入数据库。"""
    scores = calculate_scores()
    count = 0
    for s in scores:
        district = query("SELECT id FROM districts WHERE name = ?", (s["name"],))
        if not district:
            continue
        did = district[0]["id"]
        execute("""
            INSERT OR REPLACE INTO district_scores
                (district_id, score_price, score_school, score_commute,
                 score_appreciation, score_new_supply, score_total)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (did, s["score_price"], s["score_school"], s["score_commute"],
              s["score_appreciation"], s["score_new_supply"], s["score_total"]))
        count += 1
    logger.info("板块评分计算完成: %d 个板块", count)
    return count
