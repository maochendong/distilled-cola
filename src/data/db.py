"""
Structured Data Layer — SQLite 数据库层
存储楼盘、成交、学区、地铁等结构化房产数据
"""

from __future__ import annotations

import sqlite3
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent.parent / "data" / "metadata" / "sh.db"


# ── Schema ──

SCHEMA_SQL = """
-- 板块
CREATE TABLE IF NOT EXISTS districts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    UNIQUE NOT NULL,          -- 板块名（如"前滩"）
    area        TEXT    NOT NULL DEFAULT '',       -- 所属区域（如"浦东"）
    ring_road   TEXT    DEFAULT '',                -- 环线（内环/中环/外环/郊环）
    description TEXT    DEFAULT ''
);

-- 楼盘
CREATE TABLE IF NOT EXISTS properties (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    district_id     INTEGER NOT NULL REFERENCES districts(id),
    name            TEXT    NOT NULL,              -- 楼盘名
    alias           TEXT    DEFAULT '',            -- 别名
    address         TEXT    DEFAULT '',
    completion_year INTEGER DEFAULT 0,             -- 竣工年份
    total_units     INTEGER DEFAULT 0,             -- 总户数
    volume_rate     REAL    DEFAULT 0,             -- 容积率
    green_rate      REAL    DEFAULT 0,             -- 绿化率
    developer       TEXT    DEFAULT '',            -- 开发商
    property_type   TEXT    DEFAULT '住宅',        -- 住宅/公寓/别墅
    lat             REAL    DEFAULT 0,             -- 纬度
    lng             REAL    DEFAULT 0,             -- 经度
    UNIQUE(district_id, name)
);

-- 房源挂牌（链家/贝壳挂牌数据）
CREATE TABLE IF NOT EXISTS listings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    property_id     INTEGER NOT NULL REFERENCES properties(id),
    layout          TEXT    DEFAULT '',            -- 户型（如"3室2厅"）
    size_sqm        REAL    NOT NULL DEFAULT 0,    -- 面积（㎡）
    floor           TEXT    DEFAULT '',            -- 楼层（低/中/高）
    total_floor     INTEGER DEFAULT 0,             -- 总楼层
    orientation     TEXT    DEFAULT '',            -- 朝向
    decoration      TEXT    DEFAULT '',            -- 装修
    total_price     REAL    NOT NULL DEFAULT 0,    -- 总价（万）
    unit_price      REAL    NOT NULL DEFAULT 0,    -- 单价（元/㎡）
    listing_date    TEXT    DEFAULT '',            -- 挂牌日期
    status          TEXT    DEFAULT '在售',        -- 在售/已售/下架
    source          TEXT    DEFAULT 'lianjia',     -- 数据来源
    updated_at      TEXT    DEFAULT (datetime('now'))
);

-- 历史成交
CREATE TABLE IF NOT EXISTS transactions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    property_id     INTEGER NOT NULL REFERENCES properties(id),
    layout          TEXT    DEFAULT '',
    size_sqm        REAL    NOT NULL DEFAULT 0,
    floor           TEXT    DEFAULT '',
    total_price     REAL    NOT NULL DEFAULT 0,
    unit_price      REAL    NOT NULL DEFAULT 0,
    trans_date      TEXT    NOT NULL,              -- 成交日期
    source          TEXT    DEFAULT 'lianjia',
    created_at      TEXT    DEFAULT (datetime('now'))
);

-- 地铁站
CREATE TABLE IF NOT EXISTS metro_stations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT    NOT NULL,              -- 站名
    line            TEXT    NOT NULL,              -- 线路（如"2号线"）
    lat             REAL    DEFAULT 0,
    lng             REAL    DEFAULT 0
);

-- 楼盘-地铁距离
CREATE TABLE IF NOT EXISTS property_metro (
    property_id     INTEGER NOT NULL REFERENCES properties(id),
    station_id      INTEGER NOT NULL REFERENCES metro_stations(id),
    distance_m      INTEGER DEFAULT 0,            -- 步行距离（米）
    PRIMARY KEY (property_id, station_id)
);

-- 学校
CREATE TABLE IF NOT EXISTS schools (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT    NOT NULL,
    tier            TEXT    DEFAULT '',            -- 梯队（一梯队/二梯队/三梯队）
    type            TEXT    DEFAULT '',            -- 小学/初中/九年一贯
    district        TEXT    DEFAULT ''
);

-- 楼盘-学区对口
CREATE TABLE IF NOT EXISTS property_school (
    property_id     INTEGER NOT NULL REFERENCES properties(id),
    school_id       INTEGER NOT NULL REFERENCES schools(id),
    year            INTEGER DEFAULT 0,            -- 对口年份
    PRIMARY KEY (property_id, school_id)
);

-- 评分
CREATE TABLE IF NOT EXISTS district_scores (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    district_id         INTEGER NOT NULL REFERENCES districts(id),
    score_price         REAL    DEFAULT 0,
    score_school        REAL    DEFAULT 0,
    score_commute       REAL    DEFAULT 0,
    score_appreciation  REAL    DEFAULT 0,
    score_new_supply    REAL    DEFAULT 0,
    score_total         REAL    DEFAULT 0,
    updated_at          TEXT    DEFAULT (datetime('now')),
    UNIQUE(district_id)
);

-- 用户画像
CREATE TABLE IF NOT EXISTS user_profiles (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id      TEXT    UNIQUE NOT NULL,
    identity_type   TEXT    NOT NULL DEFAULT 'general',
    identity_label  TEXT    DEFAULT '',
    questionnaire   TEXT    DEFAULT '{}',
    created_at      TEXT    DEFAULT (datetime('now')),
    updated_at      TEXT    DEFAULT (datetime('now'))
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_listings_property ON listings(property_id);
CREATE INDEX IF NOT EXISTS idx_listings_price ON listings(total_price);
CREATE INDEX IF NOT EXISTS idx_transactions_property ON transactions(property_id);
CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions(trans_date);
CREATE INDEX IF NOT EXISTS idx_properties_district ON properties(district_id);
CREATE INDEX IF NOT EXISTS idx_profiles_pid ON user_profiles(profile_id);
"""


# ── 数据库连接 ──

_conn: Optional[sqlite3.Connection] = None


def get_db() -> sqlite3.Connection:
    """获取数据库连接（单例）。"""
    global _conn
    if _conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(DB_PATH))
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA foreign_keys=ON")
    return _conn


def init_db():
    """初始化数据库，创建所有表。"""
    conn = get_db()
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    logger.info("数据库已初始化: %s", DB_PATH)


def close_db():
    """关闭数据库连接。"""
    global _conn
    if _conn:
        _conn.close()
        _conn = None


# ── 查询辅助 ──

def query(sql: str, params: tuple = ()) -> list[dict]:
    """执行查询，返回 dict 列表。"""
    conn = get_db()
    rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def query_one(sql: str, params: tuple = ()) -> Optional[dict]:
    """执行查询，返回单条。"""
    rows = query(sql, params)
    return rows[0] if rows else None


def execute(sql: str, params: tuple = ()) -> int:
    """执行写入，返回受影响行数。"""
    conn = get_db()
    conn.execute(sql, params)
    conn.commit()
    return conn.total_changes


def bulk_insert(table: str, rows: list[dict]):
    """批量插入数据。"""
    if not rows:
        return
    conn = get_db()
    cols = ", ".join(rows[0].keys())
    placeholders = ", ".join(["?" for _ in rows[0]])
    sql = f"INSERT OR IGNORE INTO {table} ({cols}) VALUES ({placeholders})"
    values = [[r[k] for k in rows[0]] for r in rows]
    conn.executemany(sql, values)
    conn.commit()


# ── 域查询 ──

def search_district(keyword: str) -> list[dict]:
    """模糊搜索板块。"""
    return query(
        "SELECT * FROM districts WHERE name LIKE ?", (f"%{keyword}%",)
    )


def get_district_avg_price(district_name: str) -> Optional[float]:
    """获取板块均价（基于最近挂牌）。"""
    row = query_one("""
        SELECT AVG(l.unit_price) as avg_price
        FROM listings l
        JOIN properties p ON l.property_id = p.id
        JOIN districts d ON p.district_id = d.id
        WHERE d.name = ? AND l.status = '在售'
    """, (district_name,))
    return row["avg_price"] if row and row["avg_price"] else None


def get_district_stats(district_name: str) -> dict:
    """获取板块统计信息。"""
    return query_one("""
        SELECT
            d.name,
            COUNT(DISTINCT p.id) as property_count,
            COUNT(l.id) as listing_count,
            ROUND(AVG(l.unit_price), 0) as avg_unit_price,
            ROUND(AVG(l.total_price), 0) as avg_total_price,
            ROUND(MIN(l.total_price), 0) as min_price,
            ROUND(MAX(l.total_price), 0) as max_price,
            ROUND(AVG(l.size_sqm), 1) as avg_size
        FROM districts d
        LEFT JOIN properties p ON p.district_id = d.id
        LEFT JOIN listings l ON l.property_id = p.id AND l.status = '在售'
        WHERE d.name = ?
        GROUP BY d.id
    """, (district_name,)) or {}


def compare_districts(names: list[str]) -> list[dict]:
    """对比多个板块的统计数据。"""
    results = []
    for name in names:
        stats = get_district_stats(name)
        if stats:
            results.append(stats)
    return results


def find_listings(district_name: str = None,
                  min_price: float = 0, max_price: float = 0,
                  min_size: float = 0, max_size: float = 0,
                  layout: str = None, limit: int = 20) -> list[dict]:
    """筛选房源。"""
    conditions = ["l.status = '在售'"]
    params = []

    if district_name:
        conditions.append("d.name = ?")
        params.append(district_name)
    if min_price > 0:
        conditions.append("l.total_price >= ?")
        params.append(min_price)
    if max_price > 0:
        conditions.append("l.total_price <= ?")
        params.append(max_price)
    if min_size > 0:
        conditions.append("l.size_sqm >= ?")
        params.append(min_size)
    if max_size > 0:
        conditions.append("l.size_sqm <= ?")
        params.append(max_size)
    if layout:
        conditions.append("l.layout LIKE ?")
        params.append(f"%{layout}%")

    where = " AND ".join(conditions)
    return query(f"""
        SELECT d.name as district, p.name as property, l.layout,
               l.size_sqm, l.total_price, l.unit_price, l.floor,
               l.orientation, l.decoration, l.listing_date
        FROM listings l
        JOIN properties p ON l.property_id = p.id
        JOIN districts d ON p.district_id = d.id
        WHERE {where}
        ORDER BY l.total_price
        LIMIT ?
    """, tuple(params) + (limit,))
