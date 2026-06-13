"""
链家/贝壳 楼盘数据导入器
支持从 CSV 文件或 Lianjia 公开页面导入

数据来源：
  - CSV 文件（推荐）：导出的结构化数据
  - 贝壳找房网页（实验性）：从公开页面抓取
"""

from __future__ import annotations

import csv
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from src.data.db import (
    get_db, init_db, bulk_insert, query, query_one,
    execute, search_district,
)

logger = logging.getLogger(__name__)


# ── CSV 导入 ──

def import_from_csv(csv_path: str | Path) -> dict:
    """从 CSV 文件导入楼盘+房源数据。

    支持两种 CSV 格式：
    1. 楼盘清单：district, property, address, completion_year, total_units, volume_rate
    2. 房源清单：district, property, layout, size_sqm, floor, total_floor, orientation, decoration, total_price, unit_price

    Returns: 导入统计
    """
    init_db()
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV 文件不存在: {path}")

    stats = {"districts": 0, "properties": 0, "listings": 0, "errors": 0}

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV 文件为空或格式不正确")

        headers = [h.strip().lower() for h in reader.fieldnames]
        for row in reader:
            try:
                row = {k.strip().lower(): v.strip() for k, v in row.items()}

                # 确保板块存在
                district_name = row.get("district", "")
                if not district_name:
                    continue

                existing = search_district(district_name)
                if not existing:
                    execute(
                        "INSERT INTO districts (name, area, ring_road) VALUES (?, ?, ?)",
                        (district_name, row.get("area", ""), row.get("ring_road", "")),
                    )
                    stats["districts"] += 1

                district = search_district(district_name)[0]

                # 如果包含楼盘名 → 导入楼盘+房源
                property_name = row.get("property", "")
                if property_name:
                    prop = query_one(
                        "SELECT id FROM properties WHERE district_id=? AND name=?",
                        (district["id"], property_name),
                    )
                    if not prop:
                        execute("""
                            INSERT INTO properties
                                (district_id, name, address, completion_year,
                                 total_units, volume_rate, developer, property_type)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            district["id"], property_name,
                            row.get("address", ""),
                            _int(row.get("completion_year", 0)),
                            _int(row.get("total_units", 0)),
                            _float(row.get("volume_rate", 0)),
                            row.get("developer", ""),
                            row.get("property_type", "住宅"),
                        ))
                        stats["properties"] += 1
                        prop = query_one(
                            "SELECT id FROM properties WHERE district_id=? AND name=?",
                            (district["id"], property_name),
                        )

                    # 如果有价格字段 → 导入房源
                    if prop and row.get("total_price"):
                        execute("""
                            INSERT INTO listings
                                (property_id, layout, size_sqm, floor,
                                 total_floor, orientation, decoration,
                                 total_price, unit_price, listing_date, source)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'lianjia')
                        """, (
                            prop["id"],
                            row.get("layout", ""),
                            _float(row.get("size_sqm", 0)),
                            row.get("floor", ""),
                            _int(row.get("total_floor", 0)),
                            row.get("orientation", ""),
                            row.get("decoration", ""),
                            _float(row.get("total_price", 0)),
                            _float(row.get("unit_price", 0)),
                            row.get("listing_date", ""),
                        ))
                        stats["listings"] += 1

            except Exception as e:
                stats["errors"] += 1
                logger.warning("导入行失败: %s", e)

    logger.info("CSV 导入完成: %s", stats)
    return stats


# ── 贝壳找房 API 抓取（实验性）──

BEIKE_BASE = "https://sh.ke.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
}


def _fetch(url: str) -> Optional[str]:
    """HTTP GET 请求。"""
    try:
        import urllib.request
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read().decode("utf-8")
    except Exception as e:
        logger.warning("请求失败 %s: %s", url, e)
        return None


def _int(v) -> int:
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return 0


def _float(v) -> float:
    try:
        return float(v)
    except (ValueError, TypeError):
        return 0.0


def _extract_json_from_script(html: str, pattern: str) -> Optional[dict]:
    """从页面 script 标签中提取 JSON 数据。"""
    match = re.search(pattern, html)
    if match:
        try:
            # 处理可能的 XSS 转义
            raw = match.group(1)
            raw = raw.replace("\\'", "'").replace('\\"', '"').replace("\\\\", "\\")
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return None


def scrape_district_list(district_name: str, max_pages: int = 3) -> list[dict]:
    """从贝壳找房抓取某板块的二手房列表。

    注意：贝壳页面结构可能变化，此功能仅供实验性参考。
    正式使用推荐 CSV 导入方式。
    """
    from urllib.parse import quote
    results = []
    base_url = f"{BEIKE_BASE}/ershoufang/{quote(district_name)}"

    for page in range(1, max_pages + 1):
        url = f"{base_url}/pg{page}/"

        html = _fetch(url)
        if not html:
            break

        # 解析房源列表（贝壳页面结构）
        # 从 data-type="1" 的 li 标签中提取
        pattern = r'data-resblock-name="([^"]+)".*?data-price="([^"]+)".*?data-totalprice="([^"]+)"'
        matches = re.findall(pattern, html)

        for name, unit_price, total_price in matches:
            results.append({
                "district": district_name,
                "property": name,
                "total_price": _float(total_price),
                "unit_price": _float(unit_price),
                "source": "beike",
            })

        time.sleep(1)  # 限流

    return results


def scrape_and_import(district_name: str, max_pages: int = 3) -> dict:
    """抓取贝壳找房某板块数据并导入 SQLite。"""
    listings = scrape_district_list(district_name, max_pages)
    if not listings:
        logger.warning("未抓取到 %s 的数据", district_name)
        return {"scraped": 0, "imported": 0}

    init_db()

    # 确保板块存在
    existing = search_district(district_name)
    if not existing:
        execute(
            "INSERT INTO districts (name) VALUES (?)",
            (district_name,),
        )
        district = search_district(district_name)[0]
    else:
        district = existing[0]

    imported = 0
    for item in listings:
        prop = query_one(
            "SELECT id FROM properties WHERE district_id=? AND name=?",
            (district["id"], item["property"]),
        )
        if not prop:
            execute(
                "INSERT INTO properties (district_id, name) VALUES (?, ?)",
                (district["id"], item["property"]),
            )
            prop = query_one(
                "SELECT id FROM properties WHERE district_id=? AND name=?",
                (district["id"], item["property"]),
            )

        if prop:
            execute("""
                INSERT INTO listings
                    (property_id, total_price, unit_price, source)
                VALUES (?, ?, ?, 'beike_scrape')
            """, (prop["id"], item["total_price"], item["unit_price"]))
            imported += 1

    logger.info("从贝壳导入 %s: %d/%d", district_name, imported, len(listings))
    return {"scraped": len(listings), "imported": imported}


# ── CSV 示例模板 ──

SAMPLE_CSV = """\
district,area,ring_road,property,address,completion_year,total_units,volume_rate,layout,size_sqm,total_price,unit_price
前滩,浦东,中环,晶耀名邸,东育路202弄,2018,624,2.5,3室2厅,98,1350,137755
前滩,浦东,中环,晶耀名邸,东育路202弄,2018,624,2.5,2室2厅,82,1150,140244
前滩,浦东,中环,东方悦耀,海阳西路128弄,2020,632,3.2,2室1厅,51,780,152941
大宁,静安,中环,静安府,汶水路699弄,2020,1164,2.65,3室2厅,95,950,100000
大宁,静安,中环,静安府,汶水路699弄,2020,1164,2.65,4室2厅,135,1500,111111
大宁,静安,中环,大宁金茂府,彭江路366弄,2017,771,2.5,3室2厅,110,1350,122727
大宁,静安,中环,宝华现代城,广延路1188弄,2012,882,2.5,2室2厅,87,750,86207
徐汇滨江,徐汇,内环,百汇园,龙恒东路18弄,2019,1236,2.8,3室2厅,120,1800,150000
徐汇滨江,徐汇,内环,尚海湾豪庭,龙腾大道2180号,2017,3256,3.5,2室2厅,89,1200,134831
唐镇,浦东,外环,仁恒东郊花园,玉盘北路288弄,2018,1053,1.6,3室2厅,98,780,79592
唐镇,浦东,外环,绿城玉兰花园,齐爱路99弄,2016,1143,1.5,4室2厅,140,1250,89286
唐镇,浦东,外环,大名城紫金九号,顾唐路138弄,2020,711,2.0,3室2厅,89,700,78652
"""


def create_sample_csv(path: str | Path = "data/metadata/sample_shanghai.csv"):
    """生成示例 CSV 数据文件用于测试。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(SAMPLE_CSV, encoding="utf-8")
    logger.info("示例数据已生成: %s", p)
    return str(p)
