"""
Comparison Engine — 板块对比模块
支持 2-3 个板块并排对比：均价/学区/通勤/增值潜力/新房供应
"""
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

COMPARISON_SYSTEM_PROMPT = """你是一位上海房产数据分析师。
请基于检索到的知识，对用户指定的几个板块进行横向对比分析。

输出格式必须为 JSON:
{{
    "districts": [
        {{
            "name": "板块名",
            "scores": {{
                "price": 0-10,
                "school": 0-10,
                "commute": 0-10,
                "appreciation": 0-10,
                "new_supply": 0-10
            }},
            "avg_price": "均价描述",
            "pros": ["优点1", "优点2"],
            "cons": ["缺点1", "缺点2"],
            "verdict": "一句话总结"
        }}
    ],
    "recommendation": "综合建议"
}}
"""

COMPARISON_USER_TEMPLATE = """请对比以下上海板块：{districts}

对比维度：
1. 平均房价（新房/二手房）
2. 学区资源（学校梯队、对口稳定性）
3. 通勤便利性（到核心 CBD 的时间）
4. 增值潜力（近期涨幅、规划利好）
5. 新房供应（在售/待售新盘）

请给出每个维度的 0-10 分评分、优缺点和综合建议。"""


class ComparisonEngine:
    """板块对比引擎 — 使用现有 RAG 管线生成结构化对比"""

    def __init__(self, pipeline=None):
        self.pipeline = pipeline

    def compare(self, districts: list[str],
                pipeline=None) -> dict:
        """对指定板块进行对比分析"""
        rag = pipeline or self.pipeline
        if not rag:
            return {"error": "RAG pipeline 未配置", "districts": []}

        prompt = COMPARISON_USER_TEMPLATE.format(
            districts="、".join(districts)
        )

        try:
            result = rag.ask(
                query=prompt,
                system_prompt=COMPARISON_SYSTEM_PROMPT,
                response_format={"type": "json_object"},
            )
            raw = result.get("answer", "{}")
            # 尝试提取 JSON
            import re
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {"error": "无法解析对比结果", "raw": raw[:500]}
        except Exception as e:
            logger.error("板块对比失败: %s", e)
            return {"error": str(e), "districts": []}

    def render_comparison_table(self, data: dict) -> str:
        """将对比结果渲染为 Markdown 表格"""
        if "error" in data and not data.get("districts"):
            return f"⚠️ 对比失败: {data.get('error')}"

        rows = []
        districts = data.get("districts", [])

        if not districts:
            return "⚠️ 无对比数据"

        # 表头
        headers = ["维度"] + [d["name"] for d in districts]
        rows.append("| " + " | ".join(headers) + " |")
        rows.append("|" + "---|" * len(headers))

        # 评分行
        dims = [
            ("💰 房价", "price"),
            ("📚 学区", "school"),
            ("🚇 通勤", "commute"),
            ("📈 增值", "appreciation"),
            ("🏗️ 新房", "new_supply"),
        ]
        for dim_name, dim_key in dims:
            scores = [str(d.get("scores", {}).get(dim_key, "N/A"))
                      for d in districts]
            rows.append(f"| **{dim_name}** | " + " | ".join(scores) + " |")

        # 均价行
        rows.append("| **均价** | " + " | ".join(
            [d.get("avg_price", "N/A") for d in districts]
        ) + " |")

        # 结论行
        if "recommendation" in data:
            rows.append(f"\n**综合建议**: {data['recommendation']}")

        return "\n".join(rows)

    def render_radar_chart_data(self, data: dict) -> str:
        """生成 ECharts 雷达图数据（JSON 字符串）"""
        districts = data.get("districts", [])
        if not districts:
            return "{}"

        chart_data = {
            "tooltip": {"trigger": "item"},
            "legend": {
                "data": [d["name"] for d in districts],
                "bottom": 0,
            },
            "radar": {
                "indicator": [
                    {"name": "房价", "max": 10},
                    {"name": "学区", "max": 10},
                    {"name": "通勤", "max": 10},
                    {"name": "增值", "max": 10},
                    {"name": "新房", "max": 10},
                ],
            },
            "series": [{
                "type": "radar",
                "data": [
                    {
                        "value": [
                            d.get("scores", {}).get("price", 0),
                            d.get("scores", {}).get("school", 0),
                            d.get("scores", {}).get("commute", 0),
                            d.get("scores", {}).get("appreciation", 0),
                            d.get("scores", {}).get("new_supply", 0),
                        ],
                        "name": d["name"],
                    }
                    for d in districts
                ],
            }],
        }
        return json.dumps(chart_data, ensure_ascii=False)
