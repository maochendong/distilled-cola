"""房产领域标注模块 — 对分析段落进行结构化的实体、逻辑、建议标注。

使用教师模型（DeepSeek Pro）对段落进行逐段标注。
"""

from __future__ import annotations

import json
from pathlib import Path

from src.config import config

# 房产领域标注 Schema
ANNOTATION_SYSTEM_PROMPT = """你是上海房产分析领域的标注专家。对给定的分析段落，提取以下结构化信息。

## 标注 Schema

### 1. 实体标签（涉及哪些实体/概念）
- districts: 涉及的上海行政区
- areas: 涉及的具体板块
- projects: 涉及的楼盘/小区名
- developers: 涉及的开发商
- ring_road: 环线位置（内环/中环/外环/外郊环）
- price_range: 提及的价格区间
- policy_terms: 涉及的政策术语

### 2. 逻辑标签（分析框架类型）
Choose from:
- "板块对比" — 板块之间横向比较
- "学区分析" — 学区/教育相关分析
- "倒挂判断" — 一二手倒挂分析
- "政策解读" — 政策影响分析
- "流动性评估" — 流动性/流通性分析
- "时机判断" — 买卖时机判断
- "供需分析" — 供求关系分析
- "规划利好" — 城市规划/基建利好
- "风险提示" — 风险预警

### 3. 建议标签（博主给出的操作建议）
Choose from:
- "自住推荐" — 推荐自住
- "投资回避" — 不建议投资
- "置换窗口" — 适合置换
- "打新策略" — 新房认购策略
- "卖旧买新" — 建议卖旧换新
- "持币观望" — 建议等待
- "果断入手" — 建议买入
- "风险警示" — 提示特定风险

### 4. 推理链（博主的推演逻辑摘要）
- trigger: 分析的触发因素（数据/政策/事件）
- key_logic: 核心推演逻辑（一句话）
- conclusion: 结论要点

### 5. 置信度
- 0-1 之间的分数，表示你对标注的把握程度

## 输出格式
严格返回 JSON，不要包含其他内容。
"""


def normalize_annotation(ann: dict) -> dict:
    """将 LLM 输出的各种 key 名统一为标准格式。

    DeepSeek 的输出 key 名不固定，此函数尝试所有变体并合并。
    返回的 dict 始终包含以下 key（缺失则为空值）：
      logic_tags, suggestion_tags, areas, districts, projects,
      ring_road, price_range, policy_terms, reasoning_chain, confidence
    """
    if not isinstance(ann, dict):
        ann = {}

    def _first_list(d: dict, *keys: str) -> list:
        for k in keys:
            v = d.get(k)
            if isinstance(v, list) and v:
                return v
            if isinstance(v, str) and v.strip():
                return [v]
        return []

    def _get_nested(d: dict, *paths: str | tuple) -> object:
        """支持嵌套路径如 ('entities', 'areas') 和顶层 key。

        字符串路径直接取值（返回各种类型），
        元组路径钻入嵌套 dict 返回叶子值。
        """
        for path in paths:
            if isinstance(path, str):
                v = d.get(path)
                if v is not None and v != "" and v != []:
                    return v
            elif isinstance(path, tuple):
                v = d
                for part in path:
                    if not isinstance(v, dict):
                        v = None
                        break
                    v = v.get(part)
                if v is not None and v != "" and v != []:
                    return v
        return None

    normalized: dict = {
        "logic_tags": [],
        "suggestion_tags": [],
        "areas": [],
        "districts": [],
        "projects": [],
        "ring_road": "",
        "price_range": [],
        "policy_terms": [],
        "reasoning_chain": {},
        "confidence": 0.0,
    }

    raw_logic = _first_list(
        ann, "logic_tags", "logic_label", "logic_tag",
        "logical_tags", "logical_label", "logical_tag",
        "logical_labels", "logical_type", "logical_framework",
        "logic_labels", "logic_type", "logic_types",
        "logic_category", "logics", "logic",
        "逻辑标签",
    )
    normalized["logic_tags"] = raw_logic

    raw_suggest = _first_list(
        ann, "suggestion_tags", "suggestion_label", "suggestion_tag",
        "suggestion_labels", "suggestion_type", "suggestions",
        "advice_tags", "advice_label", "advice_tag",
        "advice_labels", "advice_type", "advice_types",
        "advice_category", "advice", "suggestion",
        "建议标签",
    )
    normalized["suggestion_tags"] = raw_suggest

    for field in ("areas", "districts", "projects", "ring_road", "price_range", "policy_terms"):
        v = _get_nested(
            ann,
            ("entities", field),
            field,
            ("entity_tags", field),
        )
        if field == "ring_road":
            normalized[field] = v if isinstance(v, str) else ""
        else:
            normalized[field] = v if isinstance(v, list) else ([v] if isinstance(v, str) and v.strip() else [])

    raw_chain = _get_nested(
        ann,
        "reasoning_chain", "推理链", "reasoning",
        "inference_chain", "reasoning_chains", "inference_chains",
    )
    if isinstance(raw_chain, dict):
        normalized["reasoning_chain"] = raw_chain

    raw_conf = _get_nested(ann, "confidence", "置信度")
    if isinstance(raw_conf, (int, float)):
        normalized["confidence"] = float(raw_conf)

    return normalized


def annotate_block(text: str, model: str | None = None) -> dict:
    """对单个分析段落进行标注。

    Args:
        text: 分析段落文本
        model: 教师模型名称（默认用 config.teacher_model）

    Returns:
        标注结果字典
    """
    return _annotate_with_api(text, model or config.teacher_model)


def annotate_batch(
    blocks: list[dict],
    batch_size: int = 5,
    model: str | None = None,
) -> list[dict]:
    """批量标注多个段落，使用教师模型 — 逐段标注。

    Args:
        blocks: 知识块列表（每块含 text 字段）
        batch_size: 每批处理数量，仅用于打印进度
        model: 教师模型名称

    Returns:
        带 annotation 字段的知识块列表
    """
    model = model or config.teacher_model
    annotated = []
    import time, json as _json
    from pathlib import Path as _Path

    # 恢复检查点
    checkpoint_path = _Path("data/processed/annotate_checkpoint.json")
    completed_ids = set()
    if checkpoint_path.exists():
        checkpoint = _json.loads(checkpoint_path.read_text(encoding="utf-8"))
        annotated = checkpoint["done"]
        completed_ids = set(checkpoint["done_ids"])
        print(f"  📦 恢复检查点: {len(completed_ids)} 块已标注")

    for i, block in enumerate(blocks):
        block_id = block.get("id", str(i))
        if block_id in completed_ids:
            continue

        text = block.get("text", "")
        if not text.strip():
            block["annotation"] = {}
            annotated.append(block)
            continue

        # 重试逻辑
        for attempt in range(3):
            try:
                annotation = annotate_block(text, model=model)
                break
            except Exception as e:
                print(f"  ⚠️ 标注 block {i} 失败 (尝试 {attempt+1}/3): {e}")
                annotation = {}
                if attempt < 2:
                    time.sleep(2 ** attempt)  # 指数退避

        block["annotation"] = annotation
        annotated.append(block)

        # 每 20 块保存检查点
        if (i + 1) % 20 == 0:
            checkpoint_path.write_text(
                _json.dumps({
                    "done": annotated,
                    "done_ids": [b.get("id", str(j)) for j, b in enumerate(annotated)],
                }, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"    标注进度: {i+1}/{len(blocks)}  💾 检查点已保存")

        # API 限流
        if (i + 1) % 5 == 0:
            time.sleep(0.5)

    # 清理检查点
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    return annotated


def extract_reasoning_chains(blocks: list[dict]) -> list:
    """从标注后的知识块中提取推理链，用于写入 ReasoningIndex。

    Args:
        blocks: 已标注的知识块列表

    Returns:
        ReasoningChain 列表（去重）
    """
    from src.knowledge_base.reasoning_index import ReasoningChain

    chains: list[ReasoningChain] = []
    seen_triggers: set[str] = set()

    for block in blocks:
        ann = block.get("annotation", {})
        if not ann:
            continue

        normalized = normalize_annotation(ann)
        chain_info = normalized.get("reasoning_chain", {})
        if isinstance(chain_info, dict):
            trigger = (chain_info.get("trigger") or "").strip()
            conclusion = (chain_info.get("conclusion") or "").strip()
            key_logic = (chain_info.get("key_logic") or "").strip()

            if trigger and conclusion:
                dedup_key = trigger[:50]
                if dedup_key not in seen_triggers:
                    seen_triggers.add(dedup_key)
                    rc = ReasoningChain(
                        trigger=trigger,
                        conclusion=conclusion,
                        areas=normalized.get("areas", []),
                        logic_tags=normalized.get("logic_tags", []),
                        districts=normalized.get("districts", []),
                    )
                    if key_logic:
                        rc.add_step(key_logic, logic_type="核心推演")
                    chains.append(rc)

    return chains


def _annotate_with_api(text: str, model: str) -> dict:
    """使用 API（DeepSeek）标注单段文本。"""
    from src.llm import chat_client

    client = chat_client()
    if not client:
        return {}
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": ANNOTATION_SYSTEM_PROMPT},
            {"role": "user", "content": f"请标注以下分析段落：\n\n{text}"},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content or "{}"
    return json.loads(content)


def save_annotated(blocks: list[dict]) -> Path:
    """保存标注后的知识块。"""
    out_path = config.processed_dir / "annotated_blocks.json"
    out_path.write_text(json.dumps(blocks, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path
