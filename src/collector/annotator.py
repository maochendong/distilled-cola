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

    for i, block in enumerate(blocks):
        text = block.get("text", "")
        if not text.strip():
            block["annotation"] = {}
            annotated.append(block)
            continue

        try:
            annotation = annotate_block(text, model=model)
        except Exception as e:
            print(f"  ⚠️ 标注 block {i} 失败: {e}")
            annotation = {}

        block["annotation"] = annotation
        annotated.append(block)

        if (i + 1) % batch_size == 0:
            print(f"    标注进度: {i+1}/{len(blocks)}")

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

        chain_info = ann.get("推理链", {})
        if isinstance(chain_info, dict):
            trigger = chain_info.get("trigger", "").strip()
            conclusion = chain_info.get("conclusion", "").strip()
            key_logic = chain_info.get("key_logic", "").strip()

            if trigger and conclusion:
                dedup_key = trigger[:50]
                if dedup_key not in seen_triggers:
                    seen_triggers.add(dedup_key)
                    rc = ReasoningChain(trigger=trigger, conclusion=conclusion)
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
