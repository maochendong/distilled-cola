"""推理链元数据回填迁移 v2 — 从 annotated_blocks.json 回补 areas/logic_tags。

方案 A：已有标注数据回补。不调 API，纯本地操作。
- 读取 data/processed/annotated_blocks.json（2.5MB，965 条）
- 每条知识块都有 LLM 标注的 areas、logic_tags、districts
- 每条推理链的 trigger 可对回源知识块的 annotation.reasoning_chain.trigger
- 覆盖率从 53%（v1 关键词匹配）提至 ~95%

用法: .venv/bin/python scripts/migrate_reasoning_metadata_v2.py
"""

from __future__ import annotations

import json
from pathlib import Path

from src.collector.annotator import normalize_annotation
from src.knowledge_base.reasoning_index import ReasoningIndex


def main() -> None:
    ri = ReasoningIndex()

    # 1) 读取标注数据
    blocks_path = Path("data/processed/annotated_blocks.json")
    if not blocks_path.exists():
        print(f"❌ 标注数据不存在: {blocks_path}")
        return

    blocks = json.loads(blocks_path.read_text(encoding="utf-8"))
    print(f"📦 加载标注数据: {len(blocks)} 条")

    # 2) 构建 trigger → metadata 映射
    # 每个 block 的 annotation 中有 reasoning_chain.trigger，
    # normalized annotation 中有 areas/logic_tags
    trigger_map = {}
    matched_count = 0
    for block in blocks:
        ann = block.get("annotation", {})
        if not ann:
            continue
        normalized = normalize_annotation(ann)
        areas = normalized.get("areas", [])
        logic_tags = normalized.get("logic_tags", [])
        districts = normalized.get("districts", [])

        # 取 reasoning_chain 中的 trigger 作为匹配 key
        chain_info = normalized.get("reasoning_chain", {})
        if not isinstance(chain_info, dict):
            chain_info = {}

        trigger = (chain_info.get("trigger") or "").strip()
        if not trigger:
            continue

        metadata = {
            "areas": ",".join(areas[:5]) if areas else "",
            "logic_tags": ",".join(logic_tags[:5]) if logic_tags else "",
            "districts": ",".join(districts[:3]) if districts else "",
        }
        # 用 trigger 前 50 字做 key（和 extract_reasoning_chains 的 dedup 逻辑一致）
        dedup_key = trigger[:50]
        if dedup_key not in trigger_map:
            trigger_map[dedup_key] = metadata
            matched_count += 1

    print(f"🔑 构建 trigger 映射: {matched_count} 条")

    # 3) 遍历推理链，用 trigger 匹配回补元数据
    all_r = ri.collection.get(include=["documents", "metadatas"])
    total = len(all_r["ids"])
    updated = 0
    skipped = 0
    no_match = 0

    print(f"\n🚀 开始迁移 {total} 条推理链...")

    for i in range(total):
        doc_id = all_r["ids"][i]
        meta = all_r["metadatas"][i]

        # 如果已有非空元数据，保留（尊重已迁移过的）
        if meta.get("areas", "").strip() and meta.get("logic_tags", "").strip():
            skipped += 1
            continue

        trigger = meta.get("trigger", "")

        # 用 trigger 前 50 字匹配
        match_key = trigger[:50]
        ann_meta = trigger_map.get(match_key)

        if not ann_meta:
            no_match += 1
            continue

        # 更新 metadata，保留原有字段
        new_meta = dict(meta)
        if ann_meta["areas"]:
            new_meta["areas"] = ann_meta["areas"]
        if ann_meta["logic_tags"]:
            new_meta["logic_tags"] = ann_meta["logic_tags"]
        if ann_meta["districts"]:
            new_meta["districts"] = ann_meta["districts"]

        ri.update_metadata(doc_id, new_meta)
        updated += 1

        if (i + 1) % 100 == 0:
            print(f"  进度: {i+1}/{total}")

    print(f"\n✅ 迁移完成:")
    print(f"   更新: {updated} 条")
    print(f"   跳过（已有元数据）: {skipped} 条")
    print(f"   无匹配: {no_match} 条")

    # 4) 统计最终覆盖率
    all_r2 = ri.collection.get(include=["metadatas"])
    has_areas = sum(1 for m in all_r2["metadatas"] if m.get("areas", "").strip())
    has_tags = sum(1 for m in all_r2["metadatas"] if m.get("logic_tags", "").strip())
    print(f"\n📊 最终覆盖率:")
    print(f"   有板块标签: {has_areas}/{total} ({has_areas/total*100:.1f}%)")
    print(f"   有逻辑标签: {has_tags}/{total} ({has_tags/total*100:.1f}%)")


if __name__ == "__main__":
    main()
