"""推理链元数据回填迁移 — 为现有 907 条推理链补充板块/逻辑标签。

从知识库中收集已知板块名，用关键词匹配推理链的 trigger/conclusion 文本，
将匹配到的标签写入 ChromaDB metadata，使 _reasoning_search 的预过滤生效。

用法: .venv/bin/python scripts/migrate_reasoning_metadata.py
"""

from __future__ import annotations

from src.knowledge_base.reasoning_index import ReasoningIndex
from src.knowledge_base.vector_store import KnowledgeIndex


def main() -> None:
    ki = KnowledgeIndex()
    ri = ReasoningIndex()

    # 1) 收集知识库中所有板块名（按长度降序，保证最长匹配优先）
    all_data = ki.collection.get(include=["metadatas"])
    areas = set()
    logic_tags = set()
    for m in all_data["metadatas"]:
        if m.get("areas"):
            for a in m["areas"].split(","):
                a = a.strip()
                if len(a) >= 2:
                    areas.add(a)
        if m.get("logic_tags"):
            for t in m["logic_tags"].split(","):
                t = t.strip()
                if t:
                    logic_tags.add(t)

    known_areas = sorted(areas, key=lambda x: (-len(x), x))
    known_logic = sorted(logic_tags)
    print(f"知识库板块标签: {len(known_areas)} 个")
    print(f"知识库逻辑标签: {len(known_logic)} 个")

    # 2) 遍历所有推理链，根据 trigger/conclusion 文本匹配标签
    all_r = ri.collection.get(include=["documents", "metadatas"])
    total = len(all_r["ids"])
    updated = 0
    skipped = 0

    print(f"\n开始迁移 {total} 条推理链...")

    for i in range(total):
        doc_id = all_r["ids"][i]
        meta = all_r["metadatas"][i]

        # 跳过已标注的（areas 非空）
        if meta.get("areas", "").strip():
            skipped += 1
            continue

        trigger = meta.get("trigger", "")
        conclusion = meta.get("conclusion", "")
        text = f"{trigger} {conclusion}"

        # 板块匹配
        matched_areas = []
        for area in known_areas:
            if area in text:
                matched_areas.append(area)

        # 逻辑标签匹配
        matched_logic = []
        for tag in known_logic:
            if tag in text:
                matched_logic.append(tag)

        # 更新 metadata
        new_meta = dict(meta)
        new_meta["areas"] = ",".join(matched_areas[:5])
        new_meta["logic_tags"] = ",".join(matched_logic[:5])
        new_meta["districts"] = ""

        ri.update_metadata(doc_id, new_meta)
        updated += 1

        if (i + 1) % 100 == 0:
            print(f"  进度: {i+1}/{total}", end="")

    print(f"\n迁移完成: {updated} 条已更新, {skipped} 条跳过")


if __name__ == "__main__":
    main()
