"""推理链重建脚本 — 从当前的 annotated_blocks.json 重新提取并索引推理链。

现状：推理链索引（907 条）与当前 annotated_blocks.json 来自不同版本，
      trigger 文本对不上，导致方案 A 回补失败。

解决：删除旧索引 → 重新 extract_reasoning_chains → 重新嵌入 → 写入新索引。
     新链自带 areas/logic_tags 元数据，覆盖率 ~63%。

用法: .venv/bin/python scripts/rebuild_reasoning_index.py
"""

from __future__ import annotations

import json
from pathlib import Path

from src.collector.annotator import extract_reasoning_chains
from src.knowledge_base.embedder import Embedder
from src.knowledge_base.reasoning_index import ReasoningIndex


def main() -> None:
    blocks_path = Path("data/processed/annotated_blocks.json")
    if not blocks_path.exists():
        print(f"❌ 标注数据不存在: {blocks_path}")
        return

    blocks = json.loads(blocks_path.read_text(encoding="utf-8"))
    print(f"📦 加载标注数据: {len(blocks)} 条")

    # 1) 提取推理链（自带 areas/logic_tags）
    chains = extract_reasoning_chains(blocks)
    print(f"🧠 提取推理链: {len(chains)} 条")

    # 统计元数据覆盖率
    with_areas = sum(1 for c in chains if c.areas)
    with_tags = sum(1 for c in chains if c.logic_tags)
    print(f"   有板块标签: {with_areas} ({with_areas/len(chains)*100:.1f}%)")
    print(f"   有逻辑标签: {with_tags} ({with_tags/len(chains)*100:.1f}%)")

    # 2) 生成嵌入
    embedder = Embedder()
    texts = [c.to_text() for c in chains]
    print(f"\n🔮 生成 {len(texts)} 条嵌入向量...")
    embeddings = embedder.embed(texts)
    print(f"   嵌入完成")

    # 3) 删除旧索引，写入新索引
    ri = ReasoningIndex()
    old_count = ri.collection.count()
    print(f"\n🗑️  删除旧索引 ({old_count} 条)...")
    ri.delete()

    # 重新初始化（delete 后 collection 被重建）
    ri2 = ReasoningIndex()
    ri2.add_chains(chains, embeddings)

    # 4) 验证
    final = ri2.collection.count()
    final_meta = ri2.collection.get(include=["metadatas"])
    has_areas = sum(1 for m in final_meta["metadatas"] if m.get("areas", "").strip())
    has_tags = sum(1 for m in final_meta["metadatas"] if m.get("logic_tags", "").strip())
    print(f"\n✅ 重建完成:")
    print(f"   索引条数: {final}")
    print(f"   有板块标签: {has_areas}/{final} ({has_areas/final*100:.1f}%)")
    print(f"   有逻辑标签: {has_tags}/{final} ({has_tags/final*100:.1f}%)")


if __name__ == "__main__":
    main()
