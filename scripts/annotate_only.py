"""仅运行标注阶段 — 单独执行，与嵌入/入库解耦。"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.collector.annotator import annotate_batch

BLOCKS_PATH = Path("data/processed/knowledge_blocks.json")
OUT_PATH = Path("data/processed/annotated_blocks.json")

def main():
    print(f"📂 加载知识块: {BLOCKS_PATH}")
    blocks: list[dict] = json.loads(BLOCKS_PATH.read_text(encoding="utf-8"))
    print(f"   共 {len(blocks)} 块")

    print(f"🏷️  开始标注...  (教师模型: deepseek-v4-pro)")
    start = time.time()
    annotated = annotate_batch(blocks)
    elapsed = time.time() - start

    ann_count = sum(1 for b in annotated if b.get("annotation"))
    print(f"   ✅ 标注完成: {ann_count}/{len(annotated)} 块, {elapsed:.1f}s")

    print(f"💾 保存到 {OUT_PATH}")
    OUT_PATH.write_text(json.dumps(annotated, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"   ✅ 已保存 ({OUT_PATH.stat().st_size / 1024:.0f} KB)")

if __name__ == "__main__":
    main()
