"""转录数据快速入库：跳过标注，直接切分→嵌入→补充到已有 ChromaDB。"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.collector.annotator import extract_reasoning_chains
from src.collector.segmenter import segment_text
from src.config import config
from src.knowledge_base.embedder import Embedder
from src.knowledge_base.vector_store import KnowledgeIndex
from src.knowledge_base.reasoning_index import ReasoningIndex

RAW_DIR = config.raw_dir

print("📂 加载转录文件...")
raw_files = sorted(RAW_DIR.glob("*.json"))

all_blocks = []
for rf in raw_files:
    try:
        data = json.loads(rf.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        continue
    text = (data.get("text") or "").strip()
    if not text:
        continue
    segments = segment_text(text)

    # 清理文件名作为 ID
    stem = rf.stem.replace("#", "_").replace(" ", "_")[:80]
    for i, seg in enumerate(segments):
        all_blocks.append({
            "id": f"trans_{stem}_{i:04d}",
            "source": f"[转录] {rf.stem}",
            "title": rf.stem,
            "chunk_index": i,
            "text": seg["text"],
            "length": seg["length"],
            "is_complete_analysis": seg.get("is_complete", False),
            "annotation": {},  # 无标注
        })

print(f"   切分后: {len(all_blocks)} 个知识块")

print(f"\n🔮 生成向量嵌入 ({len(all_blocks)} 条)...")
embedder = Embedder()
embeddings = embedder.embed([b["text"] for b in all_blocks])

print("💾 写入 ChromaDB（补充模式）...")
kb = KnowledgeIndex()
old_count = kb.collection.count()
kb.add_blocks(all_blocks, embeddings)
print(f"   旧: {old_count} → 新: {kb.collection.count()} 条")

# 推理链（转录数据无标注，但文本本身可做简单推理链）
chains = extract_reasoning_chains(all_blocks)
if chains:
    chain_embs = embedder.embed([c.to_text() for c in chains])
    ri = ReasoningIndex()
    old_rc = ri.collection.count()
    ri.add_chains(chains, chain_embs)
    print(f"   🧠 推理链: 旧 {old_rc} → 新 {ri.collection.count()} 条")

print(f"\n✅ 转录数据补充入库完成")
print(f"   新增: {kb.collection.count() - old_count} 条")
print(f"   总知识索引: {kb.collection.count()} 条")
