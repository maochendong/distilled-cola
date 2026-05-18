"""转录数据补充入库：将 Whisper 转录文本切分→标注→嵌入→补充到已有 ChromaDB。

OCR 数据已入库，转录数据（data/raw/）单独走完整流程后追加到同一索引。

用法:
  python scripts/ingest_transcriptions.py                         # 全量
  python scripts/ingest_transcriptions.py --batch-size 5          # 调标注批大小
  python scripts/ingest_transcriptions.py --skip-annotate         # 跳过标注（测试用）
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.collector.annotator import annotate_batch, extract_reasoning_chains
from src.collector.segmenter import segment_text
from src.config import config
from src.knowledge_base.embedder import Embedder
from src.knowledge_base.reasoning_index import ReasoningIndex
from src.knowledge_base.vector_store import KnowledgeIndex

RAW_DIR = config.raw_dir
CHECKPOINT_PATH = config.chroma_db_path.parent / "transcribe_checkpoint.json"
BATCH_SIZE = 20  # 嵌入 batch

_current_done: list[str] = []


def _signal_handler(signum: int, frame) -> None:
    sig_name = signal.Signals(signum).name
    print(f"\n⚠️  收到 {sig_name}，保存检查点后退出...")
    if _current_done:
        _save_checkpoint(_current_done)
        print(f"  💾 检查点已保存 ({len(_current_done)} 条)")
    print("   重新运行将从中断处继续")
    sys.exit(0)


def _load_checkpoint() -> set[str]:
    if CHECKPOINT_PATH.exists():
        ckpt = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        done = set(ckpt.get("done_ids", []))
        print(f"  📌 检查点: {len(done)} 个文件已处理")
        return done
    return set()


def _save_checkpoint(done_ids: list[str]) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {"done_ids": done_ids, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    tmp = CHECKPOINT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(ckpt, ensure_ascii=False), encoding="utf-8")
    tmp.replace(CHECKPOINT_PATH)


def _cleanup_checkpoint() -> None:
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        print("  🧹 检查点已清除")


def main() -> None:
    parser = argparse.ArgumentParser(description="转录数据补充入库")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="教师模型标注批大小 (默认 5)")
    parser.add_argument("--skip-annotate", action="store_true",
                        help="跳过标注（测试用）")
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    # ── 1. 加载转录文件 ──
    raw_files = sorted(RAW_DIR.glob("*.json"))
    print(f"📂 转录文件: {len(raw_files)} 个")

    # ── 2. 检查检查点 ──
    done_ids = _load_checkpoint()
    remaining = [f for f in raw_files if f.stem not in done_ids]
    if not remaining:
        print("\n✅ 所有转录文件已处理")
        return

    print(f"⏳ 待处理: {len(remaining)} 个文件\n")

    # ── 3. 加载并切分 ──
    print("🔪 切分转录文本...")
    all_blocks: list[dict] = []
    for rf in remaining:
        try:
            data = json.loads(rf.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"  ⚠️ 跳过无效 JSON: {rf.name}")
            continue
        text = (data.get("text") or "").strip()
        if not text:
            continue
        segments = segment_text(text)
        safe_id = rf.stem.replace("#", "_").replace(" ", "_")[:80]
        for i, seg in enumerate(segments):
            all_blocks.append({
                "id": f"trans_{safe_id}_{i:04d}",
                "source": rf.stem,
                "title": rf.stem,
                "chunk_index": i,
                "text": seg["text"],
                "length": seg["length"],
                "is_complete_analysis": seg.get("is_complete", False),
            })

    if not all_blocks:
        print("  无有效转录文本可处理")
        return

    print(f"  切分后: {len(all_blocks)} 个知识块\n")

    # ── 4. 标注 ──
    if args.skip_annotate:
        print("⏭️  跳过标注 (--skip-annotate)")
        for b in all_blocks:
            b["annotation"] = {}
    else:
        print(f"🏷️  标注 {len(all_blocks)} 块 (教师模型: {config.teacher_model}, 批大小: {args.batch_size})...")
        all_blocks = annotate_batch(all_blocks, batch_size=args.batch_size)

    # ── 5. 嵌入 + 入库 ──
    print(f"\n🔮 生成向量嵌入 ({len(all_blocks)} 条)...")
    embedder = Embedder()
    texts = [b["text"] for b in all_blocks]
    try:
        embeddings = embedder.embed(texts)
    except Exception as e:
        print(f"❌ 嵌入失败: {e}")
        sys.exit(1)

    print("💾 写入 ChromaDB（补充模式，保留已有数据）...")
    knowledge_idx = KnowledgeIndex()
    old_count = knowledge_idx.collection.count()
    knowledge_idx.add_blocks(all_blocks, embeddings)
    new_count = knowledge_idx.collection.count()
    added = new_count - old_count
    print(f"   旧: {old_count} → 新: {new_count} (+{added})")

    # Step 6: 推理链
    chains = extract_reasoning_chains(all_blocks)
    if chains:
        chain_texts = [c.to_text() for c in chains]
        chain_embs = embedder.embed(chain_texts)
        reasoning_idx = ReasoningIndex()
        reasoning_idx.add_chains(chains, chain_embs)
        print(f"   🧠 推理链: {len(chains)} 条")

    # 更新检查点
    processed_stems = [f.stem for f in remaining]
    global _current_done
    _current_done = processed_stems
    _save_checkpoint(list(done_ids | set(processed_stems)))

    # 清理（所有 file stem 都 done 时）
    all_done = done_ids | set(processed_stems)
    if all_done == {f.stem for f in raw_files}:
        _cleanup_checkpoint()

    print(f"\n✅ 转录数据补充入库完成")
    print(f"   新增转录块: {added}")
    print(f"   总知识索引: {knowledge_idx.collection.count()} 条")


if __name__ == "__main__":
    main()
