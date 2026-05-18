"""嵌入 + 入库：读取已标注的知识块，生成向量并写入 ChromaDB + 推理链索引。

支持断点续跑，崩溃后重新运行会自动从中断的 batch 继续。

用法:
  python scripts/embed_and_ingest.py                        # 完整入库
  python scripts/embed_and_ingest.py --batch-size 100       # 调大 batch 加快速度
  python scripts/embed_and_ingest.py --force                # 清空旧索引后重建
  python scripts/embed_and_ingest.py --dry-run              # 只看统计不执行

依赖配置:
  .env 中 EMBEDDING_MODEL 控制嵌入方式：
    - 留空 / bge-m3 → 本地 sentence-transformers (默认)
    - text-embedding-3-small → OpenAI API（需配置 OPENAI_API_KEY）
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

from src.config import config
from src.knowledge_base.embedder import Embedder
from src.knowledge_base.reasoning_index import ReasoningIndex
from src.knowledge_base.vector_store import KnowledgeIndex
from src.collector.annotator import extract_reasoning_chains

ANNOTATED_PATH = config.processed_dir / "knowledge_blocks.json"
CHECKPOINT_PATH = config.chroma_db_path.parent / "embed_checkpoint.json"

# 当前保存的 done_ids（供信号处理器使用）
_current_done_ids: list[str] = []


def _signal_handler(signum: int, frame) -> None:
    """收到终止信号时保存检查点后退出，避免丢失已完成 batch。"""
    sig_name = signal.Signals(signum).name
    print(f"\n⚠️  收到 {sig_name}，保存检查点后退出...")
    if _current_done_ids:
        save_checkpoint(_current_done_ids)
        print(f"  💾 检查点已保存 ({len(_current_done_ids)} 条)")
    print("   重新运行将从中断处继续")
    sys.exit(0)


def load_blocks() -> tuple[list[dict], int]:
    """加载已标注知识块并统计。"""
    if not ANNOTATED_PATH.exists():
        print(f"❌ 未找到标注数据: {ANNOTATED_PATH}")
        print("   请先运行标注流程生成 knowledge_blocks.json")
        sys.exit(1)

    blocks: list[dict] = json.loads(ANNOTATED_PATH.read_text(encoding="utf-8"))
    ann_count = sum(1 for b in blocks if b.get("annotation"))
    print(f"📂 标注知识块: {len(blocks)} 条, 含标注: {ann_count}")
    return blocks, ann_count


def load_checkpoint() -> set[str]:
    """读取断点续跑记录。"""
    if CHECKPOINT_PATH.exists():
        ckpt = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        done = set(ckpt.get("done_ids", []))
        print(f"  📌 检查点: {len(done)} 块已处理 ({CHECKPOINT_PATH})")
        return done
    return set()


def save_checkpoint(done_ids: list[str]) -> None:
    """保存断点续跑记录。"""
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {"done_ids": done_ids, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    tmp = CHECKPOINT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(ckpt, ensure_ascii=False), encoding="utf-8")
    tmp.replace(CHECKPOINT_PATH)


def get_chromadb_ids() -> set[str]:
    """读取已在 ChromaDB 中的 ID 列表。"""
    try:
        kb = KnowledgeIndex()
        return set(kb.collection.get(include=[])["ids"])
    except Exception:
        return set()


def cleanup_checkpoint() -> None:
    """入库全部完成后删除检查点。"""
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        print("  🧹 检查点已清除")


def print_progress(done: int, total: int, start: float, batch_i: int, num_batches: int) -> None:
    """打印 batch 进度和时间估算。"""
    elapsed = time.time() - start
    rate = elapsed / max(done, 1)
    remaining = (total - done) * rate

    def fmt(secs: float) -> str:
        h, r = divmod(int(secs), 3600)
        m, s = divmod(r, 60)
        if h:
            return f"{h}h{m:02d}m"
        return f"{m}m{s:02d}s"

    print(f"\n📊 批次 {batch_i}/{num_batches}  |  "
          f"进度 {done}/{total} ({done * 100 // total}%)  |  "
          f"已用 {fmt(elapsed)}  |  预计剩余 {fmt(remaining)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="embed_and_ingest: 知识块嵌入 → ChromaDB")
    parser.add_argument(
        "--batch-size", type=int, default=20,
        help="每批处理的知识块数量 (默认 20，M1 16GB 建议 ≤30)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="清空已有 ChromaDB 索引后重建",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="仅统计，不执行嵌入和入库",
    )
    args = parser.parse_args()

    # 注册信号处理器（SIGTERM=OOM killer, SIGINT=Ctrl+C）
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    # ── 1. 加载数据 ──
    blocks, ann_count = load_blocks()
    total = len(blocks)

    if ann_count == 0:
        print("⚠️  所有知识块均无标注信息，推理链将为空")
    print()

    if args.dry_run:
        print("🏁 --dry-run 模式，不做实际嵌入")
        chars = sum(b.get("length", 0) for b in blocks)
        empty = sum(1 for b in blocks if not b.get("text", "").strip())
        print(f"   总文本量: {chars:,} 字符")
        print(f"   空文本: {empty} 条")
        kb_model = "bge-m3 (本地)" if not config.openai_api_key else f"OpenAI {config.embedding_model}"
        print(f"   嵌入模型: {kb_model}")
        print(f"   数据路径: {ANNOTATED_PATH}")
        print(f"   ChromaDB: {config.chroma_db_path}")
        return

    # ── 2. 确定已处理范围 ──
    if args.force:
        print("🗑️  --force: 清空旧索引...")
        KnowledgeIndex().delete()
        ReasoningIndex().delete()
        print("   旧索引已清空")
        processed_ids: set[str] = set()
    else:
        ckpt_ids = load_checkpoint()
        chroma_ids = get_chromadb_ids()
        processed_ids = ckpt_ids | chroma_ids
        overlap = ckpt_ids & chroma_ids
        if chroma_ids:
            print(f"  📀 ChromaDB 现有: {len(chroma_ids)} 条")
        if overlap:
            print(f"  🔗 检查点与 ChromaDB 重叠: {len(overlap)} 条")

    # 筛选待处理块
    remaining = [b for b in blocks if b["id"] not in processed_ids]
    if not remaining:
        print("\n✅ 所有知识块已入库，无需处理")
        cleanup_checkpoint()
        return

    print(f"\n⏳ 待处理: {len(remaining)} / {total} 块\n")

    # ── 3. 分批嵌入 + 入库 ──
    embedder = Embedder()
    knowledge_idx = KnowledgeIndex()
    reasoning_idx = ReasoningIndex()

    batch_size = args.batch_size
    num_batches = (len(remaining) + batch_size - 1) // batch_size
    start_time = time.time()

    # 持续追踪所有已完成的 ID，包括之前已处理的
    all_done_ids: list[str] = [b["id"] for b in blocks if b["id"] in processed_ids]
    global _current_done_ids
    _current_done_ids = all_done_ids

    for batch_i in range(num_batches):
        batch_start = batch_i * batch_size
        batch_end = min(batch_start + batch_size, len(remaining))
        batch_blocks = remaining[batch_start:batch_end]
        done_so_far = len(all_done_ids) + batch_start

        print_progress(done_so_far, total, start_time, batch_i + 1, num_batches)

        # Step A: 向量化
        batch_texts = [b["text"] for b in batch_blocks]
        try:
            embeddings = embedder.embed(batch_texts)
        except Exception as e:
            print(f"\n❌ 嵌入失败 (batch {batch_i + 1}): {e}")
            print("   检查模型加载和文本内容，修复后重新运行即可从中断处继续")
            save_checkpoint(all_done_ids)
            sys.exit(1)

        # Step B: 写入知识索引
        try:
            knowledge_idx.add_blocks(batch_blocks, embeddings)
        except Exception as e:
            print(f"\n❌ ChromaDB 写入失败 (batch {batch_i + 1}): {e}")
            save_checkpoint(all_done_ids)
            sys.exit(1)

        # Step C: 提取推理链并写入
        batch_chains = extract_reasoning_chains(batch_blocks)
        if batch_chains:
            try:
                chain_texts = [c.to_text() for c in batch_chains]
                chain_embs = embedder.embed(chain_texts)
                reasoning_idx.add_chains(batch_chains, chain_embs)
            except Exception as e:
                print(f"  ⚠️ 推理链入库失败 (batch {batch_i + 1}): {e}")
                # 不退出，知识块已写入，推理链不是关键路径

        # Step D: 更新检查点
        batch_ids = [b["id"] for b in batch_blocks]
        all_done_ids.extend(batch_ids)
        _current_done_ids = all_done_ids
        save_checkpoint(all_done_ids)

        # Step E: 清理 MPS 缓存，防止多 batch 显存堆积导致 OOM
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except ImportError:
            pass

    # ── 4. 摘要 ──
    elapsed = time.time() - start_time
    hrs = int(elapsed // 3600)
    mins = int((elapsed % 3600) // 60)
    secs = int(elapsed % 60)

    print(f"\n{'=' * 42}")
    print(f"✅ 嵌入入库完成")
    print(f"   总知识块:       {total}")
    print(f"   知识索引:       {knowledge_idx.stats()['count']} 条")
    print(f"   推理链索引:     {reasoning_idx.stats()['count']} 条")
    kb_sources = knowledge_idx.stats().get("sources", [])
    print(f"   来源视频/图片:  {len(kb_sources)}")
    if elapsed > 60:
        print(f"   耗时:           {hrs}h{mins:02d}m{secs:02d}s")
    else:
        print(f"   耗时:           {elapsed:.1f}s")
    print(f"   存储:           {config.chroma_db_path}")

    cleanup_checkpoint()


if __name__ == "__main__":
    main()
