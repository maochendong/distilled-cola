"""端到端入库脚本：OCR 结果 → 清洗 → 切分 → 标注 → 嵌入 → ChromaDB + 推理链索引。

用法：
  python scripts/ingest_all.py                     # 全量入库
  python scripts/ingest_all.py --skip-annotation    # 跳过标注（测试用）
  python scripts/ingest_all.py --model large        # 用小模型标注（省钱）
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

# 加入项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.collector.annotator import annotate_batch, extract_reasoning_chains
from src.collector.segmenter import process_video
from src.config import config
from src.knowledge_base.embedder import Embedder
from src.knowledge_base.reasoning_index import ReasoningIndex
from src.knowledge_base.vector_store import KnowledgeIndex

OCR_SUMMARY = Path("data/ocr_results/_summary.json")
PROCESSED_DIR = config.processed_dir
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_and_clean() -> list[dict]:
    """加载 OCR 摘要并过滤空文本。"""
    data: list[dict] = json.loads(OCR_SUMMARY.read_text(encoding="utf-8"))
    non_empty = [d for d in data if d.get("text", "").strip()]
    print(f"📂 OCR 摘要: {len(data)} 条, 非空: {len(non_empty)} 条")
    return non_empty


def segment_all(items: list[dict]) -> list[dict]:
    """将所有 OCR 文本切分为知识块。"""
    all_blocks: list[dict] = []
    for item in items:
        folder = item.get("folder", "unknown")
        text = item.get("text", "").strip()
        if not text:
            continue
        blocks = process_video(folder, text, title=folder)
        all_blocks.extend(blocks)

    print(f"📄 切分完成: {len(items)} 条 → {len(all_blocks)} 个知识块")
    return all_blocks


def annotate_blocks(blocks: list[dict]) -> list[dict]:
    """使用教师模型标注知识块。"""
    print(f"🏷️  开始标注 {len(blocks)} 个知识块 (教师模型: {config.teacher_model})...")
    start = time.time()
    annotated = annotate_batch(blocks)
    elapsed = time.time() - start
    print(f"   ✅ 标注完成: {elapsed:.1f}s ({elapsed/len(annotated):.1f}s/块 平均)")
    return annotated


def embed_blocks(blocks: list[dict]) -> list[list[float]]:
    """生成知识块的向量嵌入。"""
    print(f"🔮 生成向量嵌入 ({len(blocks)} 条)...")
    embedder = Embedder()
    texts = [b["text"] for b in blocks]
    start = time.time()
    embeddings = embedder.embed(texts)
    elapsed = time.time() - start
    print(f"   ✅ 嵌入完成: {elapsed:.1f}s ({(elapsed/len(texts))*1000:.0f}ms/条)")
    return embeddings


def ingest_to_knowledge_index(blocks: list[dict], embeddings: list[list[float]]) -> KnowledgeIndex:
    """写入 ChromaDB 知识索引。"""
    print(f"💾 写入知识索引...")
    kb = KnowledgeIndex(embed_dim=len(embeddings[0]) if embeddings else 1024)

    # 需要先清空旧索引（维度变更时重建）
    old_count = kb.collection.count()
    if old_count > 0:
        print(f"   ⚠️ 旧索引有 {old_count} 条数据, 需要重建")
        kb.delete()
        print("   🗑️  旧索引已清空")

    kb.add_blocks(blocks, embeddings)
    stats = kb.stats()
    print(f"   ✅ 知识索引: {stats['count']} 条")
    return kb


def extract_and_ingest_reasoning_chains(blocks: list[dict], embeddings: list[list[float]]) -> ReasoningIndex:
    """从标注结果中提取推理链，写入 ReasoningIndex。"""
    chains = extract_reasoning_chains(blocks)

    if not chains:
        print("   ⚠️ 未提取到推理链（标注结果中无 trigger/conclusion 字段）")
        return ReasoningIndex(embed_dim=1024)

    # 复用对应知识块的向量作为推理链向量
    block_texts = [b["text"] for b in blocks]
    chain_embeddings = []
    for chain in chains:
        # 找到第一个 trigger 匹配的 block
        for i, b in enumerate(blocks):
            ann = b.get("annotation", {})
            rc = ann.get("推理链", {})
            if isinstance(rc, dict) and rc.get("trigger", "")[:50] == chain.trigger[:50]:
                chain_embeddings.append(embeddings[i])
                break
        else:
            chain_embeddings.append(np.mean(embeddings, axis=0).tolist())

    ri = ReasoningIndex(embed_dim=len(chain_embeddings[0]))
    ri.add_chains(chains, chain_embeddings)

    print(f"   🧠 推理链写入: {len(chains)} 条")
    return ri


def save_artifacts(blocks: list[dict]) -> None:
    """保存中间产物供检查和调试。"""
    # 知识块（含标注）
    blocks_path = PROCESSED_DIR / "knowledge_blocks.json"
    blocks_path.write_text(json.dumps(blocks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"   💾 知识块已保存: {blocks_path}")

    # 纯文本列表（用于快速检查）
    texts_path = PROCESSED_DIR / "all_texts.txt"
    texts_path.write_text(
        "\n\n=====\n\n".join(b["text"] for b in blocks),
        encoding="utf-8",
    )
    print(f"   💾 纯文本已保存: {texts_path} ({blocks_path.stat().st_size / 1024:.0f} KB)")


def main():
    skip_annotation = "--skip-annotation" in sys.argv

    # Step 1: 加载并清洗
    items = load_and_clean()

    # Step 2: 切分
    blocks = segment_all(items)
    save_artifacts(blocks)

    # Step 3: 标注
    if skip_annotation:
        print("⏭️  跳过标注 (--skip-annotation)")
        # 给每个 block 一个空标记
        for b in blocks:
            b["annotation"] = {}
    else:
        blocks = annotate_blocks(blocks)
        # 重新保存（含标注）
        save_artifacts(blocks)

    # Step 4: 嵌入
    embeddings = embed_blocks(blocks)

    # Step 5: 写入知识索引
    kb = ingest_to_knowledge_index(blocks, embeddings)

    # Step 6: 提取推理链
    if not skip_annotation:
        ri = extract_and_ingest_reasoning_chains(blocks, embeddings)
    else:
        print("⏭️  跳过推理链提取（未标注）")
        ri = ReasoningIndex(embed_dim=len(embeddings[0]))

    # Step 7: 摘要
    total_chars = sum(b["length"] for b in blocks)
    print()
    print(f"📊 ======== 入库摘要 ========")
    print(f"   来源视频/图片:   {len(items)}")
    print(f"   知识块:          {len(blocks)}")
    print(f"   总字数:          {total_chars:,}")
    print(f"   知识索引:        {kb.stats()['count']} 条")
    print(f"   推理链索引:      {ri.stats()['count']} 条")
    print(f"   ========================")


if __name__ == "__main__":
    main()
