"""OCR/转录数据 → 知识库 蒸馏入库脚本。

将 _summary.json 中的文本数据走完完整蒸馏链路：
  数据 → 切分 → 教师模型标注 → 推理链提取 → 向量化 → ChromaDB

用法:
  # 首次：清空旧索引后完整蒸馏
  python scripts/ingest_knowledge.py --reset

  # 仅 OCR 数据（等待 mlx-whisper 时的先行步骤）
  python scripts/ingest_knowledge.py --source ocr

  # 转录+OCR 融合数据（mlx-whisper 跑完后使用）
  python scripts/ingest_knowledge.py --source hybrid

  # 只标注+入库，跳过切分（已有切分结果时）
  python scripts/ingest_knowledge.py --skip-segment
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.collector.segmenter import segment_text
from src.collector.annotator import annotate_batch, extract_reasoning_chains
from src.knowledge_base.embedder import Embedder
from src.knowledge_base.vector_store import KnowledgeIndex
from src.knowledge_base.reasoning_index import ReasoningIndex

OCR_DIR = config.data_dir / "ocr_results"
RAW_DIR = config.raw_dir
SUMMARY_FILE = OCR_DIR / "_summary.json"


def numeric_key(name: str) -> list[int]:
    return [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", name)]


def load_ocr_data() -> list[dict]:
    """从 _summary.json 加载 OCR 结果。"""
    if not SUMMARY_FILE.exists():
        print(f"❌ 未找到OCR汇总: {SUMMARY_FILE}")
        sys.exit(1)
    data = json.loads(SUMMARY_FILE.read_text(encoding="utf-8"))
    # 过滤空文本
    valid = [d for d in data if d.get("text", "").strip()]
    skipped = len(data) - len(valid)
    if skipped:
        print(f"  跳过 {skipped} 条空文本")
    return valid


def load_hybrid_data() -> list[dict]:
    """融合 OCR 和 Whisper 转录数据。

    将 data/raw/ 中的转录结果与 OCR 文本合并，
    格式与 load_ocr_data 兼容（含 folder, text 字段）。
    """
    ocr_items = {d["folder"]: d for d in load_ocr_data()}

    raw_files = sorted(RAW_DIR.glob("*.json"), key=lambda f: numeric_key(f.stem))
    hybrid = []

    for rf in raw_files:
        try:
            trans = json.loads(rf.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        transcript_text = trans.get("text", "").strip()
        video_name = rf.stem

        # OCR: 根据视频文件名匹配对应的 OCR 文件夹
        ocr_text = ""
        for folder, item in ocr_items.items():
            if video_name in folder or folder in video_name:
                ocr_text = item.get("text", "").strip()
                break

        # 合并
        if transcript_text and ocr_text:
            combined = (
                f"{transcript_text}\n\n"
                f"【以下为视频画面文字补充】\n{ocr_text}"
            )
        elif transcript_text:
            combined = transcript_text
        elif ocr_text:
            combined = ocr_text
        else:
            continue

        hybrid.append({
            "folder": video_name,
            "text": combined,
        })

    print(f"  融合转录+OCR: {len(hybrid)} 条")
    return hybrid


def build_knowledge(
    source_data: list[dict],
    skip_segment: bool = False,
    batch_size: int = 5,
) -> None:
    """完整蒸馏流程：切分 → 标注 → 嵌入 → 写入。"""
    embedder = Embedder()
    knowledge_idx = KnowledgeIndex()
    reasoning_idx = ReasoningIndex()

    total_blocks = 0
    total_chains = 0

    for item in source_data:
        folder = item.get("folder", "unknown")
        text = item.get("text", "")
        if not text.strip():
            continue

        print(f"\n📄 {folder[:60]}")

        # Step 1: 切分
        if skip_segment:
            blocks = [{
                "id": re.sub(r"[^a-zA-Z0-9_一-鿿]", "_", folder)[:60],
                "source": folder,
                "text": text,
                "length": len(text),
                "is_complete_analysis": True,
            }]
        else:
            segments = segment_text(text)
            blocks = []
            for i, seg in enumerate(segments):
                safe_id = re.sub(r"[^a-zA-Z0-9_一-鿿]", "_", folder)[:50]
                blocks.append({
                    "id": f"{safe_id}_{i:04d}",
                    "source": folder,
                    "text": seg["text"],
                    "length": seg["length"],
                    "is_complete_analysis": seg.get("is_complete", False),
                })

        if not blocks:
            print("  ➖ 切分为空，跳过")
            continue

        print(f"  切分: {len(blocks)} 段")

        # Step 2: 教师模型标注
        blocks = annotate_batch(blocks, batch_size=batch_size)

        # Step 3: 提取推理链
        chains = extract_reasoning_chains(blocks)
        if chains:
            print(f"  推理链: {len(chains)} 条")

        # Step 4: 向量化知识块
        texts = [b["text"] for b in blocks if b["text"].strip()]
        if not texts:
            continue

        try:
            embeddings = embedder.embed(texts)
        except Exception as e:
            print(f"  ⚠️ 向量化失败: {e}")
            continue

        # Step 5: 写入知识索引
        valid_blocks = [b for b in blocks if b["text"].strip()]
        knowledge_idx.add_blocks(valid_blocks, embeddings)
        total_blocks += len(valid_blocks)

        # Step 6: 写入推理链索引
        if chains:
            try:
                chain_texts = [c.to_text() for c in chains]
                chain_embs = embedder.embed(chain_texts)
                reasoning_idx.add_chains(chains, chain_embs)
                total_chains += len(chains)
            except Exception as e:
                print(f"  ⚠️ 推理链入库失败: {e}")

    # 汇总
    print(f"\n{'='*40}")
    print(f"✅ 蒸馏入库完成")
    print(f"   知识块: {total_blocks} 条")
    print(f"   推理链: {total_chains} 条")
    print(f"   向量维度: 1024 (bge-m3)")
    print(f"   存储: {config.chroma_db_path}")


def main():
    parser = argparse.ArgumentParser(description="OCR/转录数据 → 知识库 蒸馏入库")
    parser.add_argument(
        "--source", choices=["ocr", "hybrid"], default="ocr",
        help="数据来源: ocr (仅画面文字) / hybrid (融合转录+OCR)",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="清空已有索引（升级向量维度时必须使用）",
    )
    parser.add_argument(
        "--skip-segment", action="store_true",
        help="跳过切分，整段入库（已有切分结果时使用）",
    )
    parser.add_argument(
        "--batch-size", type=int, default=5,
        help="教师模型标注批大小",
    )
    args = parser.parse_args()


    print(f"{'='*40}")
    print(f"🧪 蒸馏入库  --source={args.source}")
    print(f"{'='*40}")

    # 加载数据
    if args.source == "hybrid":
        source_data = load_hybrid_data()
    else:
        source_data = load_ocr_data()

    if not source_data:
        print("❌ 无有效数据可入库")
        sys.exit(1)

    print(f"   待处理: {len(source_data)} 条\n")

    # 清空旧索引（升级向量维度后必须重置）
    if args.reset:
        print("  清空旧索引...")
        try:
            ki = KnowledgeIndex()
            ki.delete()
            print("  ✅ 知识索引已清空")
        except Exception:
            print("  ⚠️ 知识索引清空失败（可能不存在）")
        try:
            ri = ReasoningIndex()
            ri.delete()
            print("  ✅ 推理链索引已清空")
        except Exception:
            print("  ⚠️ 推理链索引清空失败（可能不存在）")
        print()

    # 执行蒸馏
    build_knowledge(
        source_data,
        skip_segment=args.skip_segment,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
