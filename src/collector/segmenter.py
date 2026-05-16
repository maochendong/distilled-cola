"""逻辑段落切分 — 将转录文本按分析单元切分。

策略：检测话题转移和结构性线索，将长文本切为有意义的分析段落。
每个段落对应一个完整的「观察 → 分析 → 结论」单元。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.config import config

# 话题转移指示词 — 博主切换到新话题的典型信号
TOPIC_TRANSITIONS = [
    "再来看", "另一方面", "接下来说", "换个角度", "相比之下",
    "这里要重点讲", "还有一个关键", "聊到这个", "说回",
    "从另一个维度", "最后说一下", "先说结论",
]

# 分析结构标记
STRUCTURE_MARKERS = [
    "原因有", "逻辑是", "本质上是", "核心在于", "关键在于",
    "总结一下", "所以我的建议是", "操作上", "具体来说",
]

# 上海板块特殊标记
AREA_MARKERS = [
    "前滩", "大宁", "徐汇滨江", "北外滩", "新江湾", "唐镇",
    "张江", "金桥", "森兰", "华泾", "南大", "桃浦",
    "杨浦滨江", "东外滩", "苏河湾", "中兴城",
]


def detect_topic_boundaries(text: str) -> list[int]:
    """检测话题转移边界，返回切分点索引列表。"""
    boundaries: list[int] = []
    lines = text.split("\n")
    char_offset = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            char_offset += len(line) + 1
            continue

        # 检测话题转移词开头
        if any(stripped.startswith(t) for t in TOPIC_TRANSITIONS):
            boundaries.append(char_offset)
        # 检测结构标记
        elif any(marker in stripped[:20] for marker in STRUCTURE_MARKERS):
            boundaries.append(char_offset)
        # 检测问句（段落以问句开始，说明新角度）
        elif stripped.endswith("？" ) or stripped.endswith("?"):
            boundaries.append(char_offset)

        char_offset += len(line) + 1
    return boundaries


def is_complete_analysis(text: str) -> bool:
    """判断一段文本是否构成一个完整的分析单元（包含结论或建议）。"""
    conclusion_markers = ["建议", "结论", "所以", "因此", "总体", "综合来看"]
    return any(m in text[-100:] for m in conclusion_markers)


def segment_text(
    text: str,
    min_length: int | None = None,
    max_length: int | None = None,
) -> list[dict]:
    """将转录文本按逻辑段落切分。

    Args:
        text: 输入文本（可能是分段或连续）
        min_length: 段落最小字符数
        max_length: 段落最大字符数

    Returns:
        段落列表，每段包含 text、start_idx、end_idx、is_complete 字段
    """
    min_len = min_length or config.segment_min_length
    max_len = max_length or config.segment_max_length

    # 先用话题转移信号切分
    boundaries = detect_topic_boundaries(text)

    # 如果没有检测到边界，按段落（双换行）切分，再按硬长度限制切分
    if not boundaries:
        paragraphs = re.split(r"\n\s*\n", text)
        segments: list[dict] = []
        buffer = ""
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(buffer) + len(para) < max_len:
                buffer += para + "\n"
            else:
                if buffer:
                    segments.append({
                        "text": buffer.strip(),
                        "length": len(buffer.strip()),
                        "is_complete": is_complete_analysis(buffer),
                    })
                buffer = para + "\n"
        if buffer:
            segments.append({
                "text": buffer.strip(),
                "length": len(buffer.strip()),
                "is_complete": is_complete_analysis(buffer),
            })

        # 如果仍有段落超过 max_len，按句子硬切分
        final_segments = []
        for seg in segments:
            if seg["length"] <= max_len:
                final_segments.append(seg)
            else:
                sentences = re.split(r"(?<=[。！？.!?])\s*", seg["text"])
                buf = ""
                for sent in sentences:
                    if not sent.strip():
                        continue
                    if len(buf) + len(sent) < max_len:
                        buf += sent
                    else:
                        if buf:
                            final_segments.append({
                                "text": buf.strip(),
                                "length": len(buf.strip()),
                                "is_complete": is_complete_analysis(buf),
                            })
                        buf = sent
                if buf:
                    final_segments.append({
                        "text": buf.strip(),
                        "length": len(buf.strip()),
                        "is_complete": is_complete_analysis(buf),
                    })
        return final_segments

    # 按边界切分
    spans: list[tuple[int, int]] = []
    prev = 0
    for b in boundaries:
        if b - prev > min_len:
            spans.append((prev, b))
            prev = b
    if len(text) - prev > min_len:
        spans.append((prev, len(text)))

    # 如果没有生成足够长的段，回退到整段
    if not spans:
        return [{
            "text": text.strip(),
            "length": len(text.strip()),
            "is_complete": is_complete_analysis(text),
        }]

    segments = []
    for start, end in spans:
        segment_text = text[start:end].strip()
        if len(segment_text) < min_len and segments:
            # 过短的段落合并到前一段
            segments[-1]["text"] += "\n" + segment_text
            segments[-1]["length"] = len(segments[-1]["text"])
            continue
        segments.append({
            "text": segment_text,
            "length": len(segment_text),
            "is_complete": is_complete_analysis(segment_text),
        })

    return segments


def process_video(video_id: str, text: str, title: str = "") -> list[dict]:
    """处理单个视频的文本，返回带段落信息和标题的知识块。"""
    segments = segment_text(text)
    blocks = []
    for i, seg in enumerate(segments):
        blocks.append({
            "id": f"{video_id}_{i:04d}",
            "source": video_id,
            "title": title,
            "chunk_index": i,
            "text": seg["text"],
            "length": seg["length"],
            "is_complete_analysis": seg["is_complete"],
        })
    return blocks


def save_blocks(blocks: list[dict]) -> Path:
    """保存处理后的知识块。"""
    out_path = config.processed_dir / "knowledge_blocks.json"
    out_path.write_text(json.dumps(blocks, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path
