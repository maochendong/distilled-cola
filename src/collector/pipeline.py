"""采集管道 — 端到端的视频采集 → 转录+OCR → 切分 → 标注流程。

数据流：
  视频文件 → [音频转录] ─┐
              → [帧提取+OCR] ─┼→ 合并文本 → 逻辑切分 → 房产标注 → 知识库
              → [说话人分离] ─┘

支持数据源：
  - 本地视频文件（微信视频号/小红书/抖音等下载后）
  - 本地图片文件（博主发的截图/数据表）
  - YouTube（通过字幕 API）
"""

from __future__ import annotations

from pathlib import Path

from src.collector.annotator import annotate_batch
from src.collector.frame_extractor import FrameExtractor
from src.collector.ocr import batch_ocr
from src.collector.segmenter import process_video
from src.collector.transcriber import Transcriber


def process_local_video(
    video_path: str,
    title: str = "",
    video_id: str | None = None,
    enable_ocr: bool = True,
) -> list[dict]:
    """处理本地视频文件：转录 + 帧OCR + 切分 + 标注。

    适用于微信视频号/小红书等下载的视频。
    """
    print(f"🎬 处理本地视频: {Path(video_path).name}")
    if title:
        print(f"   📺 {title}")

    # Step 1: 音频转录
    t = Transcriber()
    segments, source_id = t.from_video_file(video_path, title=title)
    transcript_text = " ".join(s.text for s in segments)
    print(f"   🎤 音频转录: {len(transcript_text)} 字")

    # Step 2: 帧提取 + OCR
    ocr_text = ""
    if enable_ocr:
        extractor = FrameExtractor(interval_sec=8.0)
        frames = extractor.extract(video_path)

        if frames:
            frame_paths = [f["path"] for f in frames]
            ocr_results = batch_ocr(frame_paths)

            # 收集 OCR 文字（含结构化分析）
            ocr_parts = []
            for r in ocr_results:
                if r["raw_text"]:
                    text = r["raw_text"]
                    if r.get("structured", {}).get("analysis"):
                        text += f"\n[图表解读] {r['structured']['analysis']}"
                    ocr_parts.append(text)

            if ocr_parts:
                ocr_text = "\n\n--- [画面文字] ---\n\n".join(ocr_parts)
                print(f"   👁️ OCR 提取: {len(ocr_text)} 字")

    # Step 3: 合并转录 + OCR
    if ocr_text:
        combined_text = (
            f"{transcript_text}\n\n"
            f"【以下为视频画面中提取的文字内容，补充了口播中可能未覆盖的信息】\n"
            f"{ocr_text}"
        )
    else:
        combined_text = transcript_text

    vid = video_id or source_id
    return _process_and_annotate(vid, combined_text, title=title or vid)


def process_local_image(
    image_path: str,
    title: str = "",
) -> list[dict]:
    """处理单张图片（博主分享的数据截图/表格/政策文件等）。

    跳过音频转录，直接 OCR + 结构化提取 → 入库。
    """
    print(f"🖼️ 处理图片: {Path(image_path).name}")
    from src.collector.ocr import OCREngine

    engine = OCREngine()
    result = engine.extract_structured(image_path)

    text_parts = [f"[图片: {Path(image_path).name}]"]
    if result["raw_text"]:
        text_parts.append(result["raw_text"])
    if result.get("structured", {}).get("analysis"):
        text_parts.append(f"[分析] {result['structured']['analysis']}")

    combined = "\n\n".join(text_parts)
    print(f"   👁️ 提取了 {len(combined)} 字内容")

    source_id = title or Path(image_path).stem
    return _process_and_annotate(source_id, combined, title=title or source_id)


def process_youtube_video(video_id: str, title: str = "") -> list[dict]:
    """处理 YouTube 视频（通过字幕 API，无 OCR 支持）。"""
    print(f"🎬 获取 YouTube 字幕: {video_id}")
    t = Transcriber()
    segments, _ = t.from_youtube_subtitles(video_id)
    text = " ".join(s.text for s in segments)
    print(f"   📝 获取完成: {len(text)} 字")

    return _process_and_annotate(video_id, text, title=title or video_id)


def _process_and_annotate(video_id: str, text: str, title: str = "") -> list[dict]:
    """流水线后段：切分 → 标注。"""
    print(f"📄 切分为逻辑段落...")
    blocks = process_video(video_id, text, title=title)
    print(f"   → {len(blocks)} 个段落")

    print(f"🏷️ 房产领域标注...")
    annotated = annotate_batch(blocks)
    print(f"   ✅ 标注完成")

    return annotated
