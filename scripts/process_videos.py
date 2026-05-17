"""批量处理 Video/ 下所有 mp4 文件。

流程：字幕流提取 → OpenCV 抽帧 → Swift OCR → 合并保存
每个 mp4 = 一个独立问答主题
"""

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2

VIDEO_DIR = Path(__file__).parent.parent / "Video"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "processed_videos"
OCR_TOOL = Path(__file__).parent / "ocr_tool"

FRAME_INTERVAL = 2  # 秒
JPEG_QUALITY = 80


def numeric_key(name: str) -> list[int]:
    return [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", name)]


def extract_subtitles(video_path: str) -> str:
    """提取内嵌字幕流。"""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "s",
             "-show_entries", "stream=index", "-of", "csv=p=0", video_path],
            capture_output=True, text=True, timeout=30,
        )
        streams = [s.strip() for s in result.stdout.strip().split("\n") if s.strip()]
        if not streams:
            return ""
        stream_idx = streams[0]
        with tempfile.TemporaryDirectory(prefix="cola_sub_") as tmp:
            srt_path = Path(tmp) / "sub.srt"
            subprocess.run(
                ["ffmpeg", "-v", "quiet", "-i", video_path,
                 "-map", f"0:{stream_idx}", "-c:s", "srt", str(srt_path), "-y"],
                capture_output=True, text=True, timeout=120,
            )
            if srt_path.exists():
                text = srt_path.read_text(encoding="utf-8")
                # 去掉 SRT 时间轴和序号，只保留文字
                text = re.sub(r"\d+\n\d+:\d+:\d+,\d+ --> \d+:\d+:\d+,\d+\n", "", text)
                text = re.sub(r"\n\n+", "\n", text).strip()
                return text
    except Exception:
        pass
    return ""


def extract_frames(video_path: str) -> list[str]:
    """用 OpenCV 提取视频帧。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        return []

    interval = max(1, int(fps * FRAME_INTERVAL))
    frame_paths = []
    count = 0

    with tempfile.TemporaryDirectory(prefix="cola_frames_") as tmp:
        tmp_dir = Path(tmp)
        while count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if count % interval == 0:
                path = str(tmp_dir / f"frame_{count}.jpg")
                cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                frame_paths.append(path)
            count += 1

        cap.release()

        if not frame_paths:
            return []

        # OCR 帧
        ocr_texts = []
        for fp in frame_paths:
            try:
                result = subprocess.run(
                    [str(OCR_TOOL), fp],
                    capture_output=True, text=True, timeout=30,
                )
                if result.returncode == 0 and result.stdout.strip():
                    ocr_texts.append(result.stdout.strip())
            except Exception:
                pass

    return ocr_texts


def process_video(video_path: str) -> dict:
    """处理单个 mp4。"""
    path = Path(video_path)
    name = path.stem

    print(f"  🔍 检测字幕流...")
    subtitle_text = extract_subtitles(video_path)
    if subtitle_text:
        print(f"     ✅ 字幕提取: {len(subtitle_text)} 字")
    else:
        print(f"     ➖ 无字幕流")

    print(f"  🎞️ 抽帧 OCR...")
    frame_texts = extract_frames(video_path)
    ocr_text = "\n".join(frame_texts) if frame_texts else ""
    if ocr_text:
        print(f"     ✅ OCR 提取: {len(ocr_text)} 字 ({len(frame_texts)} 帧)")
    else:
        print(f"     ➖ OCR 无结果")

    # 合并
    parts = []
    if subtitle_text:
        parts.append(subtitle_text)
    if ocr_text:
        if subtitle_text:
            parts.append("--- [画面文字] ---")
        parts.append(ocr_text)

    combined = "\n\n".join(parts)

    return {
        "filename": path.name,
        "title": name,
        "text": combined,
        "chars": len(combined),
        "has_subtitles": bool(subtitle_text),
        "ocr_frames": len(frame_texts),
    }


def main():
    videos = sorted(
        [f for f in VIDEO_DIR.iterdir() if f.suffix.lower() == ".mp4"],
        key=lambda f: numeric_key(f.stem),
    )

    if not videos:
        print("❌ Video/ 下没有 mp4 文件")
        sys.exit(1)

    print(f"📂 共 {len(videos)} 个 mp4\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for i, video in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {video.name}")
        try:
            result = process_video(str(video))
            results.append(result)

            # 保存单个结果
            safe_name = re.sub(r"[^a-zA-Z0-9_一-鿿]", "_", video.stem)[:100]
            (OUTPUT_DIR / f"{safe_name}.txt").write_text(result["text"], encoding="utf-8")

            print(f"   → {result['chars']} 字\n")
        except Exception as e:
            print(f"   ❌ 失败: {e}\n")
            results.append({
                "filename": video.name,
                "title": video.stem,
                "text": "",
                "chars": 0,
                "error": str(e),
            })

    # 汇总
    summary = {
        "total": len(results),
        "with_subtitles": sum(1 for r in results if r.get("has_subtitles")),
        "with_ocr": sum(1 for r in results if r.get("ocr_frames", 0) > 0),
        "total_chars": sum(r["chars"] for r in results),
        "results": results,
    }

    summary_path = OUTPUT_DIR / "_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✅ 全部完成!")
    print(f"   处理: {summary['total']} 个视频")
    print(f"   有字幕: {summary['with_subtitles']}")
    print(f"   OCR识别: {summary['with_ocr']}")
    print(f"   总字数: {summary['total_chars']}")
    print(f"   结果: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
