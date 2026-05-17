"""批量 mlx-whisper 转录 — Apple Silicon 优化版，比标准 Whisper 快 3-5 倍。

用法:
  python scripts/batch_transcribe_mlx.py

覆盖 Video/ 下所有 mp4，保存到 data/raw/。
已完成的自动跳过（断点续传）。
"""

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import config

VIDEO_DIR = Path(__file__).parent.parent / "Video"
RAW_DIR = config.raw_dir


def numeric_key(name: str) -> list[int]:
    return [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", name)]


def save_result(stem: str, segments: list[dict], text: str, error: str = ""):
    """保存转录结果，与 batch_transcribe.py 格式一致。"""
    path = RAW_DIR / f"{stem}.json"
    path.write_text(
        json.dumps({"segments": segments, "text": text, "error": error}, ensure_ascii=False),
        encoding="utf-8",
    )


def model_size_from_name(name: str) -> str:
    """根据模型名返回 mlx-whisper 可用的 size 标识。"""
    mapping = {
        "tiny": "tiny",
        "base": "base",
        "small": "small",
        "medium": "medium",
        "large": "large",
        "large-v2": "large-v2",
        "large-v3": "large-v3",
    }
    return mapping.get(name, "large-v3")


def main():
    import mlx_whisper

    videos = sorted(
        [f for f in VIDEO_DIR.iterdir() if f.suffix.lower() == ".mp4"],
        key=lambda f: numeric_key(f.stem),
    )
    if not videos:
        print("❌ Video/ 下没有 mp4 文件")
        sys.exit(1)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    existing = set(f.stem for f in RAW_DIR.glob("*.json"))
    to_process = [v for v in videos if v.stem not in existing]
    completed_count = len(videos) - len(to_process)

    print(f"📂 共 {len(videos)} 个 mp4")
    print(f"✅ 已完成: {completed_count}")
    print(f"⏳ 待处理: {len(to_process)}")
    print()

    # 用 openai-whisper 同样的模型名，mlx-whisper 会转成自己的格式
    model_name = model_size_from_name(config.whisper_model)

    total_start = time.time()
    success = 0
    failed = 0

    for i, video in enumerate(to_process, 1):
        idx = i + completed_count
        print(f"[{idx}/{len(videos)}] {video.name}")
        start = time.time()

        try:
            result = mlx_whisper.transcribe(str(video), path=model_name)

            segments = [
                {"text": seg["text"].strip(), "start": seg["start"], "end": seg["end"]}
                for seg in result["segments"]
            ]
            text = " ".join(s["text"] for s in segments)
            save_result(video.stem, segments, text)

            elapsed = time.time() - start
            print(f"   ✅ {len(text)} 字, {len(segments)} 段 | ⏱️ {elapsed:.1f}s")
            success += 1

        except Exception as e:
            print(f"   ❌ {e}")
            save_result(video.stem, [], "", str(e))
            failed += 1

        print()

    total_time = time.time() - total_start
    print(f"{'='*40}")
    print(f"✅ 全部完成!")
    print(f"   成功: {success} | 失败: {failed}")
    print(f"   总耗时: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"   结果: {RAW_DIR}/")


if __name__ == "__main__":
    main()
