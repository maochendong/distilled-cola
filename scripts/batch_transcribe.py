"""批量 Whisper 转录 — 模型只加载一次，节省 230 次重载时间。"""

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
    path = RAW_DIR / f"{stem}.json"
    path.write_text(
        json.dumps({"segments": segments, "text": text, "error": error}, ensure_ascii=False),
        encoding="utf-8",
    )


def main():
    import whisper

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

    print(f"📂 共 {len(videos)} 个 mp4 | ✅ 已完成: {completed_count} | ⏳ 待处理: {len(to_process)}")
    print(f"🎤 加载 Whisper {config.whisper_model} 模型...", end=" ", flush=True)
    load_start = time.time()
    model = whisper.load_model(config.whisper_model)
    print(f"✅ ({time.time() - load_start:.1f}s)")
    print()

    total_start = time.time()
    success = 0
    failed = 0

    for i, video in enumerate(to_process, 1):
        print(f"[{i + completed_count}/{len(videos)}] {video.name}")
        start = time.time()

        try:
            result = model.transcribe(str(video), language="zh")

            segments = [
                {"text": seg["text"].strip(), "start": seg["start"], "end": seg["end"]}
                for seg in result["segments"]
            ]
            text = " ".join(s["text"] for s in segments)
            save_result(video.stem, segments, text)

            print(f"   ✅ {len(text)} 字, {len(segments)} 段 | ⏱️ {time.time() - start:.1f}s")
            success += 1

        except Exception as e:
            print(f"   ❌ {e}")
            save_result(video.stem, [], "", str(e))
            failed += 1

        elapsed = time.time() - start
        print(f"   ⏱️ {elapsed:.1f}s")
        print()

    total_time = time.time() - total_start
    print(f"✅ 全部完成!")
    print(f"   成功: {success} | 失败: {failed}")
    print(f"   总耗时: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"   结果: {RAW_DIR}/")


if __name__ == "__main__":
    main()
