"""批量 OCR Video/ 下所有子文件夹的图片（macOS Vision 框架）。

每个子文件夹 = 一个独立问答主题
图片按数字排序 = 叙事顺序
"""

import json
import re
import subprocess
import sys
from pathlib import Path

VIDEO_DIR = Path(__file__).parent.parent / "Video"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "ocr_results"
OCR_TOOL = Path(__file__).parent / "ocr_tool"


def numeric_key(name: str) -> list[int]:
    """按文件名中的数字排序。"""
    return [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", name)]


def ocr_image(image_path: str) -> str:
    """调用 Swift OCR 工具识别图片文字。"""
    try:
        result = subprocess.run(
            [str(OCR_TOOL), image_path],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        return ""
    except Exception:
        return ""


def process_folder(folder: Path) -> dict:
    """OCR 单个文件夹内所有图片，返回合并文本。"""
    images = sorted(
        [f for f in folder.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")],
        key=lambda f: numeric_key(f.stem),
    )

    if not images:
        return {"folder": folder.name, "text": "", "image_count": 0}

    texts = []
    for img in images:
        text = ocr_image(str(img))
        texts.append(text)

    combined = "\n".join(texts)
    return {
        "folder": folder.name,
        "text": combined,
        "image_count": len(images),
    }


def main():
    folders = sorted(
        [f for f in VIDEO_DIR.iterdir() if f.is_dir()],
        key=lambda f: numeric_key(f.name),
    )

    if not folders:
        print("❌ Video/ 下没有子文件夹")
        sys.exit(1)

    print(f"📂 共 {len(folders)} 个子文件夹")
    print(f"🔧 OCR 工具: {OCR_TOOL}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for i, folder in enumerate(folders, 1):
        print(f"[{i}/{len(folders)}] {folder.name}")
        result = process_folder(folder)
        all_results.append(result)

        # 保存单个结果
        safe_name = re.sub(r"[^a-zA-Z0-9_一-鿿]", "_", folder.name)[:100]
        (OUTPUT_DIR / f"{safe_name}.txt").write_text(
            result["text"], encoding="utf-8"
        )

        status = f"{result['image_count']} 张图"
        if result["text"]:
            status += f", {len(result['text'])} 字"
        else:
            status += ", ⚠️ 未识别到文字"
        print(f"   → {status}\n")

    # 保存汇总
    summary_path = OUTPUT_DIR / "_summary.json"
    summary_path.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"✅ 全部完成！结果保存在 {OUTPUT_DIR}/")
    print(f"   汇总文件: {summary_path}")


if __name__ == "__main__":
    main()
