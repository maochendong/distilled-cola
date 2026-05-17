"""清理 OCR 结果中的水印、广告、话题标签等干扰文字。

基于实际 OCR 数据的噪音分析（见 scripts/analyze_ocr_noise.py）：
- 公众号水印：公众号，上海小克勒 及其 OCR 伪影变体
- 知识星球推广水印
- 视频 URL 后缀（xWT111 等）
- OCR 伪影（孤立符号、乱码行）
- 空文本过滤
"""

import json
import re
from pathlib import Path

OCR_DIR = Path(__file__).parent.parent / "data" / "ocr_results"
SUMMARY_FILE = OCR_DIR / "_summary.json"

# ── 水印行（整行匹配，直接删除） ──
WATERMARK_LINES = [
    r"公众号[，,．.]\s*上海小克勒",
    r"[•·Cｃc%喝＊]\s*公众号[，,．.]\s*上海小克勒",
    r"公众号[，,．.]\s*上海小克勒.*",
    r"公众号[，,]\s*\d+套在[隽隽費費]勒",
    r"[喝而]公众号\s*\d+套在\w+勒",              # OCR水印伪影: "喝公众号 36套在費勒"
    r"\d+元/[a-z]+\s+公众号\s*[•·]?\s*上海小克勒",  # OCR水印伪影: "110956元/mino 公众号 •上海小克勒"
    r"公众号\s*$",
    r"知识星球[：:]\s*小克勒悄悄话",
    r"小克勒悄悄话\s*$",
    r"^上海小克勒\s*$",
]

# ── 行内水印（从行中移除，保留其他文字） ──
INLINE_WATERMARKS = [
    r"[•·Cｃc%喝＊]*公众号[，,．.]\s*上海小克勒",
    r"知识星球[：:]\s*小克勒悄悄话",
]

# ── 尾部噪音（移除行尾特定后缀） ──
TRAILING_NOISE = [
    r"\s*xWT\d+[\d\(\)]*\s*$",
]

# ── OCR 伪影行 ──
OCR_ARTIFACTS = [
    r"^[◇◆●○■□▲△→➡⬆⬇⬅]$",       # 孤立符号
    r"^[○Oo]\s*$",                         # 单字母
    r"^[-–—=]+$",                          # 纯分隔线
    r"^[_\s]{3,}$",                        # 纯下划线
    r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$",     # 孤立日期
]

# ── 需要整体跳过的视频（内容为空或纯噪音） ──
SKIP_FOLDERS: set[str] = set()


def clean_text(text: str) -> str:
    """清理单条文本中的干扰内容。"""
    if not text or not text.strip():
        return ""

    lines = text.split("\n")
    cleaned = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 跳过水印行
        skip = False
        for pattern in WATERMARK_LINES:
            if re.match(pattern, line):
                skip = True
                break
        if skip:
            continue

        # 跳过 OCR 伪影行
        for pattern in OCR_ARTIFACTS:
            if re.match(pattern, line):
                skip = True
                break
        if skip:
            continue

        # 移除行内水印
        for pattern in INLINE_WATERMARKS:
            line = re.sub(pattern, "", line).strip()

        # 移除尾部噪音
        for pattern in TRAILING_NOISE:
            line = re.sub(pattern, "", line).strip()

        if line:
            cleaned.append(line)

    return "\n".join(cleaned)


def main():
    if not SUMMARY_FILE.exists():
        print(f"❌ 未找到 {SUMMARY_FILE}")
        return

    data: list[dict] = json.loads(SUMMARY_FILE.read_text(encoding="utf-8"))

    total_before = sum(len(d.get("text", "")) for d in data)
    non_empty_before = sum(1 for d in data if d.get("text", "").strip())
    empty_count = 0
    noise_line_count = 0

    for item in data:
        orig = item.get("text", "")
        if not orig.strip():
            continue

        cleaned = clean_text(orig)
        noise_line_count += _count_removed_lines(orig, cleaned)

        if not cleaned.strip():
            empty_count += 1

        item["text"] = cleaned

        # 同步更新对应的 .txt 文件
        safe_name = re.sub(r"[^a-zA-Z0-9_一-鿿]", "_", item["folder"])[:100]
        txt_path = OCR_DIR / f"{safe_name}.txt"
        if txt_path.exists():
            txt_path.write_text(cleaned, encoding="utf-8")

    total_after = sum(len(d.get("text", "")) for d in data)
    non_empty_after = sum(1 for d in data if d.get("text", "").strip())

    SUMMARY_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✅ OCR 文本清理完成")
    print(f"   清理前: {total_before} 字, {non_empty_before} 条非空")
    print(f"   清理后: {total_after} 字, {non_empty_after} 条非空")
    print(f"   移除: {total_before - total_after} 字")
    print(f"   删除水印/伪影行: {noise_line_count} 行")
    print(f"   变为空文本: {empty_count} 条")


def _count_removed_lines(orig: str, cleaned: str) -> int:
    orig_lines = set(orig.split("\n"))
    cleaned_lines = set(cleaned.split("\n"))
    return len(orig_lines - cleaned_lines)


if __name__ == "__main__":
    main()
