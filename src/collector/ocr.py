"""OCR 文字提取模块 — 从图片/视频帧中提取文字。

双引擎策略：
  1. PaddleOCR（主引擎）：离线、高精度中文识别，适合截图、文档、表格
  2. GPT-4o Vision（备选）：理解复杂图表/数据看板的结构化内容

对于房产分析视频，OCR 提取的关键信息包括：
  - 数据表格中的数字和指标
  - 政策文件截图
  - 楼盘户型图/区位图标注
  - 价格走势图上的标签
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

from src.config import config
from src.llm import vision_client


def _encode_image(image_path: str) -> str:
    """将图片转为 base64。"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


class OCREngine:
    """OCR 引擎，自动选择可用后端。"""

    def __init__(self) -> None:
        self._paddle = None
        self.client = vision_client()

    def extract_text(self, image_path: str) -> str:
        """从图片中提取文字。

        优先使用 PaddleOCR（离线、快），失败时回退到 Vision API。
        """
        # 尝试 PaddleOCR
        text = self._ocr_paddle(image_path)
        if text.strip():
            return text

        # 回退 Vision API
        text = self._ocr_vision(image_path)
        return text or ""

    def extract_structured(self, image_path: str) -> dict[str, Any]:
        """对图片进行结构化提取（适合数据表格/图表）。

        Returns:
            {"raw_text": 识别的原始文字, "structured": 结构化数据, "type": "table|chart|screenshot"}
        """
        raw_text = self.extract_text(image_path)

        # 用 Vision API 做结构化理解
        if self.client:
            try:
                b64 = _encode_image(image_path)
                resp = self.client.chat.completions.create(
                    model=config.teacher_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        "这张图片是上海房产分析视频中的画面。请分析：\n"
                                        "1. 图片类型：数据表格 / 图表 / 政策截图 / 楼盘信息 / 其他\n"
                                        "2. 关键信息：提取所有重要的文字和数据\n"
                                        "3. 分析结论：这张图传递了什么观点或数据\n\n"
                                        "以 JSON 格式返回：{\"type\": \"\", \"key_info\": \"\", \"analysis\": \"\"}"
                                    ),
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"},
                                },
                            ],
                        }
                    ],
                    temperature=0.1,
                    max_tokens=1024,
                    response_format={"type": "json_object"},
                )
                content = resp.choices[0].message.content or "{}"
                structured = json.loads(content)
            except Exception:
                structured = {}
        else:
            structured = {}

        return {
            "raw_text": raw_text,
            "structured": structured,
            "image_path": image_path,
        }

    def _ocr_paddle(self, image_path: str) -> str:
        """使用 PaddleOCR 提取文字。"""
        try:
            if self._paddle is None:
                from paddleocr import PaddleOCR  # type: ignore[import-untyped]
                self._paddle = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
            result = self._paddle.ocr(image_path, cls=True)
            if not result or not result[0]:
                return ""
            texts = []
            for line in result[0]:
                texts.append(line[1][0])
            return "\n".join(texts)
        except ImportError:
            return ""
        except Exception as e:
            print(f"    ⚠️ PaddleOCR 失败: {e}")
            return ""

    def _ocr_vision(self, image_path: str) -> str:
        """使用 GPT-4o Vision 回退提取文字。"""
        if not self.client:
            return ""
        try:
            b64 = _encode_image(image_path)
            resp = self.client.chat.completions.create(
                model=config.student_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "请提取这张图片中所有的中英文文字，按原顺序输出。只输出文字内容，不要额外描述。"},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"},
                            },
                        ],
                    }
                ],
                temperature=0.1,
                max_tokens=1024,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            print(f"    ⚠️ Vision API 失败: {e}")
            return ""


def batch_ocr(frame_paths: list[str]) -> list[dict[str, Any]]:
    """批量 OCR 多帧图片，返回结构化结果。"""
    engine = OCREngine()
    results = []
    total = len(frame_paths)
    for i, path in enumerate(frame_paths):
        print(f"    📝 OCR [{i+1}/{total}]: {Path(path).name}", end="")
        result = engine.extract_structured(path)
        if result["raw_text"]:
            print(f" → {len(result['raw_text'])} 字")
        else:
            print(" → (无文字)")
        results.append(result)
    return results
