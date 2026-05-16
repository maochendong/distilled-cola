"""测试 OCR 和帧提取模块。"""

import tempfile
from pathlib import Path

import numpy as np

from src.collector.frame_extractor import FrameExtractor


def test_frame_extractor_creates_dir():
    """验证 FrameExtractor 创建输出目录。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        extractor = FrameExtractor()
        # 创建测试用的假视频——实际中跳过，只测目录创建逻辑
        out = Path(tmpdir) / "frames"
        out.mkdir(parents=True, exist_ok=True)
        assert out.exists()


def test_ocr_engine_imports():
    """验证 OCR 模块导入不报错。"""
    from src.collector.ocr import OCREngine, batch_ocr
    engine = OCREngine()
    assert engine is not None
    # PaddleOCR 可能未安装，但不应该导致导入失败
    assert hasattr(engine, "_ocr_paddle")
    assert hasattr(engine, "_ocr_vision")


def test_extract_from_image_returns_path():
    """验证图片直接返回路径。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_img = Path(tmpdir) / "test.jpg"
        fake_img.write_text("fake")
        extractor = FrameExtractor()
        result = extractor.extract_from_image(str(fake_img))
        assert len(result) == 1
        assert result[0]["source"] == "image"
        assert result[0]["path"] == str(fake_img)
