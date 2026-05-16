"""视频帧提取模块 — 从视频中提取关键帧用于 OCR 文字识别。

提取策略：
  1. 定时采样：每 N 秒取一帧
  2. 场景变换检测：用帧差法检测画面切换，在切换处取帧

房产分析视频中的关键画面：数据表格、政策截图、楼盘户型/区位图、价格走势图。
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class FrameExtractor:
    """从视频提取关键帧。"""

    def __init__(
        self,
        interval_sec: float = 5.0,
        scene_threshold: float = 30.0,
        max_frames: int = 100,
    ) -> None:
        self.interval_sec = interval_sec          # 定时采样间隔
        self.scene_threshold = scene_threshold     # 场景变化检测阈值
        self.max_frames = max_frames               # 最大帧数限制

    def extract(
        self, video_path: str, output_dir: str | None = None,
    ) -> list[dict[str, Any]]:
        """提取视频关键帧，返回帧信息列表。

        Returns:
            [{"path": "帧文件路径", "timestamp": 秒, "source": "interval|scene"}]
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        print(f"  🎞️ 视频: {fps:.1f} fps, {total_frames} 帧, {duration:.0f}s")

        out_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="cola_frames_"))
        out_dir.mkdir(parents=True, exist_ok=True)

        frames: list[dict[str, Any]] = []
        frame_count = 0
        prev_gray = None

        while frame_count < total_frames and len(frames) < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            current_sec = frame_count / fps

            # 1) 定时采样
            if frame_count % int(fps * self.interval_sec) == 0:
                path = self._save_frame(frame, out_dir, f"interval_{current_sec:.1f}s")
                frames.append({"path": path, "timestamp": current_sec, "source": "interval"})

            # 2) 场景变换检测（每 0.5s 检测一次）
            if frame_count % max(1, int(fps * 0.5)) == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    diff = cv2.mean(cv2.absdiff(gray, prev_gray))[0]
                    if diff > self.scene_threshold:
                        path = self._save_frame(frame, out_dir, f"scene_{current_sec:.1f}s")
                        frames.append({"path": path, "timestamp": current_sec, "source": "scene"})
                prev_gray = gray

            frame_count += 1

        cap.release()
        print(f"  📸 提取了 {len(frames)} 个关键帧")

        # 按时间排序、去重（同一秒内只保留一帧）
        frames.sort(key=lambda f: f["timestamp"])
        deduped: list[dict[str, Any]] = []
        seen_times: set[int] = set()
        for f in frames:
            t_key = int(f["timestamp"])
            if t_key not in seen_times:
                seen_times.add(t_key)
                deduped.append(f)
        print(f"  📸 去重后 {len(deduped)} 个关键帧")

        return deduped

    def _save_frame(self, frame: np.ndarray, out_dir: Path, name: str) -> str:
        """保存帧为 JPEG 文件。"""
        safe_name = name.replace("/", "_").replace(":", "_")
        path = str(out_dir / f"{safe_name}.jpg")
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return path

    def extract_from_image(self, image_path: str) -> list[dict[str, Any]]:
        """处理单张图片（直接返回，无需提取帧）。"""
        return [{"path": image_path, "timestamp": 0.0, "source": "image"}]
