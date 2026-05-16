"""高精度转录模块 — 从视频提取文本并区分说话人。

支持数据源：
  1. 本地视频文件（微信视频号/小红书下载后处理）：Whisper + PyAnnote
  2. YouTube：API 字幕快速获取
"""

from __future__ import annotations

import json
from pathlib import Path

from src.config import config


class TranscriptionSegment:
    """单段转录结果，包含时间戳和说话人信息。"""

    def __init__(self, text: str, start: float, end: float, speaker: str = "") -> None:
        self.text = text
        self.start = start
        self.end = end
        self.speaker = speaker

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "speaker": self.speaker,
        }


class Transcriber:
    """视频转录器，支持多种数据源。"""

    def from_video_file(self, video_path: str, title: str = "") -> tuple[list[TranscriptionSegment], str]:
        """从本地视频文件转录（微信视频号/小红书下载后处理）。

        Args:
            video_path: 本地视频文件路径
            title: 视频标题（可选）

        Returns:
            (segments, source_id)
        """
        # 确保 ffmpeg 可用（static_ffmpeg 会自动下载/提供二进制文件）
        try:
            from static_ffmpeg import add_paths
            add_paths()
        except ImportError:
            pass

        try:
            import whisper  # type: ignore[import-untyped]
        except ImportError:
            raise RuntimeError("请安装 Whisper: pip install openai-whisper")

        model_name = config.whisper_model
        print(f"  🎤 Whisper {model_name} 转录中: {video_path}")
        model = whisper.load_model(model_name)
        result = model.transcribe(video_path, language="zh")

        segments = []
        for seg in result["segments"]:
            segments.append(TranscriptionSegment(
                text=seg["text"].strip(),
                start=seg["start"],
                end=seg["end"],
            ))

        # 可选：说话人分离
        try:
            self._diarize(segments, video_path)
        except Exception:
            pass  # 不阻塞

        source_id = title or Path(video_path).stem
        return segments, source_id

    def from_youtube_subtitles(self, video_id: str, language: str = "zh-Hans") -> tuple[list[TranscriptionSegment], str]:
        """从 YouTube 获取字幕（快速，无需本地转录）。"""
        try:
            from youtube_transcript_api import YouTubeTranscriptApi  # type: ignore[import-untyped]
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[language])
            segments = [
                TranscriptionSegment(text=seg["text"], start=seg["start"], end=seg["start"] + seg["duration"])
                for seg in transcript
            ]
            return segments, video_id
        except ImportError:
            raise RuntimeError("请安装 youtube-transcript-api")

    def _diarize(self, segments: list[TranscriptionSegment], audio_path: str) -> None:
        """PyAnnote 说话人分离。"""
        try:
            from pyannote.audio import Pipeline  # type: ignore[import-untyped]
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
            diarization = pipeline(audio_path)
            for seg in segments:
                mid = (seg.start + seg.end) / 2
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    if turn.start <= mid <= turn.end:
                        seg.speaker = speaker
                        break
        except ImportError:
            pass

    def save(self, segments: list[TranscriptionSegment], filename: str) -> Path:
        """保存转录结果为 JSON。"""
        path = config.raw_dir / f"{filename}.json"
        data = {
            "segments": [s.to_dict() for s in segments],
            "text": " ".join(s.text for s in segments),
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return path
