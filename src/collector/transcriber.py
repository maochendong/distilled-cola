"""高精度转录模块 — 从视频提取文本并区分说话人。

支持数据源：
  1. 本地视频文件（微信视频号/小红书下载后处理）：Whisper + PyAnnote
  2. YouTube：API 字幕快速获取
"""

from __future__ import annotations

import json
import tempfile
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

    def from_subtitle_stream(self, video_path: str, title: str = "") -> tuple[list[TranscriptionSegment], str]:
        """从视频文件的嵌入字幕流中提取文字（有字幕视频，无需 Whisper）。

        检测优先级: srt → ass/ssa → mov_text
        """
        import subprocess

        # 检测字幕流
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-select_streams", "s",
                 "-show_entries", "stream=index:stream_tags=language",
                 "-of", "csv=p=0", video_path],
                capture_output=True, text=True, timeout=30,
            )
            streams = [s.strip() for s in result.stdout.strip().split("\n") if s.strip()]
        except Exception:
            streams = []

        if not streams:
            return [], ""

        # 取第一个字幕流
        stream_idx = streams[0].split(",")[0] if "," in streams[0] else "0"
        print(f"  📑 检测到字幕流 (stream #{stream_idx}), 直接提取")

        srt_path = Path(tempfile.mkdtemp(prefix="cola_sub_")) / "subtitle.srt"
        try:
            subprocess.run(
                ["ffmpeg", "-v", "quiet", "-i", video_path,
                 "-map", f"0:{stream_idx}", "-c:s", "srt",
                 str(srt_path), "-y"],
                capture_output=True, text=True, timeout=120,
            )
        except Exception:
            return [], ""

        if not srt_path.exists():
            return [], ""

        segments = self._parse_srt(str(srt_path))
        srt_path.unlink(missing_ok=True)
        source_id = title or Path(video_path).stem
        return segments, source_id

    def _parse_srt(self, srt_path: str) -> list[TranscriptionSegment]:
        """解析 SRT 字幕文件为 TranscriptionSegment 列表。"""
        import re
        segments = []
        with open(srt_path, encoding="utf-8") as f:
            content = f.read()

        pattern = re.compile(
            r"\d+\n(\d+:\d+:\d+,\d+) --> (\d+:\d+:\d+,\d+)\n(.+?)(?=\n\n|\Z)",
            re.DOTALL,
        )
        for match in pattern.finditer(content):
            start = self._srt_time_to_sec(match.group(1))
            end = self._srt_time_to_sec(match.group(2))
            text = match.group(3).strip().replace("\n", " ")
            if text:
                segments.append(TranscriptionSegment(text=text, start=start, end=end))

        print(f"  📑 字幕提取: {len(segments)} 条, {sum(len(s.text) for s in segments)} 字")
        return segments

    @staticmethod
    def _srt_time_to_sec(t: str) -> float:
        """将 SRT 时间格式 (00:01:23,456) 转换为秒。"""
        import re
        m = re.match(r"(\d+):(\d+):(\d+)[,.](\d+)", t)
        if not m:
            return 0.0
        h, mi, s, ms = int(m[1]), int(m[2]), int(m[3]), int(m[4])
        return h * 3600 + mi * 60 + s + ms / 1000

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
