"""蒸馏小可乐 — 上海房产分析专家 配置"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # ── API Keys ──
    deepseek_api_key: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))

    # ── DeepSeek ──
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    student_model: str = os.getenv("STUDENT_MODEL", "deepseek-v4-flash")
    teacher_model: str = os.getenv("TEACHER_MODEL", "deepseek-v4-pro")

    # ── OpenAI（退路：embedding / vision）──
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    # ── 路径 ──
    data_dir: Path = Path(os.getenv("DATA_DIR", "./data"))
    chroma_db_path: Path = Path(os.getenv("CHROMA_DB_PATH", "./data/embeddings/chroma_db"))

    # ── 处理参数 ──
    whisper_model: str = os.getenv("WHISPER_MODEL", "large-v3")
    segment_min_length: int = 100
    segment_max_length: int = 2000
    top_k: int = 5
    hybrid_alpha: float = 0.4

    # ── 知识库集合 ──
    knowledge_collection: str = "sh_knowledge"
    reasoning_collection: str = "sh_reasoning_chains"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"


config = Config()
