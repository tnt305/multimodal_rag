"""Cấu hình benchmark — externalized settings, không hardcode.

Nguồn env: biến môi trường (prefix BM_) → fallback file reproduce/env.sh
(EMBED_API_URL, EMBED_API_KEY, OPENAI_API_KEY, OPENAI_API_BASE, QA_MODEL).
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, SecretStr


def _load_env_sh(path: Path) -> dict[str, str]:
    """Nạp export VAR='value' từ file shell — dùng làm fallback nếu env trống."""
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("export "):
            body = line[len("export "):].split("=", 1)
            if len(body) == 2:
                out[body[0]] = body[1].strip("'\"")
    return out


class EmbedderConfig(BaseModel):
    """Nơi embed text/ảnh. v5 = gateway Triton GPU 3; bge3 = tinix remote."""

    api_url: str = "http://localhost:8036/v1"
    api_key: SecretStr = SecretStr("")
    model: str = "jina_v5_text"
    image_model: str = "jina_v5_image"
    query_side: str = "query"
    document_side: str = "document"
    bge3_url: str = ""
    bge3_model: str = "BAAI/bge-m3"
    bge3_query_prefix: str = "Represent this sentence for searching relevant passages: "
    batch_size: int = 16
    timeout_s: int = 300


class QASettings(BaseModel):
    """Model trả lời (text + vision) và giá USD / 1M token để tính chi phí."""

    text_model: str = "claude-opus-4.6"
    vision_model: str = "claude-opus-4.6"
    text_price_in: float = 15.0
    text_price_out: float = 75.0
    vision_price_in: float = 0.15
    vision_price_out: float = 2.5
    max_tokens: int = 900
    attempts: int = 3


class BenchmarkSettings(BaseModel):
    """Đường dẫn dữ liệu benchmark (tương đối theo repo_root)."""

    repo_root: Path = Path(__file__).resolve().parents[1]
    records: Path = Path("outputs/parsed/page_records.jsonl")
    chunks: Path = Path("outputs/chunks/chunk_records.jsonl")
    pdf: Path = Path("data/higher_education_vietnam_vi.pdf")
    assets_dir: Path = Path("outputs/parsed/assets")
    embed_v5_records: Path = Path("outputs/embeddings_v5.npz")
    embed_v5_chunks: Path = Path("outputs/embeddings_v5_chunks.npz")
    embed_v5_figures: Path = Path("outputs/embeddings_v5_figures.npz")
    embed_bge3_records: Path = Path("outputs/embeddings_bge3_pages.npz")
    embed_bge3_chunks: Path = Path("outputs/embeddings_bge3_chunks.npz")
    query_images: list[str] = ["data/p16.png", "data/p39.png"]
    out_dir: Path = Path("outputs/benchmark")

    def resolve(self) -> "BenchmarkSettings":
        """Chuyển path tương đối thành absolute theo repo_root."""
        for field in ("records", "chunks", "pdf", "assets_dir",
                      "embed_v5_records", "embed_v5_chunks", "embed_v5_figures",
                      "embed_bge3_records", "embed_bge3_chunks", "out_dir"):
            raw = getattr(self, field)
            if not raw.is_absolute():
                setattr(self, field, self.repo_root / raw)
        self.query_images = [
            str(p) if Path(p).is_absolute() else str(self.repo_root / p)
            for p in self.query_images
        ]
        return self


def build_settings() -> Settings:
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    except ImportError:
        pass

    env_sh = _load_env_sh(Path(__file__).resolve().parents[1] / "reproduce" / "env.sh")
    key = os.getenv("EMBED_API_KEY", env_sh.get("EMBED_API_KEY", ""))
    url = os.getenv("EMBED_API_URL", env_sh.get("EMBED_API_URL", "http://localhost:8036/v1"))
    bge3_url = os.getenv("BGE3_API_URL", env_sh.get("BGE3_API_URL", ""))
    bge3_model = os.getenv("BGE3_MODEL", env_sh.get("BGE3_MODEL", "BAAI/bge-m3"))
    openai_key = os.getenv("OPENAI_API_KEY", env_sh.get("OPENAI_API_KEY", ""))
    openai_base = os.getenv("OPENAI_API_BASE", env_sh.get("OPENAI_API_BASE", "https://openrouter.ai/api/v1"))
    qa_model = os.getenv("QA_MODEL", env_sh.get("QA_MODEL", "claude-opus-4.6"))
    vision_model = os.getenv("VISION_MODEL", env_sh.get("VISION_MODEL", "claude-opus-4.6"))

    if openai_key:
        os.environ.setdefault("OPENAI_API_KEY", openai_key)
    if openai_base:
        os.environ.setdefault("OPENAI_API_BASE", openai_base)

    settings = Settings(
        embedder=EmbedderConfig(api_url=url, api_key=SecretStr(key), bge3_url=bge3_url, bge3_model=bge3_model),
        qa=QASettings(text_model=qa_model, vision_model=vision_model),
    )
    settings.benchmark = settings.benchmark.resolve()
    return settings


class Settings(BaseModel):
    """Settings tổng — đối tượng duy nhất truyền vào benchmark services."""

    embedder: EmbedderConfig = EmbedderConfig()
    qa: QASettings = QASettings()
    benchmark: BenchmarkSettings = BenchmarkSettings()
