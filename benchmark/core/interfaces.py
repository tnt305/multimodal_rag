"""Interface-first — Protocol cho mọi dependency benchmark.

Services chỉ biết các Protocol này, không biết concrete implementation
(embedder cụ thể, client QA cụ thể). Dễ swap v5 ↔ bge3 ↔ local.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from benchmark.core.models import QAAnswer


class ITextEmbedder(Protocol):
    """Embed text — document cho corpus, query cho câu hỏi."""

    def text(self, values: list[str], side: str = "document") -> np.ndarray: ...


class IImageEmbedder(Protocol):
    """Embed ảnh (PNG path) — corpus figure và query ảnh."""

    def images(self, paths: list[Path]) -> np.ndarray: ...


class IQA(Protocol):
    """Gọi model trả lời — text hoặc vision, có retry + đo cost/latency."""

    def chat_text(self, question: str, context: str) -> QAAnswer: ...
    def chat_vision(self, question: str, context: str, images: list[Path]) -> QAAnswer: ...


class IPageParser(Protocol):
    """Parser PDF cấp block/text theo thứ tự — dùng cho audit reading order."""

    def extract_page_text(self, page_number: int) -> str: ...
    def extract_page_blocks(self, page_number: int) -> list[str]: ...


class ITableExtractor(Protocol):
    """Trích bảng markdown của 1 trang — dùng cho audit đọc bảng."""

    def table_markdown(self, page_number: int) -> str: ...
