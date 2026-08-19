"""Infrastructure — QA client có retry, đo latency, log token, parse citation.

Dùng openai SDK qua OpenRouter / OpenAI-compatible API (OPENAI_API_BASE). Trả QAAnswer với cost/latency.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

from openai import OpenAI

from benchmark.config import QASettings
from benchmark.core.interfaces import IQA
from benchmark.core.models import QAAnswer

_GARBAGE_MARKERS = ("Maliciously exceeding limits", "violating rules")


def _garbage(answer: str) -> bool:
    return bool(answer) and any(m in answer for m in _GARBAGE_MARKERS)


def _make_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY missing (set env hoặc configure in .env file)")
    kwargs: dict = {"api_key": key}
    base = os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
    if base:
        kwargs["base_url"] = base
    return OpenAI(**kwargs)


def parse_cited_pages(answer: str) -> list[int]:
    """Trích số trang PDF từ answer: [PDF trang N; trang in M] / trang N."""
    pages = []
    for m in re.finditer(r"trang\s+(\d+)", answer or ""):
        pages.append(int(m.group(1)))
    return pages


class QAAdapter(IQA):
    """Text + vision QA với retry, đo latency + token."""

    def __init__(self, settings: QASettings):
        self._s = settings
        self._system = (
            "Bạn là QA PDF tiếng Việt. Chỉ dùng context; mọi kết luận phải trích "
            "[PDF trang N; trang in M]. Nếu thiếu bằng chứng, nói không đủ dữ liệu."
        )

    def chat_text(self, question: str, context: str) -> QAAnswer:
        client = _make_client()
        last: QAAnswer | None = None
        for attempt in range(self._s.attempts):
            t0 = time.perf_counter()
            try:
                response = client.chat.completions.create(
                    model=self._s.text_model,
                    messages=[
                        {"role": "system", "content": self._system},
                        {"role": "user", "content": f"CÂU HỎI:\n{question}\n\nCONTEXT:\n{context}"},
                    ],
                    max_completion_tokens=self._s.max_tokens,
                )
                answer = response.choices[0].message.content or ""
                last = QAAnswer(
                    answer=answer,
                    model=self._s.text_model,
                    prompt_tokens=getattr(response.usage, "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(response.usage, "completion_tokens", 0) or 0,
                    latency_s=round(time.perf_counter() - t0, 3),
                    cited_pages=parse_cited_pages(answer),
                )
                if not _garbage(answer):
                    return last
            except Exception as exc:  # noqa: BLE001 — lỗi proxy, retry
                last = QAAnswer(answer="", model=self._s.text_model, error=f"attempt {attempt+1}: {exc}")
        return last or QAAnswer(answer="", model=self._s.text_model, error="all attempts failed")

    def chat_vision(self, question: str, context: str, images: list[Path]) -> QAAnswer:
        import base64

        client = _make_client()
        content: list[dict] = [{"type": "text", "text": f"CÂU HỎI:\n{question}\n\nCONTEXT:\n{context}\n\nTrả lời tiếng Việt và trích đúng trang."}]
        for image in images:
            uri = "data:image/png;base64," + base64.b64encode(image.read_bytes()).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": uri, "detail": "high"}})
        last: QAAnswer | None = None
        for attempt in range(self._s.attempts):
            t0 = time.perf_counter()
            try:
                response = client.chat.completions.create(
                    model=self._s.vision_model,
                    messages=[
                        {"role": "system", "content": "Bạn là QA PDF multimodal tiếng Việt. Chỉ dùng context và ảnh; không đoán số liệu không đọc được."},
                        {"role": "user", "content": content},
                    ],
                    max_tokens=4096,
                )
                answer = response.choices[0].message.content or ""
                last = QAAnswer(
                    answer=answer,
                    model=self._s.vision_model,
                    prompt_tokens=getattr(response.usage, "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(response.usage, "completion_tokens", 0) or 0,
                    latency_s=round(time.perf_counter() - t0, 3),
                    cited_pages=parse_cited_pages(answer),
                )
                if not _garbage(answer):
                    return last
            except Exception as exc:  # noqa: BLE001 — lỗi proxy, retry
                last = QAAnswer(answer="", model=self._s.vision_model, error=f"attempt {attempt+1}: {exc}")
        return last or QAAnswer(answer="", model=self._s.vision_model, error="all attempts failed")