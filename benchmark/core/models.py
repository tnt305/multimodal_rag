"""Domain models cho benchmark — dataclass thuần, có type hint rõ ràng."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class QuestionType(str, Enum):
    TEXT = "text"          # đáp án nằm trong đoạn văn
    TABLE = "table"        # đáp án nằm trong bảng
    FIGURE = "figure"      # đáp án nằm trong hình (cần caption + ngữ cảnh)


@dataclass
class Question:
    """Một câu hỏi benchmark kèm ground truth (trang + đáp án mong đợi)."""

    id: str
    type: QuestionType
    question: str
    ground_truth_page: int          # trang PDF (1-based) chứa bằng chứng
    expected: str                   # đáp án mong đợi (ngắn gọn, so sánh chuỗi/keyword)
    figure_image: str | None = None  # path ảnh (chỉ câu FIGURE)
    note: str = ""


@dataclass
class Hit:
    """Kết quả retrieval 1 item (trang hoặc chunk)."""

    id: str
    page: int
    score: float


@dataclass
class RetrievalResult:
    """Kết quả retrieval 1 câu hỏi trên 1 corpus: rank của trang đúng + list hit."""

    question_id: str
    method: str
    hits: list[Hit]
    ground_truth_page: int
    rank: int | None = None          # 1-based rank của trang đúng; None nếu miss
    found_in_top_k: bool | None = None


@dataclass
class QAAnswer:
    """Trả lời của model + metadata đo lường."""

    answer: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    cited_pages: list[int] = field(default_factory=list)   # trích từ answer
    error: str | None = None


@dataclass
class QAResult:
    """Kết quả QA 1 câu hỏi × 1 phương pháp (text-only / multimodal)."""

    question_id: str
    method: str
    answer: QAAnswer
    correct: bool = False           # nội dung đúng theo ground truth
    citation_correct: bool = False  # trích đúng trang
    abstained: bool = False         # model nói "không đủ dữ liệu"
