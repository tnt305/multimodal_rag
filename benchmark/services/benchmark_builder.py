"""Service — xây bộ câu hỏi benchmark (text/table/figure) kèm ground truth.

Ground truth được ghi nhận TAY từ nội dung PDF (đã verify ở outputs/parsed):
- text-heavy: câu trả lời nằm trong đoạn văn, ghi trang chứa câu.
- table-heavy: Bảng 3 (trang 46), dữ liệu 17 quốc gia × 3 bảng xếp hạng.
- figure-heavy: hình + caption + cross-reference + đoạn lân cận (xem figure_context).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from benchmark.config import Settings
from benchmark.core.models import Question, QuestionType

# (id, type, question, ground_truth_page, expected, note)
_RAW: list[tuple[str, str, str, int, str, str]] = [
    # ---- text-heavy (đáp án trong đoạn văn) ----
    ("T1", "text",
     "Sinh viên từ nhóm 40% hộ gia đình có hoàn cảnh kinh tế khó khăn nhất chiếm bao nhiêu phần trăm tổng số sinh viên nhập học tại các cơ sở giáo dục sau phổ thông?",
     18, "chưa đến 10%", "VHLSS, trang in 4"),
    ("T2", "text",
     "Có khoảng bao nhiêu viện nghiên cứu công lập (GRI) thuộc phạm vi quản lý của nhiều bộ ngành?",
     55, "hơn 600", "trang in 41"),
    ("T3", "text",
     "Theo khảo sát cấp trường, nguồn thu lớn nhất cho các trường đại học công lập Việt Nam là gì?",
     65, "học phí", "trang in 51"),
    # ---- table-heavy (Bảng 3, trang 46, printed 32) ----
    ("B1", "table",
     "Trong Bảng 3, Việt Nam có bao nhiêu trường trong top 1.000 của Webometrics, QS và Times Higher Education?",
     46, "0-2-1", "Bảng 3"),
    ("B2", "table",
     "Trong Bảng 3, Pháp có bao nhiêu trường trong top 1.000 của bảng xếp hạng QS?",
     46, "35", "Bảng 3"),
    ("B3", "table",
     "Trong Bảng 3, Anh quốc có bao nhiêu trường trong top 1.000 của bảng xếp hạng Times Higher Education?",
     46, "92", "Bảng 3"),
    ("B4", "table",
     "Trong Bảng 3, Đức có bao nhiêu trường trong top 1.000 của Webometrics?",
     46, "60", "Bảng 3"),
    ("B5", "table",
     "Trong Bảng 3, Nhật Bản có bao nhiêu trường trong top 1.000 của bảng xếp hạng QS?",
     46, "44", "Bảng 3"),
    ("B6", "table",
     "Trong Bảng 3, Trung Quốc có bao nhiêu trường trong top 1.000 của Times Higher Education?",
     46, "63", "Bảng 3"),
    # ---- figure-heavy (hình + caption + cross-ref + đoạn lân cận) ----
    ("F1", "figure",
     "Theo mô tả bên cạnh Hình ES.1, Việt Nam nổi lên như thế nào khi so sánh với các quốc gia trong khu vực?",
     16, "ngoại lệ", "Hình ES.1, crop data/p16.png"),
    ("F2", "figure",
     "Theo nội dung gắn với Hình 3, tổng tỉ lệ nhập học ở bậc giáo dục sau phổ thông của Việt Nam tăng từ bao nhiêu phần trăm lên bao nhiêu phần trăm?",
     39, "9% lên 28%", "Hình 3, crop data/p39.png"),
    ("F3", "figure",
     "Hình 8 cho thấy người tốt nghiệp đại học trở lên có lợi thế gì so với người có trình độ học vấn thấp hơn?",
     48, "lợi thế rõ ràng về việc làm và thu nhập", "Hình 8"),
]

_QUERY_IMAGES = {"F1": "data/p16.png", "F2": "data/p39.png"}


def build_questions(settings: Settings) -> list[Question]:
    """Build câu hỏi từ bảng RAW + gắn path ảnh cho câu FIGURE."""
    questions = []
    for qid, qtype, question, page, expected, note in _RAW:
        image = _QUERY_IMAGES.get(qid)
        if image:
            image = str(Path(image) if Path(image).is_absolute() else settings.benchmark.repo_root / image)
        questions.append(
            Question(
                id=qid,
                type=QuestionType(qtype),
                question=question,
                ground_truth_page=page,
                expected=expected,
                figure_image=image,
                note=note,
            )
        )
    return questions


def question_requires_image(q: Question) -> bool:
    return q.type == QuestionType.FIGURE and q.figure_image is not None


def load_records(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def records_by_page(records: list[dict]) -> dict[int, dict]:
    return {r["page"]: r for r in records}


def find_caption(records: list[dict], page: int) -> str:
    """Caption của hình trên trang (dòng chứa 'Hình ...'): dùng làm ngữ cảnh figure."""
    text = records_by_page(records)[page].get("text", "")
    for line in re.split(r"\n+", text):
        if re.search(r"\bHình\s+[A-Z0-9.]+:", line):
            return line.strip()
    return ""