"""Service — lắp ráp ngữ cảnh cho câu hỏi HÌNH.

Yêu cầu quan trọng (đã thống nhất với task): câu hỏi figure KHÔNG chỉ là
"đẩy ảnh + text". Model cần biết ảnh đó là Hình gì, ở trang nào, được tác giả
mô tả ra sao. Vì vậy context figure = assembly:
  1. Page metadata: PDF trang N; trang in M
  2. Caption: dòng "Hình X: ..." trên cùng trang
  3. Cross-reference: đoạn text gần nhất nhắc tên hình (mô tả ý nghĩa)
  4. Đoạn văn lân cận: text trước/sau caption trên cùng trang
"""

from __future__ import annotations

import re

from benchmark.core.models import Question


def build_figure_context(records: list[dict], q: Question, window_chars: int = 600) -> str:
    """Tạo context văn bản quanh hình trên trang ground truth của câu hỏi."""
    page = q.ground_truth_page
    record = next((r for r in records if r["page"] == page), None)
    if record is None:
        return ""
    text = record.get("text", "")
    printed = record.get("printed_page")
    caption = ""
    for line in re.split(r"\n+", text):
        if re.search(r"\bHình\s+[A-Z0-9.]+:", line):
            caption = line.strip()
            break
    figure_ref = re.search(r"\bHình\s+[A-Z0-9.]+", q.question)
    ref_name = figure_ref.group(0) if figure_ref else ""
    cross_ref = ""
    if ref_name:
        idx = text.find(ref_name)
        if idx >= 0:
            cross_ref = text[max(0, idx - 250): idx + 350]
    ctx = f"[PDF trang {page}; trang in {printed}]"
    if caption:
        ctx += f"\nCaption: {caption}"
    if cross_ref:
        ctx += f"\nMô tả liên quan: {cross_ref.strip()}"
    if not cross_ref:
        ctx += f"\nĐoạn trang: {text[:window_chars].strip()}"
    return ctx


def build_evidence_context(record: dict) -> str:
    """Context QA cho 1 record (text + bảng markdown + metadata)."""
    ctx = f"[PDF trang {record['page']}; trang in {record.get('printed_page')}]\n"
    ctx += record.get("text", "")[:7000]
    if record.get("tables_markdown"):
        ctx += f"\nBẢNG:\n{record['tables_markdown'][:4000]}"
    return ctx