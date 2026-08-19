"""Service — đánh giá QA: accuracy nội dung + citation + abstain + cost.

Phương pháp so sánh (task: "so sánh với cách đọc PDF text-only"):
- textonly_lexical : context từ retrieval lexical (text + bảng markdown), model text
- textonly_v5       : context từ retrieval v5 page-level, model text
- multimodal_figure : câu FIGURE — gửi ẢNH + ngữ cảnh quanh hình (caption,
                      cross-ref, đoạn lân cận) qua vision model
"""

from __future__ import annotations

import re
from pathlib import Path

from benchmark.core.interfaces import IQA
from benchmark.core.models import QAResult, Question, QuestionType, RetrievalResult
from benchmark.services.benchmark_builder import load_records
from benchmark.services.figure_context import build_evidence_context, build_figure_context


def _abstained(answer: str) -> bool:
    return any(k in (answer or "").lower() for k in ("không đủ dữ liệu", "không đủ dữ kiện", "không có đủ"))


def _norm(s: str) -> str:
    return re.sub(r"[\s\-–,._]", "", s.lower())


def _content_correct(answer: str, expected: str) -> bool:
    a = _norm(answer or "")
    e = _norm(expected)
    if e.isdigit():
        return e in a
    return all(k in (answer or "").lower() for k in expected.lower().split())


def _citation_correct(answer: str, ground_truth_page: int) -> bool:
    pages = [int(m) for m in re.findall(r"trang\s+(\d+)", answer or "")]
    return ground_truth_page in pages


class QAEvaluator:
    """Chạy QA cho từng câu hỏi × phương pháp, tính accuracy/citation/abstain."""

    def __init__(self, qa: IQA, records_path: Path):
        self._qa = qa
        self._records = load_records(records_path)

    def run(
        self,
        questions: list[Question],
        retrieval: dict[str, list[RetrievalResult]],
        text_methods: tuple[str, ...] = ("lexical", "v5_records"),
    ) -> dict[str, list[QAResult]]:
        by_method: dict[str, list[QAResult]] = {}
        for q in questions:
            if q.type == QuestionType.FIGURE:
                self._run_figure(q, by_method, retrieval)
            else:
                self._run_text(q, by_method, retrieval, text_methods)
        return by_method

    def _record_context(self, row: RetrievalResult, k: int = 3) -> str:
        pages = [h.page for h in row.hits[:k]]
        records = [r for r in self._records if r["page"] in pages]
        return "\n\n---\n\n".join(build_evidence_context(r) for r in records)

    def _append(self, by_method: dict, method: str, q: Question, answer) -> None:
        by_method.setdefault(method, []).append(
            QAResult(
                question_id=q.id,
                method=method,
                answer=answer,
                correct=_content_correct(answer.answer, q.expected),
                citation_correct=_citation_correct(answer.answer, q.ground_truth_page),
                abstained=_abstained(answer.answer),
            )
        )

    def _run_text(self, q: Question, by_method: dict, retrieval: dict, methods: tuple[str, ...]) -> None:
        for method in methods:
            rows = retrieval.get(method)
            if not rows:
                continue
            row = next((r for r in rows if r.question_id == q.id), None)
            if row is None or row.rank is None:
                continue
            context = self._record_context(row)
            if not context:
                continue
            self._append(by_method, method, q, self._qa.chat_text(q.question, context))

    def _run_figure(self, q: Question, by_method: dict, retrieval: dict) -> None:
        rows = retrieval.get("lexical")
        if rows:
            row = next((r for r in rows if r.question_id == q.id), None)
            if row is not None and row.rank is not None:
                self._append(by_method, "textonly_lexical", q, self._qa.chat_text(q.question, self._record_context(row)))
        context = build_figure_context(self._records, q)
        images = [Path(q.figure_image)] if q.figure_image else []
        answer = self._qa.chat_vision(q.question, context, images) if images else self._qa.chat_text(q.question, context)
        self._append(by_method, "multimodal_figure", q, answer)


def summarize_qa(by_method: dict[str, list[QAResult]]) -> dict:
    summary = {}
    for method, rows in by_method.items():
        n = len(rows)
        summary[method] = {
            "n": n,
            "accuracy": round(sum(1 for r in rows if r.correct) / n, 3),
            "citation_accuracy": round(sum(1 for r in rows if r.citation_correct) / n, 3),
            "abstain_rate": round(sum(1 for r in rows if r.abstained) / n, 3),
            "avg_latency_s": round(sum(r.answer.latency_s for r in rows) / n, 2),
            "total_prompt_tokens": sum(r.answer.prompt_tokens for r in rows),
            "total_completion_tokens": sum(r.answer.completion_tokens for r in rows),
            "errors": [r.question_id for r in rows if r.answer.error],
        }
    return summary