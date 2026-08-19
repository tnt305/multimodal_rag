"""Service — đánh giá retrieval: recall@k, MRR trên từng corpus/method.

Corpus được đăng ký qua dict: tên method → (embedder, vectors, ids).
- v5_records / bge3_records : page-level text (110 records)
- v5_chunks / bge3_chunks : chunk-level (168) — row bảng + text trang
- v5_figures : image→image (17 figure asset) — chỉ áp dụng câu FIGURE
- lexical : text trang theo overlap từ khoá (baseline cũ)
"""

from __future__ import annotations

import re
from typing import Protocol

import numpy as np

from benchmark.core.interfaces import IImageEmbedder, ITextEmbedder
from benchmark.core.models import Hit, Question, QuestionType, RetrievalResult


def _words(text: str) -> set[str]:
    return set(re.findall(r"[\wÀ-ỹ]+", text.lower()))


def lexical_retrieve(records: list[dict], question: str, k: int = 10) -> list[Hit]:
    q = _words(question)
    scored = []
    for record in records:
        text = record.get("text", "") + "\n" + record.get("tables_markdown", "")
        overlap = len(q & _words(text))
        scored.append((overlap / max(1, len(q)), record))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [
        Hit(id=r["id"], page=r["page"], score=float(s))
        for s, r in scored[:k] if s > 0
    ]


class _TextCorpus(Protocol):
    name: str
    embedder: ITextEmbedder
    vectors: np.ndarray
    ids: list[str]


class RetrievalEvaluator:
    """Chạy retrieval nhiều method cho 1 câu hỏi và tính rank của trang đúng."""

    def __init__(
        self,
        text_corpora: dict[str, tuple[ITextEmbedder, np.ndarray, list[str]]],
        image_embedder: IImageEmbedder | None,
        figure_vectors: np.ndarray | None,
        figure_ids: list[str],
        records: list[dict],
    ):
        self._text_corpora = text_corpora
        self._image_embedder = image_embedder
        self._figure_vectors = figure_vectors
        self._figure_ids = figure_ids
        self._records = records

    def run(self, questions: list[Question], top_k: int = 10) -> dict[str, list[RetrievalResult]]:
        results: dict[str, list[RetrievalResult]] = {}
        for q in questions:
            for method, hits in self._retrieve(q).items():
                results.setdefault(method, []).append(
                    RetrievalResult(
                        question_id=q.id,
                        method=method,
                        hits=hits[:top_k],
                        ground_truth_page=q.ground_truth_page,
                        rank=rank_of_page(hits, q.ground_truth_page),
                        found_in_top_k=rank_of_page(hits, q.ground_truth_page) is not None,
                    )
                )
        return results

    def _retrieve(self, q: Question) -> dict[str, list[Hit]]:
        out: dict[str, list[Hit]] = {"lexical": lexical_retrieve(self._records, q.question)}
        for name, (embedder, vectors, ids) in self._text_corpora.items():
            qv = embedder.text([q.question], side="query")[0]
            out[name] = cosine_hits(vectors, qv, ids)
        if q.type == QuestionType.FIGURE and self._image_embedder and q.figure_image:
            from pathlib import Path

            qv_img = self._image_embedder.images([Path(q.figure_image)])[0]
            out["v5_figures"] = cosine_hits(self._figure_vectors, qv_img, self._figure_ids)
        return out


def cosine_hits(vectors: np.ndarray, query: np.ndarray, ids: list[str]) -> list[Hit]:
    scores = vectors @ query
    order = np.argsort(-scores)
    return [Hit(id=ids[i], page=page_of(ids[i]), score=float(scores[i])) for i in order]


def page_of(item_id: str) -> int:
    m = re.search(r"page-(\d+)", item_id)
    return int(m.group(1)) if m else 0


def rank_of_page(hits: list[Hit], page: int) -> int | None:
    for i, hit in enumerate(hits):
        if hit.page == page:
            return i + 1
    return None


def summarize(results: dict[str, list[RetrievalResult]]) -> dict:
    """Recall@1/5/10 + MRR@10 theo method — dùng cho report."""
    summary: dict[str, dict] = {}
    for method, rows in results.items():
        n = len(rows)
        summary[method] = {
            "n": n,
            "recall_at_1": sum(1 for r in rows if r.rank == 1) / n,
            "recall_at_5": sum(1 for r in rows if r.rank is not None and r.rank <= 5) / n,
            "recall_at_10": sum(1 for r in rows if r.found_in_top_k) / n,
            "mrr": sum(1 / r.rank for r in rows if r.rank) / n,
            "ranks": {r.question_id: r.rank for r in rows},
        }
    return summary