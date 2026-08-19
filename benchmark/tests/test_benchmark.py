"""Smoke tests cho benchmark package — chạy offline không cần API."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.core.models import Hit, Question, QuestionType, RetrievalResult  # noqa: E402
from benchmark.services.audits import kendall_tau, table_heuristic_issues  # noqa: E402
from benchmark.services.benchmark_builder import build_questions  # noqa: E402
from benchmark.services.figure_context import build_figure_context  # noqa: E402
from benchmark.services.qa_eval import _content_correct, _citation_correct  # noqa: E402
from benchmark.services.retrieval_eval import lexical_retrieve, rank_of_page  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


class TestModels(unittest.TestCase):
    def test_question_types(self):
        q = Question(id="B1", type=QuestionType.TABLE, question="q", ground_truth_page=46, expected="35")
        self.assertEqual(q.type.value, "table")


class TestRetrieval(unittest.TestCase):
    def test_lexical_finds_table_page(self):
        records = [json.loads(l) for l in open(REPO / "outputs/parsed/page_records.jsonl", encoding="utf-8") if l.strip()]
        hits = lexical_retrieve(records, "Pháp có bao nhiêu trường trong top 1.000 của bảng xếp hạng QS?", k=5)
        self.assertEqual(rank_of_page(hits, 46), 1)

    def test_rank_of_page(self):
        hits = [Hit(id="a", page=3, score=0.9), Hit(id="b", page=46, score=0.8)]
        self.assertEqual(rank_of_page(hits, 46), 2)
        self.assertIsNone(rank_of_page(hits, 99))


class TestFigureContext(unittest.TestCase):
    def test_context_contains_caption_and_page(self):
        records = [json.loads(l) for l in open(REPO / "outputs/parsed/page_records.jsonl", encoding="utf-8") if l.strip()]
        q = Question(id="F1", type=QuestionType.FIGURE, question="Theo Hình ES.1, Việt Nam thế nào?", ground_truth_page=16, expected="ngoại lệ")
        ctx = build_figure_context(records, q)
        self.assertIn("trang 16", ctx)
        self.assertIn("Hình ES.1", ctx)


class TestQAMetrics(unittest.TestCase):
    def test_content_correct(self):
        self.assertTrue(_content_correct("Pháp có **35 trường**", "35"))
        self.assertFalse(_content_correct("Pháp có 32 trường", "35"))

    def test_citation_correct(self):
        self.assertTrue(_citation_correct("Theo [PDF trang 46; trang in 32]", 46))
        self.assertFalse(_citation_correct("Theo trang 12", 46))


class TestAudits(unittest.TestCase):
    def test_table_heuristic_header_only(self):
        md = "|  | Webometrics | QS | THE |\n| --- | --- | --- | --- |"
        self.assertIn("only_header_no_data", table_heuristic_issues(md))

    def test_kendall_tau_identical_and_reversed(self):
        a = ["x", "y", "z"]
        self.assertEqual(kendall_tau(a, a), 1.0)
        self.assertEqual(kendall_tau(a, list(reversed(a))), -1.0)

    def test_questions_built(self):
        from benchmark.config import BenchmarkSettings

        class _S:
            benchmark = BenchmarkSettings(repo_root=REPO)

        questions = build_questions(_S())
        self.assertEqual(len(questions), 12)
        self.assertEqual(sum(1 for q in questions if q.type == QuestionType.FIGURE), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)