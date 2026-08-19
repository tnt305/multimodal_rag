"""Service — chi phí ($/1.000 câu hỏi) và latency (p50/p95).

Task yêu cầu "cảnh báo chi phí khi dùng model multimodal": tính theo token
log được từ mỗi QA call × giá model (config), tách text-only vs vision.
"""

from __future__ import annotations

import statistics

from benchmark.config import QASettings
from benchmark.core.models import QAAnswer, QAResult


def answer_cost(answer: QAAnswer, settings: QASettings) -> float:
    if answer.model == settings.vision_model:
        price_in, price_out = settings.vision_price_in, settings.vision_price_out
    else:
        price_in, price_out = settings.text_price_in, settings.text_price_out
    return (answer.prompt_tokens / 1_000_000) * price_in + (answer.completion_tokens / 1_000_000) * price_out


def cost_summary(by_method: dict[str, list[QAResult]], settings: QASettings) -> dict:
    summary = {}
    for method, rows in by_method.items():
        costs = [answer_cost(r.answer, settings) for r in rows if not r.answer.error]
        latencies = [r.answer.latency_s for r in rows if r.answer.latency_s > 0]
        summary[method] = {
            "cost_per_1000_questions": round(sum(costs) / len(costs) * 1000, 2) if costs else 0.0,
            "total_cost": round(sum(costs), 4),
            "latency_p50_s": round(statistics.median(latencies), 2) if latencies else 0.0,
            "latency_p95_s": round(sorted(latencies)[int(len(latencies) * 0.95) - 1], 2) if len(latencies) > 5 else round(max(latencies), 2) if latencies else 0.0,
        }
    return summary


def latency_stage_report(embed_ms: float, retrieve_ms: float, qa_ms: float, n_queries: int) -> dict:
    """Breakdown 3 stage của pipeline query (từ runner đo thực tế)."""
    return {
        "n_queries": n_queries,
        "embed_query_ms": round(embed_ms / max(1, n_queries), 1),
        "retrieve_ms": round(retrieve_ms / max(1, n_queries), 1),
        "qa_ms": round(qa_ms / max(1, n_queries), 1),
        "total_ms": round((embed_ms + retrieve_ms + qa_ms) / max(1, n_queries), 1),
    }