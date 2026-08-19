"""Orchestrator benchmark — chạy toàn bộ: retrieval + QA + audit + cost/latency.

Usage:
    $PYTHON benchmark/run_benchmark.py
    (cần EMBED_API_URL cho v5 và BGE3_API_URL trong .env nếu dùng BGE-M3)

Output: outputs/benchmark/{retrieval,qa,audits,cost,summary}.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.config import build_settings  # noqa: E402
from benchmark.infrastructure.embedder import BGE3Embedder, V5Embedder  # noqa: E402
from benchmark.infrastructure.qa_client import QAAdapter  # noqa: E402
from benchmark.services.audits import run_all_audits  # noqa: E402
from benchmark.services.benchmark_builder import build_questions, load_records  # noqa: E402
from benchmark.services.cost_latency import cost_summary, latency_stage_report  # noqa: E402
from benchmark.services.qa_eval import QAEvaluator, summarize_qa  # noqa: E402
from benchmark.services.retrieval_eval import RetrievalEvaluator, summarize  # noqa: E402


def _load_vectors(path: Path, key: str = "record_embeddings") -> tuple[np.ndarray, list[str]]:
    data = np.load(path)
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    ids = meta.get("record_ids") or meta.get("ids")
    if ids is None:
        raise KeyError(f"{path}.json thiếu record_ids/ids")
    return data[key] if key in data else data["embeddings"], ids


def _ensure_figure_embeddings(settings) -> tuple[np.ndarray, list[str]]:
    """Embed 17 figure asset bằng v5 (nếu chưa có npz) → (vectors, ids)."""
    path = settings.benchmark.embed_v5_figures
    if path.exists():
        data = np.load(path)
        pages = [int(p) for p in data["figure_pages"]]
        return data["figure_embeddings"], [f"page-{p:03d}" for p in pages]
    embedder = V5Embedder(settings.embedder)
    assets = sorted(settings.benchmark.assets_dir.glob("page_*_figure_page.png"))
    pages = [int(a.name.split("_")[1]) for a in assets]
    vectors = embedder.images(assets)
    np.savez_compressed(path, figure_embeddings=vectors, figure_pages=np.asarray(pages, dtype=np.int32))
    return vectors, [f"page-{p:03d}" for p in pages]


def main() -> int:
    settings = build_settings()
    out_dir = settings.benchmark.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(settings.benchmark.records)
    questions = build_questions(settings)

    v5 = V5Embedder(settings.embedder)
    bge3 = BGE3Embedder(settings.embedder)
    v5_rec, _ = _load_vectors(settings.benchmark.embed_v5_records)
    v5_chunks, chunk_ids = _load_vectors(settings.benchmark.embed_v5_chunks)
    bge3_rec, _ = _load_vectors(settings.benchmark.embed_bge3_records)
    bge3_chunks, _ = _load_vectors(settings.benchmark.embed_bge3_chunks)
    fig_vec, fig_ids = _ensure_figure_embeddings(settings)

    evaluator = RetrievalEvaluator(
        text_corpora={
            "v5_records": (v5, v5_rec, [r["id"] for r in records]),
            "v5_chunks": (v5, v5_chunks, chunk_ids),
            "bge3_records": (bge3, bge3_rec, [r["id"] for r in records]),
            "bge3_chunks": (bge3, bge3_chunks, chunk_ids),
        },
        image_embedder=v5,
        figure_vectors=fig_vec,
        figure_ids=fig_ids,
        records=records,
    )

    t0 = time.perf_counter()
    retrieval = evaluator.run(questions, top_k=10)
    t_retrieval = time.perf_counter() - t0
    ret_summary = summarize(retrieval)
    print("=== RETRIEVAL (recall@1 / recall@5 / MRR) ===")
    for method, s in ret_summary.items():
        print(f"  {method:<14} r@1={s['recall_at_1']:.2f} r@5={s['recall_at_5']:.2f} MRR={s['mrr']:.3f}")

    qa = QAAdapter(settings.qa)
    qa_eval = QAEvaluator(qa, settings.benchmark.records)
    t0 = time.perf_counter()
    qa_results = qa_eval.run(questions, retrieval)
    t_qa = time.perf_counter() - t0
    qa_summary = summarize_qa(qa_results)
    print("\n=== QA (accuracy / citation / abstain) ===")
    for method, s in qa_summary.items():
        print(f"  {method:<18} acc={s['accuracy']:.2f} cit={s['citation_accuracy']:.2f} abstain={s['abstain_rate']:.2f} (n={s['n']})")

    audits = run_all_audits(settings, records, sample_pages=[16, 18, 36, 39, 46, 48, 55, 65])
    print("\n=== AUDITS ===")
    print(f"  table detector F1={audits['table']['detector_p_r_f1']['f1']} (flagged={audits['table']['flagged_pages']})")
    print(f"  figure coverage={audits['figure']['coverage']} missed={audits['figure']['missed_pages']}")
    print(f"  reading order mean_tau={audits['reading_order']['mean_tau']}")
    ocr = audits["ocr"]["pages"]
    cer_vals = [p["cer"] for p in ocr.values() if "cer" in p]
    print(f"  ocr cer mean={sum(cer_vals)/max(1,len(cer_vals)):.4f} (pages checked={len(cer_vals)})")

    cost = cost_summary(qa_results, settings.qa)
    print("\n=== COST / LATENCY ===")
    for method, s in cost.items():
        print(f"  {method:<18} ${s['cost_per_1000_questions']:.2f}/1k Q p50={s['latency_p50_s']}s")
    latency = latency_stage_report(embed_ms=t_retrieval, retrieve_ms=t_retrieval, qa_ms=t_qa * 1000, n_queries=len(questions))

    payload = {
        "retrieval": ret_summary,
        "qa": qa_summary,
        "qa_details": {
            method: [
                {
                    "question_id": r.question_id,
                    "method": r.method,
                    "answer": r.answer.answer,
                    "correct": r.correct,
                    "citation_correct": r.citation_correct,
                    "abstained": r.abstained,
                    "error": r.answer.error,
                    "prompt_tokens": r.answer.prompt_tokens,
                    "completion_tokens": r.answer.completion_tokens,
                    "latency_s": round(r.answer.latency_s, 2),
                }
                for r in rows
            ]
            for method, rows in qa_results.items()
        },
        "audits": audits,
        "cost": cost,
        "latency": latency,
        "questions": [{"id": q.id, "type": q.type.value, "gt_page": q.ground_truth_page, "expected": q.expected} for q in questions],
    }
    for name, data in payload.items():
        (out_dir / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved -> {out_dir}/{{retrieval,qa,audits,cost,latency,summary}}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())