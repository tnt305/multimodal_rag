"""INFERENCE TIME — orchestrator: embed query → retrieval → context → QA.

Một lệnh cho toàn bộ luồng hỏi đáp (thay ask.py — vẫn giữ ask.py cho tương thích):

    python src/query.py --records outputs/parsed/page_records.jsonl \
        --embeddings outputs/embeddings_v5.npz \
        --question 'Pháp có bao nhiêu trường top 1.000 QS?'
    python src/query.py ... --vision          # gửi ảnh asset của trang hit
    python src/query.py ... --embeddings outputs/embeddings_bge3_pages.npz --embedder bge3

Bước:
  1. embed query (side=query — prefix Query: do Triton thêm / bge3 thêm prefix retrieval)
  2. cosine: vectors @ query → top-k records
  3. context: [PDF trang N; trang in M] + text
  4. QA: chat_text hoặc chat_vision qua OpenRouter / OpenAI API
  5. JSON: {question, method, retrieved[], answer}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api_client import chat_text, chat_vision  # noqa: E402
from embed_bge3 import QUERY_PREFIX, DEFAULT_URL as BGE3_URL, DEFAULT_MODEL as BGE3_MODEL  # noqa: E402
from embed_bge3 import embed_texts as _bge3_embed  # noqa: E402
from embed_jina import JinaV5Api  # noqa: E402


def load_records(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def embed_query(question: str, embedder: str, api_url: str, api_key: str, bge3_url: str) -> np.ndarray:
    if embedder == "bge3":
        vec = _bge3_embed(bge3_url, BGE3_MODEL, [question], 1, prefix=QUERY_PREFIX)[0]
    else:
        vec = JinaV5Api(api_url, api_key).text([question], side="query")[0]
    vec = np.asarray(vec, dtype=np.float32)
    return vec / np.linalg.norm(vec)


def retrieve(records: list[dict], vectors: np.ndarray, query_vec: np.ndarray, k: int) -> list[dict]:
    scores = vectors @ query_vec
    order = np.argsort(-scores)[:k]
    return [{**records[i], "retrieval_score": round(float(scores[i]), 5)} for i in order]


def context_for(hits: list[dict], max_chars: int = 7000) -> str:
    parts = []
    for record in hits:
        parts.append(
            f"[PDF trang {record['page']}; trang in {record.get('printed_page')}]\n"
            f"{record.get('text', '')[:max_chars]}"
        )
    return "\n\n---\n\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description="Inference: embed query → retrieval → QA")
    ap.add_argument("--records", type=Path, required=True, help="page_records.jsonl")
    ap.add_argument("--embeddings", type=Path, required=True, help="npz từ index.py (embeddings_v5.npz / embeddings_bge3_pages.npz)")
    ap.add_argument("--question", required=True)
    ap.add_argument("--embedder", choices=["v5", "bge3"], default="v5", help="model embed query phải khớp với npz")
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--vision", action="store_true", help="gửi ảnh asset của trang hit qua vision model")
    ap.add_argument("--model", default=None, help="Model QA (mặc định lấy từ QA_MODEL / VISION_MODEL trong .env)")
    ap.add_argument("--api-url", default=os.getenv("EMBED_API_URL", "http://localhost:8036/v1"))
    ap.add_argument("--api-key", default=os.getenv("EMBED_API_KEY", ""))
    ap.add_argument("--bge3-url", default=BGE3_URL)
    args = ap.parse_args()

    if not args.model:
        args.model = os.getenv("VISION_MODEL", "openai/gpt-4o-mini") if args.vision else os.getenv("QA_MODEL", "openai/gpt-4o-mini")

    records = load_records(args.records)
    data = np.load(args.embeddings)
    key = next(k for k in data.files if data[k].ndim == 2)
    vectors = data[key]
    meta = json.loads(args.embeddings.with_suffix(".npz.json").read_text(encoding="utf-8"))
    ids = meta["record_ids"]
    if len(ids) != len(records) or any(r["id"] != i for r, i in zip(records, ids)):
        raise SystemExit(f"--records không khớp record_ids trong {args.embeddings}.json")

    query_vec = embed_query(args.question, args.embedder, args.api_url, args.api_key, args.bge3_url)
    hits = retrieve(records, vectors, query_vec, args.top_k)
    context = context_for(hits)

    result = {
        "question": args.question,
        "method": f"{args.embedder}:{args.embeddings.name}",
        "retrieved": [{"id": h["id"], "page": h["page"], "printed_page": h.get("printed_page"), "score": h["retrieval_score"]} for h in hits],
    }
    if args.vision:
        assets_dir = args.records.parent / "assets"
        images = [assets_dir / Path(h["rendered_page_asset"]).name for h in hits if h.get("rendered_page_asset")]
        result["images"] = [str(p) for p in images]
        result["answer"] = chat_vision(args.question, context, images, args.model)
    else:
        result["answer"] = chat_text(args.question, context, args.model)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()