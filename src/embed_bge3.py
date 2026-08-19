"""Embed records bằng BGE-M3 (remote BGE3_API_URL endpoint) — 1024 chiều, normalized.

Query BẮT BUỘC prefix "Represent this sentence for searching relevant passages: ".
Records (document) không cần prefix.

Usage:
    python src/embed_bge3.py --records outputs/parsed/page_records.jsonl \
        --output outputs/embeddings_bge3_records.npz
    python src/embed_bge3.py --records outputs/chunks/chunk_records.jsonl \
        --output outputs/embeddings_bge3_chunks.npz
"""

from __future__ import annotations

import os
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DEFAULT_URL = os.getenv("BGE3_API_URL", "").strip()
DEFAULT_MODEL = os.getenv("BGE3_MODEL", "tinix-embedding-cosine").strip()


def embed_texts(url: str, model: str, texts: list[str], batch_size: int = 16, prefix: str | None = None) -> np.ndarray:
    if not url:
        raise RuntimeError("BGE3_API_URL is missing. Please configure BGE3_API_URL in your .env file or environment.")
    vectors: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        if prefix:
            batch = [prefix + t for t in batch]
        resp = requests.post(url, json={"model": model, "input": batch}, timeout=120)
        resp.raise_for_status()
        data = resp.json()["data"]
        vectors.extend(np.asarray(item["embedding"], dtype=np.float32) for item in sorted(data, key=lambda x: x["index"]))
    return np.stack(vectors)


def _record_text(r: dict) -> str:
    parts = [f"Trang {r.get('page')}: " + (r.get("text") or "")]
    if r.get("tables_markdown"):
        parts.append("Bảng: " + r["tables_markdown"])
    return "\n\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.records.read_text(encoding="utf-8").splitlines() if l.strip()]
    texts = [_record_text(r) for r in rows]
    vectors = embed_texts(args.url, args.model, texts, args.batch_size)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    ids = [r.get("id") or f"page-{r['page']:03d}" for r in rows]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, embeddings=vectors)
    args.output.with_suffix(".npz.json").write_text(
        json.dumps({"record_ids": ids, "dim": vectors.shape[1], "model": args.model, "n": len(rows)}, indent=2),
        encoding="utf-8",
    )
    print(f"saved {vectors.shape} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())