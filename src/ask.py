from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np

from api_client import chat_text, chat_vision
from embed_jina import JinaV5Api


def words(text: str) -> set[str]:
    return set(re.findall(r"[\wÀ-ỹĐđ]+", text.lower(), flags=re.UNICODE))


def retrieve(records: list[dict], question: str, k: int) -> list[dict]:
    query = words(question)
    scored = []
    for record in records:
        text = record.get("text", "") + "\n" + record.get("tables_markdown", "")
        overlap = len(query & words(text))
        score = overlap / max(1, len(query))
        scored.append((score, record))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [{**record, "retrieval_score": round(score, 5)} for score, record in scored[:k] if score > 0]


def retrieve_embeddings(records: list[dict], question: str, k: int, npz: Path, api_url: str, api_key: str) -> list[dict]:
    """Retrieval bằng cosine similarity với embedding records đã lưu (embed_jina --records-embed)."""
    data = np.load(npz)
    vectors = data["record_embeddings"]
    meta = json.loads(npz.with_suffix(".json").read_text(encoding="utf-8"))
    ids = meta["record_ids"]
    if len(ids) != len(records) or any(r["id"] != i for r, i in zip(records, ids)):
        raise SystemExit("--records không khớp với record_ids trong embeddings .json")
    query = JinaV5Api(api_url, api_key).text([question], side="query")[0]
    scores = vectors @ query
    order = np.argsort(-scores)[:k]
    return [{**records[i], "retrieval_score": round(float(scores[i]), 5)} for i in order]


def context_for(records: list[dict]) -> str:
    chunks = []
    for record in records:
        chunks.append(
            f"[PDF trang {record['page']}; trang in {record.get('printed_page')}]\n"
            f"{record.get('text', '')[:7000]}"
        )
    return "\n\n---\n\n".join(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve PDF evidence and ask via API")
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--model", default=os.getenv("QA_MODEL", "qwen/qwen3-vl-8b-instruct"))
    parser.add_argument("--vision", action="store_true")
    parser.add_argument("--vision-model", default=os.getenv("VISION_MODEL", "qwen/qwen3-vl-8b-instruct"))
    parser.add_argument("--embeddings", type=Path, help="npz embeddings (embed_jina --records-embed) để retrieval bằng cosine thay vì lexical")
    parser.add_argument("--api-url", default=os.getenv("EMBED_API_URL", "http://localhost:8036/v1"))
    parser.add_argument("--api-key", default=os.getenv("EMBED_API_KEY", ""))
    args = parser.parse_args()

    records = [json.loads(line) for line in args.records.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.embeddings:
        hits = retrieve_embeddings(records, args.question, args.top_k, args.embeddings, args.api_url, args.api_key)
        method = f"embeddings:{args.embeddings.name}"
    else:
        hits = retrieve(records, args.question, args.top_k)
        method = "lexical"
    if not hits:
        raise SystemExit("No evidence matched. Try a shorter query or increase parsing quality.")
    context = context_for(hits)
    result = {"question": args.question, "method": method, "retrieved": [{"id": h["id"], "page": h["page"], "printed_page": h.get("printed_page"), "score": h["retrieval_score"]} for h in hits]}
    if args.vision:
        images = []
        root = args.records.parent
        for hit in hits:
            asset = hit.get("rendered_page_asset")
            if asset:
                images.append(root / asset)
        result["answer"] = chat_vision(args.question, context, images, args.vision_model)
    else:
        result["answer"] = chat_text(args.question, context, args.model)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
