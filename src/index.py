"""OFFLINE INDEXING — orchestrator: parse PDF → chunk → embed → npz.

Một lệnh cho toàn bộ luồng index (mỗi bước có thể chạy riêng bằng script con):

    python src/index.py --pdf data/higher_education_vietnam_vi.pdf
      ├─ 1. parse_pdf.parse()        → outputs/parsed/  (records + assets + stats)
      ├─ 2. chunk_records            → outputs/chunks/chunk_records.jsonl  (nếu --chunks)
      └─ 3. embed                    → outputs/embeddings_v5.npz (+ _chunks, _figures)
             text/image: jina v5 qua gateway Triton (--embedder v5, mặc định)
             text-only : BGE-M3 remote              (--embedder bge3)

Bước 3 đã có sẵn npz thì bỏ qua trừ khi --re-embed.
"""

from __future__ import annotations

import argparse
import os
import json
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

from chunk_records import chunk_records  # noqa: E402
from embed_bge3 import _record_text as _bge3_record_text  # noqa: E402
from embed_bge3 import embed_texts as _bge3_embed  # noqa: E402
from embed_jina import JinaV5Api, load_records  # noqa: E402
from parse_pdf import parse  # noqa: E402


def _embed_v5(embedder: JinaV5Api, texts: list[str], batch: int, side: str) -> np.ndarray:
    vectors = np.concatenate([embedder.text(texts[i : i + batch], side=side) for i in range(0, len(texts), batch)])
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def _save_npz(path: Path, vectors: np.ndarray, ids: list[str], model: str, side: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, record_embeddings=vectors)
    path.with_suffix(".npz.json").write_text(
        json.dumps({"model": model, "record_ids": ids, "dim": int(vectors.shape[1]), "n": int(vectors.shape[0]), "side": side}, indent=2),
        encoding="utf-8",
    )
    print(f"  saved {vectors.shape} -> {path}")


def _figure_paths(records: list[dict], assets_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for record in records:
        asset = record.get("rendered_page_asset")
        if asset:
            paths.append(Path(asset) if Path(asset).is_absolute() else assets_dir / Path(asset).name)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline indexing: parse → chunk → embed → npz")
    ap.add_argument("--pdf", type=Path, default=Path("data/higher_education_vietnam_vi.pdf"))
    ap.add_argument("--out", type=Path, default=Path("outputs/parsed"), help="parse output dir")
    ap.add_argument("--embedder", choices=["v5", "bge3", "none"], default="v5")
    ap.add_argument("--chunks", action="store_true", help="cũng build chunk records + embed chunks")
    ap.add_argument("--figures", action="store_true", help="embed 17 figure assets (chỉ v5)")
    ap.add_argument("--re-embed", action="store_true", help="chạy lại embed kể cả khi npz đã tồn tại")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--force-parse", action="store_true", help="chạy lại parse kể cả khi records đã tồn tại")
    ap.add_argument("--api-url", default=os.getenv("EMBED_API_URL", "http://localhost:8036/v1"))
    ap.add_argument("--api-key", default=os.getenv("EMBED_API_KEY", ""))
    args = ap.parse_args()

    if not args.pdf.exists():
        raise SystemExit(f"PDF không tồn tại: {args.pdf}")

    records_path = args.out / "page_records.jsonl"
    print("==> [1/3] PARSE")
    if records_path.exists() and not args.force_parse:
        records = load_records(records_path)
        print(f"  dùng lại {records_path.name} ({len(records)} records, --force-parse để chạy lại)")
    else:
        stats = parse(args.pdf, args.out, ocr_threshold=80, render_figures=True)
        records = load_records(records_path)
        print(f"  {stats['pages']} trang, {stats['table_page_count']} trang bảng, {stats['figure_page_count']} trang hình, {stats['ocr_page_count']} trang OCR")

    chunk_path = args.out.parent / "chunks" / "chunk_records.jsonl"
    if args.chunks:
        print("==> [2/3] CHUNK")
        chunk_path.parent.mkdir(parents=True, exist_ok=True)
        chunks = chunk_records(records)
        with chunk_path.open("w", encoding="utf-8") as handle:
            for c in chunks:
                handle.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"  {len(chunks)} chunks -> {chunk_path}")

    if args.embedder == "none":
        print("==> [3/3] SKIP EMBED (--embedder none)")
        return 0

    print("==> [3/3] EMBED")
    records_npz = args.out.parent / f"embeddings_{args.embedder}.npz"
    if not records_npz.exists() or args.re_embed:
        if args.embedder == "v5":
            embedder = JinaV5Api(
                args.api_url or "http://localhost:8036/v1",
                args.api_key or "",
            )
            texts = [f"{r.get('text', '')}\n\n{r.get('tables_markdown', '')}" for r in records]
            vectors = _embed_v5(embedder, texts, args.batch_size, side="document")
            ids = [r["id"] for r in records]
            _save_npz(records_npz, vectors, ids, "jinaai/jina-embeddings-v5-omni-nano (Triton jina_v5_text)", "document")
        else:
            bge3_url = os.getenv("BGE3_API_URL", "")
            bge3_model = os.getenv("BGE3_MODEL", "tinix-embedding-cosine")
            texts = [_bge3_record_text(r) for r in records]
            vectors = _bge3_embed(bge3_url, bge3_model, texts, args.batch_size)
            vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
            _save_npz(records_npz, vectors, [r["id"] for r in records], f"BAAI/bge-m3 ({bge3_model})", "document")
    else:
        print(f"  bỏ qua {records_npz.name} (đã tồn tại, dùng --re-embed để chạy lại)")

    if args.chunks:
        chunk_npz = chunk_path.with_name(f"embeddings_{args.embedder}_chunks.npz")
        if not chunk_npz.exists() or args.re_embed:
            chunk_recs = load_records(chunk_path)
            if args.embedder == "v5":
                embedder = JinaV5Api(args.api_url or "http://localhost:8036/v1", args.api_key or "")
                vectors = _embed_v5(embedder, [c["text"] for c in chunk_recs], args.batch_size, side="document")
            else:
                bge3_url = os.getenv("BGE3_API_URL", "")
                bge3_model = os.getenv("BGE3_MODEL", "tinix-embedding-cosine")
                vectors = _embed_bge3(bge3_url, bge3_model, [c["text"] for c in chunk_recs], args.batch_size)
                vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
            _save_npz(chunk_npz, vectors, [c["id"] for c in chunk_recs], "jinaai/jina-embeddings-v5-omni-nano" if args.embedder == "v5" else "BAAI/bge-m3", "document")
        else:
            print(f"  bỏ qua {chunk_npz.name} (đã tồn tại)")

    if args.figures and args.embedder == "v5":
        fig_npz = args.out.parent / "embeddings_v5_figures.npz"
        if not fig_npz.exists() or args.re_embed:
            assets_dir = args.out / "assets"
            paths = _figure_paths(records, assets_dir)
            embedder = JinaV5Api(args.api_url or "http://localhost:8036/v1", args.api_key or "")
            vectors = embedder.images(paths)
            vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
            pages = [int(p.name.split("_")[1]) for p in paths]
            fig_npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(fig_npz, figure_embeddings=vectors, figure_pages=np.asarray(pages, dtype=np.int32))
            print(f"  saved {vectors.shape} -> {fig_npz}")
        else:
            print(f"  bỏ qua {fig_npz.name} (đã tồn tại)")

    print("==> OFFLINE INDEXING DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())