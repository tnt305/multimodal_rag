"""Chunk records từ page records — mức dòng bảng + mức trang.

- Mỗi dòng dữ liệu của bảng → 1 chunk riêng (kèm caption Bảng + header),
  giúp retrieval tìm đúng ROW thay vì cả trang.
- Mỗi trang → 1 chunk text (không kèm bảng, tránh trùng).

Usage:
    python src/chunk_records.py --records outputs/parsed/page_records.jsonl \
        --output outputs/chunks/chunk_records.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _md_rows(tables_markdown: str) -> list[list[str]]:
    """Tách markdown bảng thành các dòng (list cells), bỏ header + separator."""
    tables = tables_markdown.split("\n\n")
    out: list[list[str]] = []
    for table in tables:
        lines = [l for l in table.splitlines() if l.strip().startswith("|")]
        if len(lines) < 2:
            continue
        for line in lines[2:]:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if any(cells):
                out.append(cells)
    return out


def _caption_of(text: str) -> str:
    match = re.search(r"(?m)^\s*(Bảng\s+\d+[^\n]{0,80})", text)
    return match.group(1).strip() if match else "Bảng"


def chunk_records(records: list[dict]) -> list[dict]:
    chunks: list[dict] = []
    for record in records:
        page = record["page"]
        printed = record.get("printed_page")
        tables = _md_rows(record.get("tables_markdown", ""))
        if tables:
            caption = _caption_of(record.get("text", ""))
            header = " | ".join(tables[0])
            for idx, cells in enumerate(tables[1:]):
                row = " | ".join(cells)
                chunks.append({
                    "id": f"page-{page:03d}-row-{idx}",
                    "page": page,
                    "printed_page": printed,
                    "text": f"{caption} (PDF trang {page}; trang in {printed}) — {header} — {row}",
                })
        text = (record.get("text") or "").strip()
        if text:
            chunks.append({
                "id": f"page-{page:03d}-text",
                "page": page,
                "printed_page": printed,
                "text": f"Trang {page} (PDF trang {page}; trang in {printed}): {text}",
            })
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, required=True, help="page_records.jsonl từ parse_pdf.py")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    records = [json.loads(l) for l in args.records.read_text(encoding="utf-8").splitlines() if l.strip()]
    chunks = chunk_records(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(json.dumps({"ok": True, "chunks": len(chunks), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()