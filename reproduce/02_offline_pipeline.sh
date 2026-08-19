#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source reproduce/env.sh
PY="${PYTHON:-python3}"
PDF="${1:-data/higher_education_vietnam_vi.pdf}"
OUT="${2:-outputs/parsed}"
"$PY" src/parse_pdf.py "$PDF" --out "$OUT"
"$PY" src/embed_jina.py --dry-run --device auto --truncate-dim 512
"$PY" tests/test_smoke.py --records "$OUT/page_records.jsonl"
cat "$OUT/stats.json"
