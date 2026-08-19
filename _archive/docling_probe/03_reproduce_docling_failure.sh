#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PDF="${1:-data/higher_education_vietnam_vi.pdf}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="logs/docling_full_${STAMP}.log"
mkdir -p logs outputs/docling_full
set +e
python3 src/docling_probe.py "$PDF" --out outputs/docling_full --full-ocr >"$LOG" 2>&1
CODE=$?
set -e
printf '{"exit_code":%s,"log":"%s"}\n' "$CODE" "$LOG" | tee logs/docling_full_${STAMP}.json
if [[ "$CODE" -ne 0 ]]; then
  echo "Docling full OCR exited non-zero; inspect $LOG and outputs/docling_full/result.json."
else
  echo "Docling full OCR completed; inspect outputs/docling_full/result.json."
fi
exit 0
