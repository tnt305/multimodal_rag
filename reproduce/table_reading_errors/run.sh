#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source reproduce/env.sh
PY="${PYTHON:-python3}"
rm -rf reproduce/table_reading_errors/artifacts/*.json reproduce/table_reading_errors/artifacts/*.md reproduce/table_reading_errors/artifacts/*.txt
"$PY" reproduce/table_reading_errors/reproduce_table_error.py \
  --pdf data/higher_education_vietnam_vi.pdf \
  --page 46 \
  --out reproduce/table_reading_errors/artifacts
"$PY" - <<'PY'
import json
from pathlib import Path
p = Path('reproduce/table_reading_errors/artifacts/repair_report.json')
r = json.loads(p.read_text(encoding='utf-8'))
assert r['repaired']['checks']['Pháp'] is True
assert r['repaired']['checks']['Việt Nam'] is True
print('TABLE_REPAIR_TEST=PASS')
PY
