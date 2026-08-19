#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  source reproduce/env.sh
fi
"${PYTHON:-python3}" src/api_client.py api-check --chat "$@"
