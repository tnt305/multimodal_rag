#!/usr/bin/env bash
# Env dùng chung cho repo. Source trước khi chạy:
#   source reproduce/env.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Tự động nạp .env nếu tồn tại
if [ -f "$ROOT/.env" ]; then
    set -a
    source "$ROOT/.env"
    set +a
fi

# Python trên máy này
export PYTHON="${PYTHON:-/home/thiendc/projects/.conda/bin/python}"

# Answer model (OpenRouter API)
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-https://openrouter.ai/api/v1}"
export QA_MODEL="${QA_MODEL:-openai/gpt-4o-mini}"
export VISION_MODEL="${VISION_MODEL:-openai/gpt-4o-mini}"

# Embedding Jina v5 omni nano qua embedding-gateway (Triton)
export EMBED_API_KEY="${EMBED_API_KEY:-}"
export EMBED_API_URL="${EMBED_API_URL:-http://localhost:8036/v1}"

# BGE-M3 Remote Embedding Gateway
export BGE3_API_URL="${BGE3_API_URL:-http://localhost:8080/v1/embeddings}"
export BGE3_MODEL="${BGE3_MODEL:-tinix-embedding-cosine}"
