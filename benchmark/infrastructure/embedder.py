"""Infrastructure — embedder v5 (gateway Triton) và bge3 (tinix remote).

Cả hai implement ITextEmbedder/IImageEmbedder. dùng chung normalize + cosine.
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import numpy as np

from benchmark.config import EmbedderConfig
from benchmark.core.interfaces import IImageEmbedder, ITextEmbedder


def _post_json(url: str, payload: dict, headers: dict | None = None, timeout: int = 300) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers or {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class V5Embedder(ITextEmbedder, IImageEmbedder):
    """jina-embeddings-v5-omni-nano qua embedding-gateway (Triton GPU 3, :8036)."""

    def __init__(self, config: EmbedderConfig):
        self._config = config
        self._headers = {"Content-Type": "application/json"}
        if config.api_key.get_secret_value():
            self._headers["Authorization"] = f"Bearer {config.api_key.get_secret_value()}"
        self._base = config.api_url.rstrip("/")

    def text(self, values: list[str], side: str = "document") -> np.ndarray:
        payload: dict = {"model": self._config.model, "input": values}
        if side and side != self._config.document_side:
            payload["side"] = side
        data = _post_json(f"{self._base}/embeddings", payload, self._headers, self._config.timeout_s)
        rows = [item["embedding"] for item in data["data"]]
        return np.asarray(rows, dtype=np.float32)

    def images(self, paths: list[Path]) -> np.ndarray:
        import base64

        uris = [
            "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode("ascii")
            for p in paths
        ]
        data = _post_json(
            f"{self._base}/images/embeddings",
            {"model": self._config.image_model, "input": uris},
            self._headers,
            self._config.timeout_s,
        )
        rows = [item["embedding"] for item in data["data"]]
        return np.asarray(rows, dtype=np.float32)


class BGE3Embedder(ITextEmbedder):
    """BAAI/bge-m3 (tinix-embedding-cosine) qua vLLM remote — text-only.

    Query PHẢI thêm prefix retrieval (bge-m3 official), document không prefix.
    """

    def __init__(self, config: EmbedderConfig):
        self._config = config

    def text(self, values: list[str], side: str = "document") -> np.ndarray:
        if side == "query" and self._config.bge3_query_prefix:
            values = [self._config.bge3_query_prefix + v for v in values]
        data = _post_json(
            self._config.bge3_url,
            {"model": self._config.bge3_model, "input": list(values)},
            timeout=self._config.timeout_s,
        )
        rows = [item["embedding"] for item in sorted(data["data"], key=lambda d: d["index"])]
        return np.asarray(rows, dtype=np.float32)


def cosine_matrix(query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """(Nq, D) @ (D, Nc) → (Nq, Nc); cả hai đã L2-normalized."""
    return query @ corpus.T