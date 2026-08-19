from __future__ import annotations

import argparse
import base64
import json
import os
import urllib.request
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def batches(values: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


class JinaV5:
    def __init__(self, model_name: str, device: str = "auto", truncate_dim: int = 512):
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install torch and sentence-transformers to run local embeddings.") from exc
        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.truncate_dim = truncate_dim
        # v5 omni nano: retrieval task dùng prefix Query:/Document:, bỏ audio tower
        # (modality="vision") để tiết kiệm VRAM. Cần transformers>=5.0 (multimodal
        # processor) + peft (adapter LoRA).
        self.model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
            device=device,
            model_kwargs={"default_task": "retrieval", "modality": "vision"},
        )
        self.model.eval()

    def _call(self, values: list[Any], side: str = "document") -> np.ndarray:
        if side == "query":
            output = self.model.encode_query(
                values,
                batch_size=1,
                convert_to_numpy=True,
                normalize_embeddings=True,
                truncate_dim=self.truncate_dim,
                show_progress_bar=False,
            )
        elif side == "bare":
            output = self.model.encode(
                values,
                batch_size=1,
                convert_to_numpy=True,
                normalize_embeddings=True,
                truncate_dim=self.truncate_dim,
                show_progress_bar=False,
            )
        else:
            output = self.model.encode_document(
                values,
                batch_size=1,
                convert_to_numpy=True,
                normalize_embeddings=True,
                truncate_dim=self.truncate_dim,
                show_progress_bar=False,
            )
        output = np.asarray(output, dtype=np.float32)
        if output.ndim == 1:
            output = output[None, :]
        return output

    def text(self, values: list[str], side: str = "document") -> np.ndarray:
        return self._call(values, side=side)

    def images(self, values: list[Any]) -> np.ndarray:
        return self._call(values, side="document")


class JinaV5Api:
    """Gọi jina-embeddings-v5-omni-nano qua embedding-gateway (Triton) thay vì transformers.

    Text:  POST {api_url}/embeddings        model=jina_v5_text (side: query|document|bare)
    Image: POST {api_url}/images/embeddings model=jina_v5_image (data URI base64)
    """

    TEXT_MODEL = "jina_v5_text"
    IMAGE_MODEL = "jina_v5_image"

    def __init__(self, api_url: str, api_key: str, truncate_dim: int = 512):
        if truncate_dim != 512:
            raise ValueError("API mode chỉ hỗ trợ truncate_dim=512 (cố định trong Triton model)")
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.truncate_dim = truncate_dim

    def _post(self, path: str, payload: dict[str, Any]) -> list[list[float]]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            f"{self.api_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [item["embedding"] for item in data["data"]]

    def text(self, values: list[str], side: str = "document") -> np.ndarray:
        payload = {"model": self.TEXT_MODEL, "input": values}
        if side and side != "document":
            payload["side"] = side
        rows = self._post("/embeddings", payload)
        return np.asarray(rows, dtype=np.float32)

    def images(self, values: list[Path]) -> np.ndarray:
        uris = ["data:image/png;base64," + base64.b64encode(p.read_bytes()).decode("ascii") for p in values]
        rows = self._post("/images/embeddings", {"model": self.IMAGE_MODEL, "input": uris})
        return np.asarray(rows, dtype=np.float32)


def load_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Jina v5 omni nano text/image embeddings (local transformers hoặc API/Triton)")
    parser.add_argument("--records", type=Path, help="page_records.jsonl")
    parser.add_argument("--root", type=Path, default=Path("."), help="repo root for relative assets")
    parser.add_argument("--query", action="append", default=[])
    parser.add_argument("--image", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=Path("outputs/embeddings.npz"))
    parser.add_argument("--model", default="jinaai/jina-embeddings-v5-omni-nano")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--truncate-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--api", action="store_true", help="embed qua embedding-gateway (Triton) thay vì transformers local")
    parser.add_argument("--api-url", default=os.getenv("EMBED_API_URL", "http://localhost:8036/v1"))
    parser.add_argument("--api-key", default=os.getenv("EMBED_API_KEY", ""))
    parser.add_argument("--records-embed", action="store_true", help="embed toàn bộ records (text + tables_markdown) thay vì query/image")
    args = parser.parse_args()

    if args.dry_run:
        print(json.dumps({"ok": True, "model": args.model, "device": args.device, "truncate_dim": args.truncate_dim, "api": args.api, "download": not args.api}, indent=2))
        return
    if not args.records and not args.query and not args.image:
        parser.error("provide at least one --query/--image or --records with --records-embed")
    if args.records_embed and not args.records:
        parser.error("--records-embed requires --records")

    if args.api:
        embedder: Any = JinaV5Api(args.api_url, args.api_key, args.truncate_dim)
        device = "api"
    else:
        embedder = JinaV5(args.model, args.device, args.truncate_dim)
        device = embedder.device

    from PIL import Image

    output: dict[str, Any] = {"model": args.model, "device": device, "truncate_dim": args.truncate_dim, "api": args.api}
    arrays: dict[str, np.ndarray] = {}
    if args.query:
        arrays["query_embeddings"] = np.concatenate([embedder.text(batch, side="query") for batch in batches(args.query, args.batch_size)])
        output["queries"] = args.query
        output["query_side"] = "query"
    if args.image:
        if args.api:
            image_inputs: Any = args.image
        else:
            image_inputs = [Image.open(path).convert("RGB") for path in args.image]
        arrays["image_embeddings"] = np.concatenate([embedder.images(batch) for batch in batches(image_inputs, args.batch_size)])
        output["images"] = [str(path) for path in args.image]
    if args.records_embed:
        records = load_records(args.records)
        texts = [f"{record.get('text', '')}\n\n{record.get('tables_markdown', '')}" for record in records]
        arrays["record_embeddings"] = np.concatenate([embedder.text(batch, side="document") for batch in batches(texts, args.batch_size)])
        output["records"] = str(args.records)
        output["record_ids"] = [record["id"] for record in records]
        output["record_side"] = "document"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    args.output.with_suffix(".json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, **output, "output": str(args.output), "shapes": {key: list(value.shape) for key, value in arrays.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
