from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def make_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is missing. Please configure OPENAI_API_KEY in .env file or environment.")
    kwargs: dict[str, Any] = {"api_key": key}
    base = os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1").strip()
    if base:
        kwargs["base_url"] = base
    return OpenAI(**kwargs)


def safe_model_list(client: OpenAI) -> list[dict[str, Any]]:
    response = client.models.list()
    models = []
    for item in response.data:
        models.append({"id": item.id, "owned_by": getattr(item, "owned_by", None)})
    return models


def usage(response: Any) -> dict[str, Any]:
    value = getattr(response, "usage", None)
    if value is None:
        return {}
    result = {}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens"):
        item = getattr(value, name, None)
        if item is not None:
            result[name] = item
    return result


GARBAGE_MARKERS = ("Maliciously exceeding limits", "violating rules")


def _garbage(answer: str) -> bool:
    return bool(answer) and any(marker in answer for marker in GARBAGE_MARKERS)


def chat_text(question: str, context: str, model: str, attempts: int = 3) -> dict[str, Any]:
    last: dict[str, Any] = {}
    for attempt in range(attempts):
        try:
            client = make_client()
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Bạn là QA PDF tiếng Việt. Chỉ dùng context; mọi kết luận phải trích [PDF trang N; trang in M]. Nếu thiếu bằng chứng, nói không đủ dữ liệu."},
                    {"role": "user", "content": f"CÂU HỎI:\n{question}\n\nCONTEXT:\n{context}"},
                ],
                max_completion_tokens=900,
            )
            result = {"model": model, "answer": response.choices[0].message.content, "usage": usage(response), "finish_reason": response.choices[0].finish_reason}
            if not _garbage(result["answer"]):
                return result
            last = result
        except Exception as exc:
            last = {"model": model, "answer": f"ERROR: {exc}", "attempt": attempt + 1}
    return last


def image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def chat_vision(question: str, context: str, images: list[Path], model: str, attempts: int = 3) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": f"CÂU HỎI:\n{question}\n\nCONTEXT:\n{context}\n\nTrả lời tiếng Việt và trích đúng trang."}]
    for image in images:
        content.append({"type": "image_url", "image_url": {"url": image_data_url(image), "detail": "high"}})
    last: dict[str, Any] = {}
    for attempt in range(attempts):
        try:
            client = make_client()
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Bạn là QA PDF multimodal tiếng Việt. Chỉ dùng context và ảnh; không đoán số liệu không đọc được."},
                    {"role": "user", "content": content},
                ],
                max_tokens=4096,
            )
            result = {"model": model, "answer": response.choices[0].message.content, "usage": usage(response), "finish_reason": response.choices[0].finish_reason}
            if not _garbage(result["answer"]):
                return result
            last = result
        except Exception as exc:
            last = {"model": model, "answer": f"ERROR: {exc}", "attempt": attempt + 1}
    return last


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenAI-compatible API smoke test and QA client")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("models", help="Call GET /models and print only IDs")
    check = sub.add_parser("api-check", help="Validate key/base URL and optionally chat")
    check.add_argument("--chat", action="store_true")
    check.add_argument("--model", default=os.getenv("QA_MODEL", "qwen3-vl-8b-instruct"))
    ask = sub.add_parser("ask", help="Text QA using an explicit context file")
    ask.add_argument("question")
    ask.add_argument("--context", type=Path, required=True)
    ask.add_argument("--model", default=os.getenv("QA_MODEL", "qwen3-vl-8b-instruct"))
    vision = sub.add_parser("ask-vision", help="Multimodal QA with context file and PNG assets")
    vision.add_argument("question")
    vision.add_argument("--context", type=Path, required=True)
    vision.add_argument("--image", type=Path, action="append", required=True)
    vision.add_argument("--model", default=os.getenv("VISION_MODEL", "qwen3-vl-8b-instruct"))
    args = parser.parse_args()

    if args.command == "models":
        print(json.dumps({"ok": True, "models": safe_model_list(make_client())}, ensure_ascii=False, indent=2))
    elif args.command == "api-check":
        client = make_client()
        result: dict[str, Any] = {"ok": True, "base_url_configured": bool(os.getenv("OPENAI_API_BASE")), "models": safe_model_list(client)}
        if args.chat:
            response = client.chat.completions.create(model=args.model, messages=[{"role": "user", "content": "Reply with exactly: API_OK"}], max_completion_tokens=20)
            result["chat"] = {"model": args.model, "answer": response.choices[0].message.content, "usage": usage(response)}
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "ask":
        print(json.dumps(chat_text(args.question, args.context.read_text(encoding="utf-8"), args.model), ensure_ascii=False, indent=2))
    elif args.command == "ask-vision":
        print(json.dumps(chat_vision(args.question, args.context.read_text(encoding="utf-8"), args.image, args.model), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
