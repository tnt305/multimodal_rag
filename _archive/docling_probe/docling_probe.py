from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--out", type=Path, default=Path("outputs/docling"))
    parser.add_argument("--full-ocr", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    result = {"mode": "full_ocr" if args.full_ocr else "lite", "pdf": str(args.pdf)}
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        options = PdfPipelineOptions()
        options.do_ocr = args.full_ocr
        options.do_table_structure = True
        options.generate_page_images = False
        options.generate_picture_images = False
        converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})
        converted = converter.convert(str(args.pdf))
        markdown = converted.document.export_to_markdown()
        (args.out / "document.md").write_text(markdown, encoding="utf-8")
        result["status"] = "ok"
        result["markdown_chars"] = len(markdown)
        result["tables"] = len(getattr(converted.document, "tables", []))
        result["pictures"] = len(getattr(converted.document, "pictures", []))
    except BaseException as exc:
        # BaseException includes KeyboardInterrupt/SystemExit; the shell wrapper records the process exit too.
        result["status"] = "error"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)[:2000]
    result["seconds"] = round(time.time() - started, 2)
    (args.out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
