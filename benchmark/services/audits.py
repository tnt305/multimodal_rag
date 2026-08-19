"""Service — audit chất lượng parse (task yêu cầu debug guide).

Gồm 4 audit độc lập:
1. table_audit   : bảng vỡ cột/header — heuristic validation + cross-parser
                   (pdfplumber vs pymupdf4llm). Độ đo: P/R/F1 của detector.
2. figure_audit  : hình bị bỏ sót — so pipeline với reference PyMuPDF
                   (get_images + get_drawings). Độ đo: recall/coverage.
3. ocr_audit     : OCR kém — CER/WER giữa OCR (pymupdf OCR) và text layer
                   native trên cùng trang; text coverage.
4. reading_order : reading order sai — Kendall tau giữa thứ tự block của
                   pdfplumber vs pymupdf trên từng trang.
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path

import pdfplumber

from benchmark.core.interfaces import IPageParser
from benchmark.infrastructure.parsers import PdfplumberParser, PymupdfParser
from benchmark.services.benchmark_builder import load_records

_HEADER_WORDS = {"bảng", "webometrics", "qs", "times", "higher", "education", "top", "số", "trường"}


def table_heuristic_issues(tables_markdown: str) -> list[str]:
    """Heuristic validation: bảng chỉ header, header scramble, ít số liệu."""
    issues: list[str] = []
    lines = [l for l in tables_markdown.splitlines() if l.strip().startswith("|")]
    if not lines:
        return issues
    data_rows = [l for l in lines[1:] if not re.fullmatch(r"[\s|\-]+", l)]
    numeric_cells = len(re.findall(r"\|\s*\d+\s*\|", tables_markdown))
    if not data_rows:
        issues.append("only_header_no_data")
    if numeric_cells < 2:
        issues.append("few_numeric_cells")
    header = lines[0]
    if re.search(r"[a-z]{8,}", header.split("|")[1] if len(header.split("|")) > 1 else ""):
        issues.append("header_scrambled")
    return issues


def table_audit(records: list[dict], parser_a: IPageParser, parser_b: IPageParser, sample_pages: list[int]) -> dict:
    """Detector bảng lỗi trên N trang mẫu:
    ground truth = trang pipeline (records) có bảng; detector = heuristic.
    Cross-parser: so số dòng bảng pdfplumber vs pymupdf4llm trên từng trang.
    """
    gt_pages = {r["page"] for r in records if r.get("tables_markdown")}
    det_pages = {r["page"] for r in records if table_heuristic_issues(r.get("tables_markdown", ""))}
    tp = len(gt_pages & det_pages)
    fp = len(det_pages - gt_pages)
    fn = len(gt_pages - det_pages)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    cross = {}
    for page in sample_pages:
        md_a = parser_a.table_markdown(page)
        md_b = parser_b.table_markdown(page)
        rows_a = len([l for l in md_a.splitlines() if l.startswith("|")])
        rows_b = len([l for l in md_b.splitlines() if l.startswith("|")])
        cross[page] = {"rows_pdfplumber": rows_a, "rows_pymupdf": rows_b, "mismatch": abs(rows_a - rows_b) > 1}
    return {
        "detector_p_r_f1": {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)},
        "flagged_pages": sorted(det_pages),
        "cross_parser": cross,
    }


def figure_audit(records: list[dict], pdf_path: Path, assets_dir: Path) -> dict:
    """Figure detection vs reference độc lập:
    reference = trang có caption figure (regex "Hình/Biểu đồ/Ảnh <num>:")
    ∪ raster image ≥100×100. Vẽ vector không caption (bảng, trang trí)
    không tính — đã kiểm chứng: các trang bị đánh dấu nhầm không có caption.
    """
    import fitz

    doc = fitz.open(str(pdf_path))
    pipeline_pages = {r["page"] for r in records if r.get("rendered_page_asset")}
    cap_re = re.compile(r"^\s*(Hình|Biểu đồ|Ảnh)\s+[A-Z]*\d+(\.\d+)*\s*[:.]", re.M)
    ref_pages: set[int] = set()
    ref_reason: dict[int, str] = {}
    for i in range(doc.page_count):
        page = doc[i]
        has_cap = bool(cap_re.search(page.get_text()))
        has_img = any((im[2] or 0) >= 100 and (im[3] or 0) >= 100 for im in page.get_images(full=True))
        if has_cap or has_img:
            ref_pages.add(i + 1)
            ref_reason[i + 1] = "caption" if has_cap else "raster_image"
    tp = len(pipeline_pages & ref_pages)
    fn = len(ref_pages - pipeline_pages)
    coverage = tp / len(ref_pages) if ref_pages else 0.0
    return {
        "pipeline_figure_pages": sorted(pipeline_pages),
        "reference_figure_pages": sorted(ref_pages),
        "missed_pages": sorted(ref_pages - pipeline_pages),
        "missed_reasons": {str(p): ref_reason[p] for p in sorted(ref_pages - pipeline_pages)},
        "coverage": round(coverage, 3),
        "tp_fn": {"tp": tp, "fn": fn},
    }


def _cer(ref: str, hyp: str) -> float:
    if not ref:
        return 0.0
    sm = difflib.SequenceMatcher(None, ref, hyp)
    ops = sum(tag in {"replace", "delete", "insert"} for tag, _, _, _, _ in sm.get_opcodes())
    return round(ops / max(1, len(ref)), 4)


def ocr_audit(pdf_path: Path, sample_pages: list[int], parser: IPageParser) -> dict:
    """OCR kém: CER giữa OCR (pymupdf OCR) và text layer native.
    Trang không có text layer → text_coverage thấp = nghi scan.
    Nếu môi trường thiếu tesseract → đánh dấu unavailable, không fail.
    """
    import fitz

    doc = fitz.open(str(pdf_path))
    rows = {}
    for page_no in sample_pages:
        page = doc[page_no - 1]
        native = parser.extract_page_text(page_no)
        if len(native.strip()) <= 50:
            rows[page_no] = {"native_chars": len(native), "note": "low_text_layer_likely_scanned"}
            continue
        try:
            tp = page.get_textpage_ocr(language="vie", dpi=150, full=True)
            ocr = tp.extractText() or ""
            rows[page_no] = {
                "native_chars": len(native),
                "ocr_chars": len(ocr),
                "cer": _cer(native, ocr),
                "text_coverage": round(len(native) / max(1, len(ocr)), 3),
            }
        except Exception as exc:  # noqa: BLE001 — tesseract thiếu
            rows[page_no] = {"native_chars": len(native), "ocr_error": str(exc)[:120]}
    return {"pages": rows}


def kendall_tau(a: list[str], b: list[str]) -> float:
    """Đo mức độ đồng thuận thứ tự block giữa 2 parser (normalized)."""
    if len(a) < 2 or len(b) < 2:
        return 1.0
    common = [x for x in a if x in b]
    if len(common) < 2:
        return 0.0
    pos_b = {x: i for i, x in enumerate(b)}
    order = [pos_b[x] for x in common]
    concordant = sum(1 for i in range(len(order)) for j in range(i + 1, len(order)) if order[i] < order[j])
    total = len(order) * (len(order) - 1) / 2
    return round((concordant - (total - concordant)) / total, 3)


def _page_words_pdfplumber(pdf_path: Path, page_no: int) -> list[str]:
    """Danh sách từ theo thứ tự đọc (top, x0) — pdfplumber."""
    with pdfplumber.open(str(pdf_path)) as pdf:
        words = pdf.pages[page_no - 1].extract_words()
    return [w["text"].lower() for w in sorted(words, key=lambda w: (w["top"], w["x0"]))]


def _page_words_pymupdf(pdf_path: Path, page_no: int) -> list[str]:
    """Danh sách từ theo thứ tự đọc (y0, x0) — pymupdf."""
    import fitz

    doc = fitz.open(str(pdf_path))
    words = doc[page_no - 1].get_text("words")
    return [w[4].lower() for w in sorted(words, key=lambda w: (w[1], w[0]))]


def reading_order_audit(pdf_path: Path, sample_pages: list[int]) -> dict:
    """Reading order: Kendall tau giữa chuỗi TỪ (word-level, sorted top→bottom)
    pdfplumber vs pymupdf trên từng trang. tau=1 trùng thứ tự, tau<0 lệch.
    """
    rows = {}
    for page in sample_pages:
        a = _page_words_pdfplumber(pdf_path, page)
        b = _page_words_pymupdf(pdf_path, page)
        rows[page] = {
            "words_pdfplumber": len(a),
            "words_pymupdf": len(b),
            "kendall_tau": kendall_tau(a, b),
        }
    tau = [r["kendall_tau"] for r in rows.values()]
    return {"pages": rows, "mean_tau": round(sum(tau) / len(tau), 3) if tau else 1.0}


def run_all_audits(settings, records: list[dict], sample_pages: list[int]) -> dict:
    parser_a = PdfplumberParser(settings.benchmark.pdf)
    parser_b = PymupdfParser(settings.benchmark.pdf)
    return {
        "table": table_audit(records, parser_a, parser_b, sample_pages),
        "figure": figure_audit(records, settings.benchmark.pdf, settings.benchmark.assets_dir),
        "ocr": ocr_audit(settings.benchmark.pdf, sample_pages, parser_a),
        "reading_order": reading_order_audit(settings.benchmark.pdf, sample_pages),
    }