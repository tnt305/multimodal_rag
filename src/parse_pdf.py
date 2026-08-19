from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import fitz
import pdfplumber
from PIL import Image

try:
    import pytesseract
except ImportError:
    pytesseract = None


def clean(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\u00ad", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def printed_page(text: str) -> int | None:
    first = text.splitlines()[0] if text.splitlines() else ""
    match = re.match(r"^\s*(\d{1,3})(?:\s|$)", first)
    return int(match.group(1)) if match else None


def table_markdown(table: list[list[Any]]) -> str:
    rows = []
    for row in table or []:
        row = [clean("" if cell is None else str(cell)).replace("|", "\\|").replace("\n", " ") for cell in row]
        rows.append(row)
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    return "\n".join(
        ["| " + " | ".join(rows[0]) + " |", "| " + " | ".join(["---"] * width) + " |"]
        + ["| " + " | ".join(row) + " |" for row in rows[1:]]
    )


TABLE_CAPTION = re.compile(r"(?m)^\s*(?:Bảng|Table)\s+\d+\s*:")


def native_data_rows(text: str) -> list[list[str]]:
    """Generic fallback: lines ending with 2+ cells (số hoặc gạch ngang) và label không chứa chữ số.

    Trình tách bảng (pdfplumber) có thể chỉ trả về khung tiêu đề cho bảng dùng
    text định vị + đường kẻ màu (vd Bảng 3, trang 46). Khi đó mọi dữ liệu nằm
    trong văn bản gốc ở reading order đúng: "Quốc gia 40 35 32".
    """
    cell = r"(?:\d+|—|–|-)"
    rows: list[list[str]] = []
    for line in text.splitlines():
        line = " ".join(line.split())
        if not line or "://" in line:
            continue
        numbers = re.findall(r"\d+", line)
        if len(numbers) < 2:
            continue
        trailing = re.search(r"(?:\s+" + cell + r")+$", line)
        if not trailing:
            continue
        label = line[: trailing.start()].strip()
        if not label or re.search(r"\d", label) or re.match(r"^\W", label):
            continue
        rows.append([label, *re.findall(cell, trailing.group())])
    return rows


def cell_text(cell: Any) -> str:
    return clean("" if cell is None else str(cell))


def grid_header(grid: list[list[Any]]) -> list[str]:
    """Ghép header grid nhiều dòng (group label + năm) thành 1 dòng, bỏ cột trống.

    vd grid 2 hàng ["Số lượng cơ sở đào tạo", "", ""] / ["", "2010", "2016"]
    -> ["Số lượng cơ sở đào tạo", "Số lượng cơ sở đào tạo 2010", "... 2016"].
    """
    rows = [[cell_text(c) for c in row] for row in grid]
    width = max(len(row) for row in rows)
    merged: list[str] = []
    group = ""
    for j in range(width):
        top = rows[0][j] if j < len(rows[0]) else ""
        sub = rows[1][j] if len(rows) > 1 and j < len(rows[1]) else ""
        if top:
            group = top
            merged.append(f"{top} {sub}".strip() if sub else top)
        elif sub:
            merged.append(f"{group} {sub}".strip() if group else sub)
    return merged


def header_plausible(cells: list[str], native_words: set[str]) -> bool:
    """Header chỉ đáng tin nếu từ ngữ của nó xuất hiện trong native text.

    Bảng dùng encoding hỏng (vd trang 62) cho header bị đảo ngược ký tự
    ("cốuq hnA" thay vì "Anh quốc") — các từ đó không nằm trong native text.
    """
    unknown = 0
    for cell in cells:
        for word in re.findall(r"\w+", cell.lower()):
            if word.isdigit() or word in native_words:
                continue
            unknown += 1
    return unknown <= 2


def rebuild_table(grid: list[list[Any]], rows: list[list[str]], native_text: str) -> list[list[str]]:
    """Ghép header từ grid pdfplumber (nếu tin cậy) với dữ liệu từ native text.

    Thứ tự ưu tiên header:
      1. Header merge nhiều dòng khớp đúng số cột dữ liệu (vd trang 40).
      2. Header 1 dòng khớp cột và không quá 1 ô trống (vd trang 46, 64).
      3. Không dùng header (vd trang 62 — header bị scramble).
    """
    width = max(len(row) for row in rows)
    padded = [(row + [""] * width)[:width] for row in rows]
    if not grid:
        return padded
    native_words = set(re.findall(r"\w+", native_text.lower()))
    merged = grid_header(grid)
    if len(merged) == width and header_plausible(merged, native_words):
        return [merged] + padded
    row0 = [cell_text(c) for c in grid[0]]
    empty = sum(1 for cell in row0 if not cell)
    if len(row0) == width and empty <= 1 and header_plausible(row0, native_words):
        return [row0] + padded
    return padded


def ocr(page: fitz.Page, path: Path) -> str:
    if pytesseract is None:
        return ""
    pix = page.get_pixmap(dpi=220, alpha=False)
    pix.save(path)
    languages = set(pytesseract.get_languages(config=""))
    lang = "vie+eng" if "vie" in languages else "eng"
    return clean(pytesseract.image_to_string(Image.open(path), lang=lang))


def parse(pdf: Path, out: Path, ocr_threshold: int = 80, render_figures: bool = True) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    assets = out / "assets"
    assets.mkdir(exist_ok=True)
    document = fitz.open(pdf)
    records: list[dict[str, Any]] = []
    figure_pages: list[int] = []
    table_pages: list[int] = []
    table_fallback_pages: list[int] = []
    ocr_pages: list[int] = []
    embedded_images = 0

    with pdfplumber.open(pdf) as plumber:
        for index, page in enumerate(document):
            physical_page = index + 1
            native = clean(page.get_text("text", sort=True))
            tables = []
            try:
                tables = plumber.pages[index].extract_tables() or []
            except Exception:
                pass
            extracted_total = sum(len(table) for table in tables)
            fallback_rows = native_data_rows(native) if TABLE_CAPTION.search(native) else []
            used_fallback = bool(fallback_rows and len(fallback_rows) > extracted_total)
            if used_fallback:
                tables = [rebuild_table(tables[0] if tables else [], fallback_rows, native)]
                table_fallback_pages.append(physical_page)
            table_mds = [table_markdown(table) for table in tables]
            table_mds = [table for table in table_mds if table]
            if table_mds:
                table_pages.append(physical_page)

            page_assets: list[str] = []
            for image_no, image in enumerate(page.get_images(full=True), start=1):
                try:
                    data = document.extract_image(image[0])
                    name = f"page_{physical_page:03d}_embedded_{image_no:02d}.{data['ext']}"
                    target = assets / name
                    target.write_bytes(data["image"])
                    page_assets.append(str(target.relative_to(out)))
                    embedded_images += 1
                except Exception:
                    continue

            has_figure_caption = bool(re.search(r"(?m)^\s*Hình\s+", native))
            rendered_page = None
            if render_figures and has_figure_caption:
                name = f"page_{physical_page:03d}_figure_page.png"
                target = assets / name
                page.get_pixmap(dpi=150, alpha=False).save(target)
                rendered_page = str(target.relative_to(out))
                page_assets.append(rendered_page)
                figure_pages.append(physical_page)

            ocr_text = ""
            if len(native) < ocr_threshold:
                target = assets / f"page_{physical_page:03d}_ocr.png"
                ocr_text = ocr(page, target)
                if ocr_text:
                    ocr_pages.append(physical_page)

            merged = native
            if len(ocr_text) > len(native):
                merged = ocr_text
            if table_mds:
                merged += "\n\n" + "\n\n".join(table_mds)

            records.append({
                "id": f"page-{physical_page:03d}",
                "page": physical_page,
                "printed_page": printed_page(native),
                "source": pdf.name,
                "text": merged,
                "native_text": native,
                "ocr_text": ocr_text,
                "tables_markdown": "\n\n".join(table_mds),
                "images": page_assets,
                "rendered_page_asset": rendered_page,
                "has_table": bool(table_mds),
                "has_table_fallback": used_fallback,
                "has_image": bool(page_assets),
                "has_ocr": bool(ocr_text),
            })

    with (out / "page_records.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    stats = {
        "pdf": str(pdf),
        "pages": len(records),
        "table_pages": table_pages,
        "table_page_count": len(table_pages),
        "table_fallback_pages": table_fallback_pages,
        "table_fallback_page_count": len(table_fallback_pages),
        "figure_pages": figure_pages,
        "figure_page_count": len(figure_pages),
        "ocr_pages": ocr_pages,
        "ocr_page_count": len(ocr_pages),
        "embedded_images": embedded_images,
        "native_text_chars": sum(len(r["native_text"]) for r in records),
        "merged_text_chars": sum(len(r["text"]) for r in records),
    }
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse PDF into page records with page metadata.")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--out", type=Path, default=Path("outputs/parsed"))
    parser.add_argument("--ocr-threshold", type=int, default=80)
    parser.add_argument("--no-figure-render", action="store_true")
    args = parser.parse_args()
    print(json.dumps(parse(args.pdf, args.out, args.ocr_threshold, not args.no_figure_render), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
