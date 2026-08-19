"""Infrastructure — parser độc lập cho audit (pdfplumber + pymupdf4llm).

Implement IPageParser / ITableExtractor. Dùng cho:
- reading order: so thứ tự block giữa 2 parser
- table audit: đối chiếu bảng trích từ parser độc lập với pipeline chính
"""

from __future__ import annotations

from pathlib import Path

import pdfplumber

from benchmark.core.interfaces import IPageParser


class PdfplumberParser(IPageParser):
    """pdfplumber — text + table theo thứ tự đọc, block theo vị trí (top→bottom)."""

    def __init__(self, pdf_path: Path):
        self._pdf_path = pdf_path

    def extract_page_text(self, page_number: int) -> str:
        with pdfplumber.open(self._pdf_path) as pdf:
            page = pdf.pages[page_number - 1]
            return page.extract_text() or ""

    def extract_page_blocks(self, page_number: int) -> list[str]:
        with pdfplumber.open(self._pdf_path) as pdf:
            page = pdf.pages[page_number - 1]
            blocks = []
            for block in page.extract_words() or []:
                blocks.append(block.get("text", ""))
            return blocks

    def table_markdown(self, page_number: int) -> str:
        with pdfplumber.open(self._pdf_path) as pdf:
            page = pdf.pages[page_number - 1]
            lines = []
            for table in page.extract_tables() or []:
                for row in table:
                    cells = [("" if c is None else str(c).replace("\n", " ")) for c in row]
                    lines.append("| " + " | ".join(cells) + " |")
            return "\n".join(lines)


class PymupdfParser(IPageParser):
    """PyMuPDF — text blocks theo thứ tự reading (block order)."""

    def __init__(self, pdf_path: Path):
        import fitz

        self._doc = fitz.open(str(pdf_path))

    def extract_page_text(self, page_number: int) -> str:
        return self._doc[page_number - 1].get_text("text") or ""

    def extract_page_blocks(self, page_number: int) -> list[str]:
        blocks = []
        for block in self._doc[page_number - 1].get_text("blocks"):
            if block[6] == 0:  # chỉ text block
                blocks.append(block[4].strip())
        return [b for b in blocks if b]

    def table_markdown(self, page_number: int) -> str:
        """Trích bảng bằng pymupdf4llm (có luận header/border) — dùng làm parser độc lập."""
        import pymupdf4llm

        md = pymupdf4llm.to_markdown(
            self._doc,
            pages=[page_number - 1],
            write_on_blank=False,
        )
        table_lines = [l for l in md.splitlines() if l.strip().startswith("|")]
        return "\n".join(table_lines)