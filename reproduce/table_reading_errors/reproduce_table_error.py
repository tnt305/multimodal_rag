from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import fitz
import pdfplumber

HEADERS = [
    "Quốc gia",
    "Webometrics top 1.000",
    "QS top 1.000",
    "Times Higher Education top 1.000",
]
EXPECTED = {
    "Pháp": [40, 35, 32],
    "Việt Nam": [0, 2, 1],
}


def markdown(rows: list[list[str]]) -> str:
    body = [[str(x) for x in row] for row in rows]
    return "\n".join(
        ["| " + " | ".join(HEADERS) + " |", "| --- | ---: | ---: | ---: |"]
        + ["| " + " | ".join(row) + " |" for row in body]
    )


def parse_native_rows(native: str) -> list[list[str]]:
    start = native.find("Bảng 3:")
    end = native.find("Nguồn:", start + 1)
    section = native[start:end if end >= 0 else None]
    rows: list[list[str]] = []
    pattern = re.compile(r"^(?P<country>.+?)\s+(?P<web>\d+)\s+(?P<qs>\d+)\s+(?P<the>\d+)\s*$")
    for line in section.splitlines():
        line = " ".join(line.split())
        match = pattern.match(line)
        if match:
            rows.append([match.group("country"), match.group("web"), match.group("qs"), match.group("the")])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce and repair a broken PDF table extraction")
    parser.add_argument("--pdf", type=Path, default=Path("data/higher_education_vietnam_vi.pdf"))
    parser.add_argument("--page", type=int, default=46, help="1-based physical PDF page")
    parser.add_argument("--out", type=Path, default=Path("reproduce/table_reading_errors/artifacts"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # Baseline: pdfplumber's table detector can return only a header/grid for this
    # visually clear table because the PDF uses positioned text and colored rules.
    with pdfplumber.open(args.pdf) as pdf:
        raw_tables = pdf.pages[args.page - 1].extract_tables() or []
    (args.out / "baseline_pdfplumber.json").write_text(json.dumps(raw_tables, ensure_ascii=False, indent=2), encoding="utf-8")

    document = fitz.open(args.pdf)
    page = document[args.page - 1]
    native = page.get_text("text", sort=True)
    (args.out / "native_page_text.txt").write_text(native, encoding="utf-8")
    repaired_rows = parse_native_rows(native)
    repaired_md = markdown(repaired_rows)
    (args.out / "repaired_table.md").write_text(repaired_md + "\n", encoding="utf-8")

    by_country = {row[0]: [int(row[1]), int(row[2]), int(row[3])] for row in repaired_rows}
    result = {
        "pdf": str(args.pdf),
        "physical_page": args.page,
        "printed_page": 32 if args.page == 46 else None,
        "baseline": {
            "extractor": "pdfplumber.extract_tables",
            "table_count": len(raw_tables),
            "rows_per_table": [len(table) for table in raw_tables],
            "note": "Inspect baseline_pdfplumber.json; this is the intentionally fragile baseline.",
        },
        "repaired": {
            "row_count": len(repaired_rows),
            "columns": HEADERS,
            "key_rows": {country: by_country.get(country) for country in EXPECTED},
            "checks": {country: by_country.get(country) == values for country, values in EXPECTED.items()},
            "markdown": str(args.out / "repaired_table.md"),
        },
    }
    (args.out / "repair_report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
