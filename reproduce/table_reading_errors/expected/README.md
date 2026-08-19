# Table reading error reproduction

## Input

The test uses the real sample PDF at `data/higher_education_vietnam_vi.pdf`, physical PDF page 46, printed page 32. The page visibly contains **Bảng 3** with four columns: country, Webometrics, QS and Times Higher Education.

## Expected baseline problem

`pdfplumber.extract_tables()` is the intentionally fragile baseline. On this PDF it may return only a header/grid or rows with missing cells because the table is positioned text with colored rules rather than a simple HTML-like table. The exact output is saved to `artifacts/baseline_pdfplumber.json`; the test does not hide or overwrite it.

## Expected repaired result

The repair keeps the native PDF text as a fallback, isolates the section between `Bảng 3:` and `Nguồn:`, and parses rows with a country name followed by three integer columns. The expected key rows are:

| Quốc gia | Webometrics | QS | Times Higher Education |
|---|---:|---:|---:|
| Pháp | 40 | 35 | 32 |
| Việt Nam | 0 | 2 | 1 |

`artifacts/repair_report.json` must contain `checks.Pháp=true` and `checks.Việt Nam=true`. `artifacts/repaired_table.md` contains the repaired Markdown table.

## Debug interpretation

If the baseline has the correct row count and the repaired check fails, the PDF layout changed and the row parser needs a new bbox/regex rule. If the baseline is incomplete but the repaired checks pass, the failure is in the table detector, not the LLM. If both fail, inspect `native_page_text.txt` and render the page at 200–300 dpi; the issue may be OCR or reading order rather than table extraction.
