from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.records.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert records, "No records were produced"
    pages = {record["page"]: record for record in records}
    assert 18 in pages and 46 in pages, "Expected sample pages 18 and 46"
    assert pages[18]["printed_page"] == 4, pages[18]["printed_page"]
    assert pages[46]["printed_page"] == 32, pages[46]["printed_page"]
    assert "Hình ES.2" in pages[18]["text"]
    assert "Bảng 3" in pages[46]["text"]
    assert pages[18]["rendered_page_asset"], "Vector figure page was not rendered"
    assert pages[46]["tables_markdown"], "Table Markdown is empty"
    print(json.dumps({"ok": True, "records": len(records), "page18": pages[18]["rendered_page_asset"], "page46_table": True}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
