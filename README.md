# Vietnamese PDF QA  -  RAG đa phương thức + Benchmark

Pipeline QA đa phương thức cho PDF học thuật tiếng Việt (110 trang, Giáo dục đại học Việt Nam): parse (PyMuPDF4LLM + pdfplumber) → bảng→Markdown → hình → embedding (jina v5 / BGE-M3) → retrieval → QA text/vision kèm trích dẫn trang → **benchmark có thước đo** (retrieval, QA accuracy/citation, audit parse, chi phí, latency). Kết quả tổng hợp: [`report.md`](report.md).

## 1. Môi trường & Cấu hình `.env`

Tạo tệp `.env` từ mẫu `.env.example` và điền `OPENAI_API_KEY` (từ OpenRouter hoặc provider OpenAI-compatible):

```bash
git clone https://github.com/tnt305/multimodal_rag.git
cd multimodal_rag
pip install -r requirements.txt
cp .env.example .env
# Chỉnh sửa .env: đặt OPENAI_API_KEY=sk-or-v1-..., OPENAI_API_BASE, EMBED_API_URL, BGE3_API_URL

source reproduce/env.sh     # tự động load .env, thiết lập $PYTHON, API keys, URLs
sudo apt-get install -y tesseract-ocr tesseract-ocr-vie   # cho OCR audit
```

## 2. Các API dùng trong dự án

### 2.1. Embedding gateway (Triton GPU)  -  Cấu hình qua `EMBED_API_URL`
Mặc định: `http://localhost:8036/v1` (Model `jina_v5_text`, `jina_v5_image`).

### 2.2. BGE-M3 remote  -  Cấu hình qua `BGE3_API_URL`
Mặc định: `http://localhost:8080/v1/embeddings` (Model `BAAI/bge-m3`).

### 2.3. QA LLM API (OpenRouter / OpenAI-compatible)  -  Cấu hình qua `OPENAI_API_BASE`
Mặc định: `https://openrouter.ai/api/v1` với `QA_MODEL` / `VISION_MODEL` (`qwen/qwen3-vl-8b-instruct`).

```bash
source reproduce/env.sh
$PYTHON src/api_client.py models
```

## 3. Luồng hoạt động: Offline Indexing → Inference Time

```
══════════ OFFLINE (chạy 1 lần khi có PDF mới) ══════════

PDF ──► ① parse_pdf.py ──► records + assets ──► ② embed_jina.py ──► npz
                                   │                    │
        text (reading order)       │                    └─ side=document
        bảng → tables_markdown     │                       (jina_v5_text / jina_v5_image)
        hình → render PNG asset    │
        OCR nếu trang ít text      ▼
        page_records.jsonl  ──►  page_records.jsonl + embeddings_v5.npz
                                 (110 records)          (110×512 + record_ids)

══════════ INFERENCE (mỗi câu hỏi) ══════════

Câu hỏi ──► ① embed side=query (jina_v5_text, ~ms)
         ──► ② cosine: vectors @ query (110×512, ~0.5ms) → top-k (mặc định 4)
         ──► ③ context: [PDF trang 46; trang in 32] + text (cắt 7000 ký tự)
         ──► ④ QA: chat_text (text/bảng) hoặc chat_vision (+ ảnh asset trang hit)
              └─ LLM QA qua OpenRouter / OpenAI API, trích dẫn trang
         ──► ⑤ JSON: {question, method, retrieved[], answer}
```

**Chi tiết từng bước:**

### Offline Indexing
1. **Parse** (`src/parse_pdf.py`): mỗi trang → native text (`get_text("text", sort=True)` = reading order), bảng qua `pdfplumber.extract_tables()` + fallback `rebuild_table()` khi bảng vỡ (chỉ header, trang 40/46/62/64), ảnh embedded lưu PNG, trang có caption `Hình` → render cả trang thành `assets/page_XXX_figure_page.png`, trang < 80 ký tự → OCR pytesseract. Output: `outputs/parsed/page_records.jsonl` (110 records) + `stats.json` + `assets/`.
2. **Embed** (`src/embed_jina.py --records-embed --api`): mỗi record → `text + tables_markdown`, **side=`document`** → gateway Triton → `embeddings_v5.npz` + `record_ids` json. Chunks (mức dòng bảng/đoạn văn) và figures (17 asset, vision tower) tương tự → `embeddings_v5_chunks.npz`, `embeddings_v5_figures.npz`.

### Inference Time
1. **Embed query**: `ask.py:39`  -  `JinaV5Api.text([question], side="query")` (prefix `Query:` tự thêm trong Triton).
2. **Retrieval**: `ask.py:40-41`  -  `scores = vectors @ query`, `argsort` lấy top-k records.
3. **Build context**: `ask.py:45-52`  -  mỗi hit thành `[PDF trang N; trang in M]` + text.
4. **QA**: text-only → `chat_text` (qwen/qwen3-vl-8b-instruct); `--vision` → `chat_vision` gửi ảnh asset của các hit (base64) + context.
5. **Trả về**: JSON gồm `question`, `method`, `retrieved` (id/page/printed_page/score), `answer`.

## 4. Luồng OFFLINE INDEXING (1 lệnh)

```bash
source reproduce/env.sh
# Indexing mặc định (dùng lại parse/embeddings cũ nếu đã có):
$PYTHON src/index.py --pdf data/higher_education_vietnam_vi.pdf \
  --embedder v5 --chunks --figures        # v5 = Triton gateway (mặc định)

# Chạy lại toàn bộ từ dữ liệu PDF gốc (bắt buộc parse lại & re-embed sạch):
$PYTHON src/index.py --pdf data/higher_education_vietnam_vi.pdf \
  --embedder v5 --chunks --figures --force-parse --re-embed
# bge3: --embedder bge3 (chunks cũng chạy, figures chỉ dành cho v5)
```

`src/index.py` = orchestrator 3 bước: **parse → chunk → embed** (gọi lại các script con):

| Bước | Script con | Output |
|---|---|---|
| 1. Parse | `src/parse_pdf.py` | `outputs/parsed/page_records.jsonl` (110) + `stats.json` + `assets/` |
| 2. Chunk | `src/chunk_records.py` | `outputs/chunks/chunk_records.jsonl` (223 chunks: mỗi dòng bảng + trang) |
| 3. Embed | `src/embed_jina.py` / `src/embed_bge3.py` | `outputs/embeddings_v5.npz` (110×512), `embeddings_v5_chunks.npz` (223×512), `embeddings_v5_figures.npz` (17×512) |

*Xác nhận chạy lại từ PDF gốc:* Parse 110 trang (36 trang bảng, 17 trang hình), tạo 223 chunks, lưu 3 tệp vector `embeddings_v5*.npz` thành công.

## 5. Luồng INFERENCE TIME (1 lệnh)

```bash
source reproduce/env.sh
# Text-only: jina v5 (side=query) → cosine top-4 → qwen/qwen3-vl-8b-instruct kèm trích dẫn
$PYTHON src/query.py --records outputs/parsed/page_records.jsonl \
  --embeddings outputs/embeddings_v5.npz \
  --question 'Pháp có bao nhiêu trường trong top 1.000 của QS?'

# Vision: + ảnh asset của trang hit gửi cho Vision LLM
$PYTHON src/query.py --records outputs/parsed/page_records.jsonl \
  --embeddings outputs/embeddings_v5.npz \
  --question 'Hình ES.1 cho thấy điều gì về Việt Nam?' --vision

# BGE-M3: --embedder bge3 --embeddings outputs/embeddings_bge3_pages.npz
```

`src/query.py` = orchestrator 4 bước: **embed query (side=query) → cosine top-k → context `[PDF trang N; trang in M]` → QA (text hoặc vision)**. Output JSON: `{question, method, retrieved[], answer}`.

*Ví dụ kết quả thực tế qua kiểm thử:*
- **Text Mode**: Query *"Pháp có bao nhiêu trường trong top 1.000 của QS?"* → Top-1 hit: `page-046` (score 0.65186) → Trả lời: *"Pháp có 35 trường trong top 1.000 của bảng xếp hạng QS. [PDF trang 46; trang in 32]"* (Chính xác 100%).
- **Vision Mode**: Query *"Hình ES.1 cho thấy điều gì về Việt Nam?"* (`--vision`) → Top-1 hit: `page-016` (score 0.42955) → Gửi kèm asset `page_016_figure_page.png` → Trả lời chi tiết chỉ số tỉ lệ tốt nghiệp ĐH 19% và số năm đi học 10.5 năm kèm trích dẫn trang 16.

## 6. Parse chi tiết (offline, không cần API)

```bash
bash reproduce/02_offline_pipeline.sh        # hoặc chạy trực tiếp:
$PYTHON src/parse_pdf.py data/higher_education_vietnam_vi.pdf --out outputs/parsed
```

Tạo `outputs/parsed/page_records.jsonl` (110 trang: text, `tables_markdown`, `embedded_images`, `rendered_page_asset`), `stats.json` (ghi `table_fallback_pages`), `assets/page_*_figure_page.png` (17 figure).

**Fix đọc bảng** (`table_fallback_pages = [40, 46, 62, 64]`): khi `pdfplumber.extract_tables()` chỉ trả header (bảng vỡ), parser fallback sang native text  -  mỗi dòng `label + 2+ ô số` thành một hàng; header lấy từ grid nếu khớp, loại header scramble (trang 62), merge 2 hàng grid (trang 40). Reproduce lỗi gốc: `bash reproduce/table_reading_errors/run.sh`.

## 7. Benchmark

```bash
source reproduce/env.sh
$PYTHON benchmark/run_benchmark.py      # cần gateway :8036 + bge3 remote
$PYTHON -m unittest benchmark.tests.test_benchmark   # 9 smoke tests offline
```

### Đánh giá cái gì?

**Retrieval**  -  12 câu hỏi (3 text T1–T3, 6 bảng B1–B6, 3 hình F1–F3) có ground-truth trang + số liệu. Mỗi câu được embed (side=query) rồi cosine với từng corpus, đo recall@1/5/10 + MRR:

| Phương pháp | Corpus | Model |
|---|---|---|
| lexical | toàn bộ records (từ khoá overlap) | không cần model |
| v5_records / v5_chunks | `embeddings_v5.npz` / `embeddings_v5_chunks.npz` | jina v5 (Triton, 512-d) |
| bge3_records / bge3_chunks | `embeddings_bge3_{pages,chunks}.npz` | BAAI/bge-m3 (1024-d) |
| v5_figures | 17 figure assets | jina v5 vision tower (ảnh→ảnh) |

**QA**  -  context từ top-3 hits → model trả lời, đo **accuracy** (khớp giá trị GT), **citation accuracy** (trích đúng trang), **abstain rate**:

| Phương pháp | Câu | Model |
|---|---|---|
| textonly_lexical / textonly_v5 | text + bảng | qwen/qwen3-vl-8b-instruct (text) |
| multimodal_figure | hình (ảnh crop query + caption + đoạn lân cận) | qwen/qwen3-vl-8b-instruct (vision) |

**Audit parse**  -  `audits.json`: bảng (detector P/R/F1 + cross-parser pdfplumber vs pymupdf), hình (coverage vs reference caption/raster), reading order (Kendall τ giữa 2 parser theo word), OCR (CER giữa OCR vs text layer).

**Cost & latency**  -  `cost.json` ($/1.000 câu hỏi theo token×giá model text/vision), `latency.json` (breakdown embed/retrieval/QA, p50/p95).

### Outputs (`outputs/benchmark/`)

| File | Nội dung |
|---|---|
| `retrieval.json` | recall@1/5/10 + MRR cho 6 phương pháp |
| `qa.json` + `qa_details.json` | tổng hợp + từng câu kèm answer/citation/latency |
| `audits.json` | 4 audit parse |
| `cost.json`, `latency.json` | chi phí + tốc độ |
| `questions.json` | bộ câu hỏi benchmark |

Kết quả hiện tại (tóm tắt  -  chi tiết `report.md`): v5 **12/12 r@1** page & chunk; lexical 10/12; bge3 chunks 10/12; QA v5 acc 0.67 + citation 100% câu bảng; multimodal figure acc 0.67 với Vision LLM trên toàn bộ F1–F3; OCR CER 3.2%; chi phí QA v5 records ~$0.60/1k câu, multimodal figure ~$0.39/1k câu (`qwen/qwen3-vl-8b-instruct`).

## 8. Các failure mode cần phân biệt

| Triệu chứng | Nguyên nhân |
|---|---|
| `embed_jina.py` encode lỗi | local mode segfault → dùng `--api` |
| Gateway 404/timeout | chưa load model trong Triton (`/v2/repository/models/jina_v5_text/load`) hoặc gateway chết (xem `gateway_run.log`) |
| QA trả lỗi hoặc chuỗi rác | LLM API flaky  -  retry đã built-in, kiểm tra `OPENAI_API_KEY` trong `.env` |
| BGE-M3 top-1 tụt | quên prefix `Represent this sentence...` cho query |
| Retrieval ảnh sai trang | cross-modal ảnh→text yếu; phải ảnh→ảnh rồi map trang |
| `02_offline_pipeline.sh` lỗi | dependency/parser, không phải API key |
| Docling full bị kill | memory pressure → chạy lite: bỏ `--full-ocr` |

## 9. Layout

| Path | Vai trò |
|---|---|
| `src/index.py` | **OFFLINE** orchestrator: parse → chunk → embed (1 lệnh) |
| `src/query.py` | **INFERENCE** orchestrator: embed query → retrieval → QA (1 lệnh) |
| `src/parse_pdf.py` | Parse metadata, text, bảng (fix fallback), hình, OCR |
| `src/chunk_records.py` | Chunk records → mức dòng bảng + mức trang |
| `src/embed_jina.py` | Embed v5  -  local (`JinaV5`) hoặc gateway Triton (`JinaV5Api`) |
| `src/embed_bge3.py` | Embed BGE-M3 (remote tinix) |
| `src/api_client.py` | Chat text/vision + retry client cho OpenRouter / OpenAI API |
| `src/ask.py` | CLI cũ (giữ tương thích; dùng `src/query.py` mới) |
| `benchmark/` | Package đánh giá: `core/` (models, interfaces), `infrastructure/` (embedder, QA client, parsers), `services/` (builder, figure context, retrieval/QA eval, audits, cost), `run_benchmark.py`, `tests/` |
| `outputs/parsed/`, `outputs/chunks/` | Records + chunks |
| `outputs/embeddings_v5{,_chunks,_figures}.npz`, `embeddings_bge3_{pages,chunks}.npz` | Vector đã build |
| `outputs/benchmark/` | Kết quả đánh giá |
| `report.md` | Báo cáo kết quả + debug guide |
| `_archive/` | Docling probe + bản parse cũ (không nằm trong pipeline) |
| `reproduce/` | Script tái lập từ đầu |
| `data/` | PDF + ảnh query (`p16.png`, `p39.png`, `p48.png`) |

## References

[1]: https://huggingface.co/jinaai/jina-embeddings-v5-omni-nano "Jina Embeddings v5 Omni Nano model card"
[2]: https://huggingface.co/BAAI/bge-m3 "BGE-M3 model card"
[3]: https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/api.html "PyMuPDF4LLM API"
[4]: https://docling-project.github.io/docling/ "Docling documentation"