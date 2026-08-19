# Embedding design

## Chosen model

The embedding model is `jinaai/jina-embeddings-v5-omni-nano` (served qua Triton gateway, models `jina_v5_text` / `jina_v5_image`). It is a multimodal omni model: text + vision tower trong cùng vector space, 768 chiều đầy đủ (Matryoshka, deployment cắt `truncate_dim=512`), hỗ trợ 89 ngôn ngữ (kể cả tiếng Việt), max seq 8192, last-token pooling. Retrieval bắt buộc phân biệt side — `Query:` / `Document:` prefix — gateway tự thêm theo input `side` (`query`/`document`/`bare`). ~1.04B params, BF16, license CC BY-NC 4.0 — kiểm tra license trước khi dùng thương mại.

Model cũ `jinaai/jina-clip-v2` đã **xoá khỏi máy** (không còn cache, không còn tham chiếu trong code).

## Why not an embeddings API by default

The chat/vision API used for answer generation is not the same thing as a multimodal embedding API. Embeddings chạy qua **gateway riêng** (`http://localhost:8036`, Triton GPU) — không dùng API chat để tạo vector. Query string (side=query) và document/records (side=document) được encode bởi cùng model v5, nên cosine similarity có nghĩa xuyên modality (text↔ảnh dùng chung vector space, nhưng thực nghiệm cho thấy ảnh→ảnh mạnh hơn ảnh→text).

## GPU profile

Triton GPU 3 chạy cả 2 model v5 text/image ở **float16** (`torch_dtype=torch.float16`) — VRAM ~2GB, image encode 0.1–1.2s/ảnh. Batch 8 cho text: ~34 ms/record (110 records ≈ 3.7s). Trước khi fix fp16, model fp32 chiếm 10.8GB và image inference treo do GPU error sau OOM.

## Retrieval strategy

1. Parse mỗi trang, giữ `page`, `printed_page`, `text`, `tables_markdown`, rendered figure assets.
2. Build cheap lexical candidate set (hoặc bỏ qua khi dùng vector).
3. Encode records (text + tables_markdown, side=document) và query (side=query) qua gateway; figure assets encode bằng vision tower (`jina_v5_image`).
4. Chuẩn hóa L2 + cosine similarity; lưu float32 trong `embeddings_v5.npz` (kèm `record_ids` json).
5. Chỉ gửi top-k evidence cho model trả lời. Model trả lời không phải model embedding.

## Models used in this repository

| Role | Default model | Location | Requires API key |
|---|---|---|---|
| Text embedding | `jina_v5_text` (jina-embeddings-v5-omni-nano) | Triton gateway `:8036` | Yes (EMBED_API_KEY) |
| Image embedding | `jina_v5_image` (cùng model) | Triton gateway `:8036` | Yes (EMBED_API_KEY) |
| Text embedding (alternative) | `BAAI/bge-m3` (`tinix-embedding-cosine`, 1024-d) | Remote embedding endpoint (`BGE3_API_URL`) | No |
| Text answer | `openai/gpt-4o-mini` (hoặc model tùy chọn) | OpenRouter / OpenAI-compatible API | Yes |
| Image-aware answer | `openai/gpt-4o-mini` (vision) | OpenRouter / OpenAI-compatible API | Yes |

References: https://huggingface.co/jinaai/jina-embeddings-v5-omni-nano và https://huggingface.co/BAAI/bge-m3.

## Minimal embedding test (qua gateway)

```bash
source reproduce/env.sh
curl -s http://localhost:8036/v1/embeddings \
  -H "Authorization: Bearer $EMBED_API_KEY" -H "Content-Type: application/json" \
  -d '{"model": "jina_v5_text", "input": ["Bảng 3 vị trí của Việt Nam trong xếp hạng đại học"], "side": "query"}'
```

Wrapper repo: `src/embed_jina.py` (`JinaV5Api` cho gateway, `JinaV5` cho transformers local) và `benchmark/infrastructure/embedder.py` (`V5Embedder`/`BGE3Embedder`) — cả hai chuẩn hóa L2 trước khi lưu/similarity.