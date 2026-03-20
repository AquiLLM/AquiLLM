# Bigger vLLM Models Design

**Date:** 2026-03-16

## Goal

Replace the currently configured self-hosted chat, embedding, and reranker models with the larger variants the user selected, while preserving the existing OpenAI-compatible vLLM endpoints used by AquiLLM and Mem0.

## Scope

- Update the main chat model to `DavidAU/Qwen3.5-40B-Claude-4.5-Opus-High-Reasoning-Thinking`.
- Update the app embedding model and Mem0 embedding model to `Qwen/Qwen3-VL-Embedding-8B`.
- Update the app reranker model to `Qwen/Qwen3-VL-Reranker-8B`.
- Raise Mem0 embedding dimensions to match the larger embedding model.
- Enable bitsandbytes 8-bit weight loading for the chat and embedder vLLM services.
- Enable FP8 KV cache for the chat, embedder, and reranker vLLM services.
- Raise only the main chat model context limit to `100000`.
- Preserve the current service topology and OpenAI-compatible URLs:
  - `VLLM_BASE_URL`
  - `APP_EMBED_BASE_URL`
  - `APP_RERANK_BASE_URL`
  - `MEM0_LLM_BASE_URL`
  - `MEM0_EMBED_BASE_URL`

## Non-Goals

- No Docker Compose topology changes.
- No GPU memory retuning beyond what is already present in `.env`.
- No retrieval quality tuning, prompt tuning, or throughput tuning.
- No embedder or reranker max-length increase in this change.
- No migration of old embeddings in this change.

## Design

The change is configuration-only. The existing vLLM services already provide the OpenAI-compatible boundary required by Mem0 and the app. We will keep those endpoints and swap only the backing model IDs, quantization flags, chat context limit, and related embedding dimensions.

Because the embedding model changes to a larger model with a different vector shape, existing document embeddings and Mem0/Qdrant vectors should be treated as stale and rebuilt after deployment. That rebuild is a follow-up operational step, not part of this config patch.

The app's Postgres-backed document embedding fields are still `vector(1024)`, so `APP_EMBED_DIMS` must remain `1024` for now to avoid schema mismatches. Mem0 uses Qdrant instead, so it can move to a new `4096`-dimensional collection immediately.

The reranker remains full precision for weights. In this environment and vLLM version, `Qwen/Qwen3-VL-Reranker-8B` fails to load through the bitsandbytes 8-bit sequence-classification path with a classification-head shape mismatch, so only FP8 KV cache is retained for that service.

## Verification

- Confirm `.env` now references the larger chat, embedding, and reranker model IDs.
- Confirm Mem0 embedding dimension settings reflect the larger embedder configuration.
- Confirm `APP_EMBED_DIMS` remains `1024` until the pgvector schema is migrated.
- Confirm the main chat model uses `VLLM_MAX_MODEL_LEN=100000`.
- Confirm the chat and embedder extra args specify `--kv-cache-dtype fp8` and bitsandbytes 8-bit loading.
- Confirm the reranker extra args specify `--kv-cache-dtype fp8` without bitsandbytes flags.
- Confirm the Mem0 and app base URLs remain unchanged and continue to point at OpenAI-compatible vLLM endpoints.

## References

- https://huggingface.co/DavidAU/Qwen3.5-40B-Claude-4.5-Opus-High-Reasoning-Thinking
- https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B
- https://huggingface.co/Qwen/Qwen3-VL-Reranker-8B
