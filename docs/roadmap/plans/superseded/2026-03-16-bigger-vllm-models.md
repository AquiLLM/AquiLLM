# Bigger vLLM Models Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Swap AquiLLM's configured self-hosted chat, embedding, and reranker models to the larger requested variants while preserving the OpenAI-compatible Mem0 integration, enabling bitsandbytes 8-bit weights plus FP8 KV cache where supported, and raising chat context to `100000`.

**Architecture:** Keep the existing vLLM service topology and OpenAI-compatible base URLs untouched. Update only `.env` model identifiers, vLLM extra args, main chat max length, and the Mem0 embedding dimensions so the app and Mem0 continue to call the same endpoints with more capable backing models. Leave the app's Postgres-backed embedding width at `1024` until a schema migration is added. Keep the reranker on full-precision weights because the current vLLM bitsandbytes sequence-classification path fails for this model.

**Tech Stack:** Docker Compose, vLLM OpenAI server, Django, Mem0, Qdrant, environment-variable based configuration

---

## Chunk 1: Environment Configuration

### Task 1: Update chat model configuration

**Files:**
- Modify: `.env`

- [ ] **Step 1: Update the main chat model ID**

Set:

```env
VLLM_MODEL=DavidAU/Qwen3.5-40B-Claude-4.5-Opus-High-Reasoning-Thinking
MEM0_LLM_VLLM_MODEL=DavidAU/Qwen3.5-40B-Claude-4.5-Opus-High-Reasoning-Thinking
```

- [ ] **Step 2: Preserve OpenAI-compatible app and Mem0 chat URLs**

Keep these values unchanged:

```env
VLLM_BASE_URL=http://vllm:8000/v1
MEM0_LLM_BASE_URL=http://vllm:8000/v1
MEM0_VLLM_BASE_URL=http://vllm:8000/v1
```

- [ ] **Step 3: Verify the diff**

Run: `git diff -- .env`
Expected: only the intended chat-model lines changed in this step.

### Task 2: Update embedding model configuration

**Files:**
- Modify: `.env`

- [ ] **Step 1: Update app and Mem0 embedding model IDs**

Set:

```env
APP_EMBED_MODEL=Qwen/Qwen3-VL-Embedding-8B
MEM0_EMBED_MODEL=Qwen/Qwen3-VL-Embedding-8B
MEM0_EMBED_VLLM_MODEL=Qwen/Qwen3-VL-Embedding-8B
MEM0_EMBED_TOKENIZER=Qwen/Qwen3-VL-Embedding-8B
```

- [ ] **Step 2: Keep app dimensions stable and raise Mem0 dimensions**

Set:

```env
APP_EMBED_DIMS=1024
MEM0_EMBED_DIMS=4096
MEM0_COLLECTION_NAME=mem0_vl_4096_v1
```

- [ ] **Step 3: Preserve OpenAI-compatible embedding URLs**

Keep these values unchanged:

```env
APP_EMBED_BASE_URL=http://vllm_embed:8000/v1
MEM0_EMBED_BASE_URL=http://vllm_embed:8000/v1
```

- [ ] **Step 4: Verify the diff**

Run: `git diff -- .env`
Expected: embedding model IDs updated, Mem0 dimensions raised, app dimensions unchanged at `1024`, and base URLs unchanged.

### Task 3: Update reranker model configuration

**Files:**
- Modify: `.env`

- [ ] **Step 1: Update reranker model ID and tokenizer**

Set:

```env
APP_RERANK_MODEL=Qwen/Qwen3-VL-Reranker-8B
APP_RERANK_VLLM_MODEL=Qwen/Qwen3-VL-Reranker-8B
APP_RERANK_TOKENIZER=Qwen/Qwen3-VL-Reranker-8B
```

- [ ] **Step 2: Preserve current reranker override pattern**

Keep:

```env
APP_RERANK_VLLM_TRUST_REMOTE_CODE=1
APP_RERANK_VLLM_EXTRA_ARGS=--runner pooling --dtype float16 --swap-space 16 --chat-template /templates/qwen3_vl_reranker.jinja --hf-overrides {"architectures":["Qwen3VLForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}
```

- [ ] **Step 3: Verify the diff**

Run: `git diff -- .env`
Expected: reranker model lines updated, override wiring unchanged.

## Chunk 2: Verification and Handoff

### Task 4: Verify final configuration state

**Files:**
- Modify: `.env`

- [ ] **Step 1: Confirm key environment values**

Run:

```bash
rg -n "VLLM_MODEL|VLLM_MAX_MODEL_LEN|VLLM_EXTRA_ARGS|APP_EMBED_MODEL|APP_EMBED_DIMS|APP_RERANK_MODEL|APP_RERANK_VLLM_EXTRA_ARGS|MEM0_LLM_VLLM_MODEL|MEM0_EMBED_MODEL|MEM0_EMBED_DIMS|MEM0_EMBED_VLLM_EXTRA_ARGS" .env
```

Expected: the larger requested models are configured, `VLLM_MAX_MODEL_LEN=100000`, chat and embed extra-arg lines include `--kv-cache-dtype fp8` plus bitsandbytes 8-bit loading, the reranker extra-arg line includes `--kv-cache-dtype fp8` without bitsandbytes flags, `MEM0_EMBED_DIMS=4096`, and `APP_EMBED_DIMS=1024`.

- [ ] **Step 2: Review final diff**

Run:

```bash
git diff -- .env
```

Expected: only the intended model/config lines changed.

- [ ] **Step 3: Note operational follow-up**

Document that existing retrieval and memory vectors should be rebuilt after deployment because the embedding model changed, and that a future pgvector schema migration is required before the app can use `4096`-dimensional document embeddings directly.
