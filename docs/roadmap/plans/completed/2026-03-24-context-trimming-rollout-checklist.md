# Smart context trimming rollout checklist

Companion to `2026-03-24-smarter-context-trimming-implementation.md`.

## Profile A — baseline (length preflight only)

- `CONTEXT_PACKER_ENABLED=0`
- `TOKEN_EFFICIENCY_ENABLED=1` (if you use cross-provider preflight today)
- `PROMPT_BUDGET_CONTEXT_LIMIT` aligned with your model / `OPENAI_CONTEXT_LIMIT` / `VLLM_MAX_MODEL_LEN`

## Profile B — salience-aware packer

- `CONTEXT_PACKER_ENABLED=1`
- Same `PROMPT_BUDGET_CONTEXT_LIMIT` / `OPENAI_CONTEXT_LIMIT` as Profile A for a fair comparison
- Optional: `TOOL_SEARCH_COMPACT_PAYLOAD=1` for shorter vector-search JSON (models must tolerate short keys `r`,`i`,`d`,`c`,`n`,`x`,`u`,`ty`)

## Before widening traffic

1. Run targeted pytest (see `2026-03-23-caching-rag-token-efficiency-rollout-notes.md` regression list).
2. Compare on a fixed prompt set (50–100 chats): overflow rate, mean prompt tokens, p95 time-to-first-token, qualitative answer quality.
3. Watch logs for `context_pack stats` (counts/tokens only); confirm no prompt bodies in log lines.

## Rollback

- Set `CONTEXT_PACKER_ENABLED=0` and redeploy; no code revert required.
