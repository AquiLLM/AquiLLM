"""Application-specific Prometheus metrics for AquiLLM runtime paths."""
from prometheus_client import Counter, Gauge, Histogram

# ── RAG / Chunk search ────────────────────────────────────────────────────
chunk_search_duration = Histogram(
    "aquillm_chunk_search_duration_seconds",
    "Chunk search latency by phase",
    ["phase"],  # "vector", "trigram", "rerank", "total"
)

rag_cache_ops = Counter(
    "aquillm_rag_cache_ops_total",
    "RAG cache operations",
    ["metric", "result"],  # result: "hit" or "miss"
)

# ── Context packing ──────────────────────────────────────────────────────
context_pack_tokens = Gauge(
    "aquillm_context_pack_tokens",
    "Token count during context packing",
    ["stage"],  # "before", "after"
)

# ── Chat ─────────────────────────────────────────────────────────────────
chat_latency = Histogram(
    "aquillm_chat_latency_seconds",
    "Chat processing latency by phase",
    ["phase"],  # "memory_augmentation", "llm_spin", "delta_persist"
)

# ── Ingestion ────────────────────────────────────────────────────────────
ingestion_items = Counter(
    "aquillm_ingestion_items_total",
    "Ingestion items processed",
    ["source", "status"],  # source: "upload", "web", "arxiv"; status: "queued", "rejected"
)
