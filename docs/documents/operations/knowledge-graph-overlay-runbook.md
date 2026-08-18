# Knowledge Graph Overlay Runbook

Status: initial runtime identity record. Task 19 of the implementation plan will
extend this document with configuration, rollout, rollback, retention, and
ownership procedures. Knowledge-graph builds and retrieval remain disabled by
default.

### Durable embedding model identity

Collection graph builds require `APP_EMBED_MODEL_REVISION` to identify the
exact embedding checkpoint or a provider-attested immutable model snapshot.
The value is part of every collection artifact and embedding audit signature,
alongside a non-secret digest of the normalized provider endpoint, the
provider/model name, 1024 dimensions, preprocessing version, maximum input
length, and batch size. An endpoint change therefore also fails an in-flight
build closed. An empty revision value fails durable graph builds closed. This
is intentional: do not invent a revision for a mutable remote alias such as
`text-embedding-3-small`. The no-GPU OpenAI profile explicitly requests 1024
dimensions; ordinary non-KG query embeddings keep their existing
availability-oriented behavior.

For the self-hosted profile, compose uses `APP_EMBED_MODEL` as both the actual
vLLM checkpoint and served model name, and passes
`APP_EMBED_MODEL_REVISION` to vLLM's `--revision` option. Qwen3-VL-Embedding-2B
documents MRL output dimensions from 64 through 2048, so the strict KG adapter
always sends `dimensions=1024` independently of the ordinary app compatibility
flag; it rejects any response that is not exactly 1024 values or reports a
different served model. Compose also pins the tokenizer to the same APP model
identity. Keep the Mem0 model setting aligned with this shared endpoint. See the
[model card](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) and vLLM's
[embedding request protocol](https://docs.vllm.ai/en/v0.9.1/api/vllm/entrypoints/openai/protocol.html#vllm.entrypoints.openai.protocol.EmbeddingCompletionRequest).

## Runtime identity (v1)

The local extractor is an optional, worker-only runtime. Its v1 identity is:

| Component | Pinned identity |
| --- | --- |
| Python package and extra | `gliner2[local]==1.3.2` |
| Model | `fastino/gliner2-base-v1` |
| Immutable model revision | `8437ba583a733d87f56ae902f3b197934eedd58e` |

The resolved dependency set retains `torch==2.11.0`,
`transformers==5.3.0`, and `tokenizers==0.22.2`, and adds `peft==0.20.0`.
Dependency resolution succeeds, but that does not prove runtime API or model
compatibility. Task 6 uses fake modules without network access and verifies only
the provider API contract. Task 18 implements the
`check_knowledge_graph_extractor` command; Task 21 runs the real pinned package
and checkpoint smoke check in the optional worker. That Task 21 smoke must pass
before this identity is treated as an operationally validated extractor.

### Optional installation and dependency verification

Install the extra only in the dedicated knowledge-graph worker environment:

```bash
uv sync --extra knowledge-graph-local
```

Verify the lock and installed package identities without downloading the model:

```bash
uv lock --check
uv run --extra knowledge-graph-local python -c "from importlib.metadata import version; print(version('gliner2'), version('peft'))"
```

The expected output from the Python check is `1.3.2 0.20.0`. Model checkpoint
download and inference verification are intentionally deferred: Task 18 adds
`check_knowledge_graph_extractor`, and Task 21 executes it in the optional
worker using the exact model and revision above.

### Default web installation exclusion

The `knowledge-graph-local` extra is not part of AquiLLM's default dependency
set. A normal `uv sync` does not install it, and `requirements.txt` intentionally
does not include GLiNER2. Web, ASGI, Django model, and Celery task-registration
imports must continue to work when GLiNER2 and its optional ML modules are
unavailable. Only the dedicated knowledge-graph worker image may opt into this
extra.
