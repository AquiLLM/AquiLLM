# Knowledge Graph Overlay Runbook

Status: initial runtime identity record. Task 19 of the implementation plan will
extend this document with configuration, rollout, rollback, retention, and
ownership procedures. Knowledge-graph builds and retrieval remain disabled by
default.

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
compatibility. Task 6's fake-provider and pinned-runtime smoke checks must pass
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
download and inference verification are intentionally deferred to the Task 6
smoke check, which must use the exact model and revision above.

### Default web installation exclusion

The `knowledge-graph-local` extra is not part of AquiLLM's default dependency
set. A normal `uv sync` does not install it, and `requirements.txt` intentionally
does not include GLiNER2. Web, ASGI, Django model, and Celery task-registration
imports must continue to work when GLiNER2 and its optional ML modules are
unavailable. Only the dedicated knowledge-graph worker image may opt into this
extra.
