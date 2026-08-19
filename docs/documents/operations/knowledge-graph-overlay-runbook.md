# Knowledge Graph Overlay Runbook

**Status:** rollout blocked; measured gates are `PENDING_MEASUREMENT`

**Shipping state:** `KG_BUILD_ENABLED=0`, `KG_OVERLAY_ENABLED=0`

**Scope:** collection-scoped graph indexing and permission-filtered RAG expansion

The knowledge graph is a rebuildable index over existing documents and chunks.
It is not a source of truth and it never grants access. Vector, trigram, and
exact retrieval remain the baseline. If graph configuration, storage, or
ranking fails, retrieval returns that baseline without graph candidates.

## Safety invariants

- Build and retrieval controls are independent and off by default.
- Only documents already authorized for the request can seed or receive an
  expansion. Both relation endpoints are permission-filtered before traversal.
- Canonical entities are an internal identity registry. They have no chunks,
  relations, or access grants and are not a user-enumerable global graph.
- Every returned result is a real, authorized `TextChunk`. Graph nodes, scores,
  labels, and triples never become public result rows or citations.
- The overlay is bounded by fixed v1 ceilings and a 150 ms total local-database
  deadline. It performs no extraction, model inference, or network request at
  query time.
- Activations are atomic. Retrieval uses one read-only repeatable-read snapshot,
  so a concurrent rebuild yields one coherent graph version or a graph miss.
- Pruning never removes active/building state, source documents, or chunks.
- `KG_EVAL_BYPASS_ALLOWED` remains `0` in every deployed environment. The
  debug/test-only bypass is not a production rollout mechanism.

## Configuration contract

Copy the knowledge-graph block from `.env.example` without changing the two
feature flags. Invalid graph settings disable or fail only graph work according
to the fail-open boundary; they must not make the ordinary web application or
baseline retrieval unavailable.

### Build, extractor, and lifecycle controls

| Variable | Shipping value | Operational rule |
| --- | ---: | --- |
| `KG_BUILD_ENABLED` | `0` | Enable before backfill; independent of retrieval. |
| `KG_OVERLAY_ENABLED` | `0` | Enable only after all measured gates pass. |
| `KG_EXTRACTOR_PROVIDER` | `gliner2_local` | v1 provider; loaded only in the optional worker. |
| `KG_EXTRACTOR_FAIL_OPEN` | `1` | Extraction failure must not break ingest or baseline RAG. |
| `KG_EXTRACTION_QUEUE` | `knowledge-graph-extraction` | Dedicated worker queue; use the same value in routing, commands, and Compose. |
| `KG_GLINER2_MODEL` | `fastino/gliner2-base-v1` | Immutable model identity is paired with the revision below. |
| `KG_GLINER2_REVISION` | `8437ba583a733d87f56ae902f3b197934eedd58e` | Never replace with a branch, tag, or `latest`. |
| `KG_GLINER2_DEVICE` | `cpu` | Shipping extraction device. |
| `KG_GLINER2_BATCH_SIZE` | `8` | Bounded provider batch. |
| `KG_GLINER2_CACHE_DIR` | `/root/.cache/huggingface` | Persist this directory for the optional worker. |
| `KG_GLINER2_LOCAL_FILES_ONLY` | `0` | Use for controlled prefetch, then use `1` for offline startup only with a durable cache. |
| `KG_ARTIFACT_RETENTION_DAYS` | `30` | Minimum terminal age before a row may be considered for pruning. |
| `KG_ARTIFACT_KEEP_SUPERSEDED` | `2` | Retain at least the configured newest superseded generations per logical scope. |
| `KG_EVAL_BYPASS_ALLOWED` | `0` | Must remain `0` outside explicit debug/test evaluation. |

Collection graph builds also require a non-empty immutable
`APP_EMBED_MODEL_REVISION`. See
[Durable embedding model identity](#durable-embedding-model-identity).

### Retrieval algorithm and hard envelope

The shipping values below are the measured rollout configuration. With the
exception of RRF `k` and the restart probability, each maximum is also the
immutable v1 hard ceiling. A deployment may lower a bounded value, but doing so
changes the algorithm signature and is not covered by the shipping gates.
Production rollout requires `MAX_HOPS=2` and `PPR_ITERATIONS=8` unless a new
comparison report and gate approval cover another configuration.

| Variable | Shipping value | Valid v1 range |
| --- | ---: | ---: |
| `KG_OVERLAY_ALGORITHM` | `ppr_v1` | exact value only |
| `KG_OVERLAY_RRF_K` | `60` | `1..1000` |
| `KG_OVERLAY_MAX_SEEDS` | `64` | `1..64` |
| `KG_OVERLAY_MAX_SCOPE_DOCUMENTS` | `10000` | `1..10000` |
| `KG_OVERLAY_MAX_SCOPE_COLLECTIONS` | `128` | `1..128` |
| `KG_OVERLAY_MAX_HOPS` | `2` | `1..2` |
| `KG_OVERLAY_MAX_FANOUT` | `10` | `1..10` |
| `KG_OVERLAY_MAX_NODES` | `200` | `1..200` |
| `KG_OVERLAY_MAX_EDGES` | `1000` | `1..1000` |
| `KG_OVERLAY_MAX_EVIDENCE_ROWS` | `3000` | `1..3000` |
| `KG_OVERLAY_MAX_EVIDENCE_PER_EDGE` | `3` | `1..3` |
| `KG_OVERLAY_MAX_MENTIONS_PER_ENTITY` | `2` | `1..2` |
| `KG_OVERLAY_PPR_RESTART` | `0.20` | finite and strictly between `0` and `1` |
| `KG_OVERLAY_PPR_ITERATIONS` | `8` | `1..8` |
| `KG_OVERLAY_MAX_CANDIDATES` | `20` | `1..20` |
| `KG_OVERLAY_MAX_PER_DOCUMENT` | `3` | `1..3` |
| `KG_OVERLAY_TIMEOUT_MS` | `150` | `1..150` |

The following cross-field rules also apply before any ORM query:

- scope collections must not exceed scope documents;
- edges must not exceed nodes multiplied by fan-out;
- per-edge evidence must not exceed total evidence;
- per-document candidates must not exceed total candidates.

The lowercase SHA-256 algorithm signature covers every effective value above,
the canonical resolver version, and the frozen RRF/PPR transition, evidence,
and seed-scoring versions. Any effective change requires new measurement and
approval. Request seeds, authorization scopes, and active graph rows instead
belong to the request/snapshot graph-version signatures.

## Durable embedding model identity

Collection graph builds require `APP_EMBED_MODEL_REVISION` to identify the
exact embedding checkpoint or a provider-attested immutable model snapshot.
The value is part of every collection artifact and embedding audit signature,
alongside a non-secret digest of the normalized provider endpoint, the
provider/model name, 1024 dimensions, preprocessing version, maximum input
length, and batch size. An endpoint change therefore fails an in-flight build
closed. An empty revision fails durable graph builds closed. Do not invent a
revision for a mutable remote alias such as `text-embedding-3-small`.

For the self-hosted profile, Compose uses `APP_EMBED_MODEL` as both the actual
vLLM checkpoint and served model name and passes `APP_EMBED_MODEL_REVISION` to
vLLM's `--revision` option. The strict graph adapter always requests exactly
1024 dimensions and rejects any response with another width or served model.
Keep the tokenizer and Mem0 client model setting aligned with the shared
endpoint, while keeping Mem0 lifecycle and storage separate from this overlay.

## Optional worker and immutable extractor

The local extractor is installed only in the dedicated knowledge-graph worker.
Its v1 runtime identity is:

| Component | Pinned identity |
| --- | --- |
| Python package and extra | `gliner2[local]==1.3.2` |
| Model | `fastino/gliner2-base-v1` |
| Immutable model revision | `8437ba583a733d87f56ae902f3b197934eedd58e` |

The resolved environment retains `torch==2.11.0`, `transformers==5.3.0`, and
`tokenizers==0.22.2`, and adds `peft==0.20.0`. Ordinary web, ASGI, Django model,
and Celery task-registration imports must work without this optional runtime.

For a local dependency-only check:

```bash
uv sync --extra knowledge-graph-local
uv lock --check
uv run --extra knowledge-graph-local python -c "from importlib.metadata import version; print(version('gliner2'), version('peft'))"
```

The expected versions are `1.3.2 0.20.0`.

For the deployable worker, use the same Compose file and environment throughout
the prefetch and smoke check:

```bash
docker compose --env-file .env -f deploy/compose/development.yml --profile knowledge-graph build worker_knowledge_graph
docker compose --env-file .env -f deploy/compose/development.yml --profile knowledge-graph run --rm --no-deps --entrypoint uv worker_knowledge_graph pip check --python /opt/venv/bin/python
docker compose --env-file .env -f deploy/compose/development.yml --profile knowledge-graph run --rm --no-deps --entrypoint /opt/venv/bin/python worker_knowledge_graph manage.py check_knowledge_graph_extractor
```

The final command intentionally loads the configured checkpoint, checks exact
entity/relation spans on a smoke fixture, and prints only provider/package/model
identity. Run it first with controlled network access and a persistent cache.
If the deployed worker must be offline, set `KG_GLINER2_LOCAL_FILES_ONLY=1`
only after prefetch and repeat the exact smoke command to prove the cache is
complete. Never enable builds when this check fails.

## Mandatory rollout

Do not skip or reorder these steps.

1. **Build the optional worker image.** Use the Compose build and `pip check`
   commands above. Keep both feature flags `0`.
2. **Prefetch and verify the immutable checkpoint.** Run the extractor check
   against the exact model/revision and, if applicable, repeat offline.
3. **Start the dedicated worker.** Confirm it consumes only the configured
   extraction queue:

   ```bash
   docker compose --env-file .env -f deploy/compose/development.yml --profile knowledge-graph up -d worker_knowledge_graph
   ```

4. **Enable builds only.** Set `KG_BUILD_ENABLED=1` and keep
   `KG_OVERLAY_ENABLED=0`; restart every service that reads the setting. Confirm
   ordinary retrieval is unchanged.
5. **Backfill one explicitly approved representative collection.** Use its real
   positive integer primary key; preview first, then submit one durable request:

   ```bash
   docker compose --env-file .env -f deploy/compose/development.yml exec web /opt/venv/bin/python manage.py rebuild_knowledge_graph --collection <positive-integer-pk> --dry-run
   docker compose --env-file .env -f deploy/compose/development.yml exec web /opt/venv/bin/python manage.py rebuild_knowledge_graph --collection <positive-integer-pk>
   docker compose --env-file .env -f deploy/compose/development.yml exec web /opt/venv/bin/python manage.py inspect_knowledge_graph --request-id <printed-request-uuid> --wait --timeout-seconds 1800
   ```

   Run operational management commands inside the selected deployment's
   Compose network. The application settings use the service DNS names `db`
   and `redis`; an unconfigured host Python process cannot reach them. Use the
   same reviewed Compose file for every command in one rollout.

6. **Inspect quality and failures.** Review bounded lifecycle state and the
   aggregate failed/stale counts; use private source data under normal access
   controls to sample extraction, resolution, filtering, aliases, relation
   direction, and candidate identity decisions:

   ```bash
   docker compose --env-file .env -f deploy/compose/development.yml exec web /opt/venv/bin/python manage.py inspect_knowledge_graph --collection <positive-integer-pk>
   ```

7. **Run the Task 20 one-snapshot comparison and record gates.** Never enable the evaluation bypass on the deployed graph worker. Its normal queue and
   `DJANGO_DEBUG=0` boundary must remain unchanged. Instead, create one unique,
   safe temporary queue and run a separate worker plus every evaluation command
   on that queue. The deployed worker cannot consume these tasks because it
   continues to consume only its configured non-evaluation queue.

   The following Bash procedure uses the same Compose network, disables both
   shipping flags in every temporary process, gives the report restrictive
   creation permissions, waits for both requests, and removes the worker and
   purges its unique queue on success, interruption, or failure. Replace the
   fixture placeholders with two through four approved collection IDs and
   caller-generated UUIDs. `--no-deps` reuses the already-running deployment
   database and broker and creates no temporary dependency containers.

   ```bash
   set -euo pipefail
   install -d -m 700 artifacts
   KG_EVAL_QUEUE="knowledge-graph-eval-$(date +%s)-$$"
   KG_EVAL_WORKER_NAME="aquillm-kg-eval-$(date +%s)-$$"

   cleanup_kg_eval() {
     trap - EXIT INT TERM
     docker rm -f "$KG_EVAL_WORKER_NAME" >/dev/null 2>&1 || true
     docker compose --env-file .env -f deploy/compose/development.yml --profile knowledge-graph run --rm --no-deps \
       -e KG_EXTRACTION_QUEUE="$KG_EVAL_QUEUE" \
       --entrypoint /opt/venv/bin/celery worker_knowledge_graph \
       -A aquillm purge -Q "$KG_EVAL_QUEUE" -f >/dev/null 2>&1 || true
     unset KG_EVAL_QUEUE KG_EVAL_WORKER_NAME
     unset -f kg_eval_python 2>/dev/null || true
   }
   trap 'status=$?; cleanup_kg_eval; exit "$status"' EXIT
   trap 'cleanup_kg_eval; exit 130' INT
   trap 'cleanup_kg_eval; exit 143' TERM

   docker compose --env-file .env -f deploy/compose/development.yml --profile knowledge-graph run -d --name "$KG_EVAL_WORKER_NAME" --no-deps \
     -e KG_EXTRACTION_QUEUE="$KG_EVAL_QUEUE" \
     -e DJANGO_DEBUG=1 \
     -e KG_EVAL_BYPASS_ALLOWED=1 \
     -e KG_BUILD_ENABLED=0 \
     -e KG_OVERLAY_ENABLED=0 \
     worker_knowledge_graph

   kg_eval_python() {
     docker compose --env-file .env -f deploy/compose/development.yml --profile knowledge-graph run --rm --no-deps \
       --user "$(id -u):$(id -g)" \
       -e KG_EXTRACTION_QUEUE="$KG_EVAL_QUEUE" \
       -e DJANGO_DEBUG=1 \
       -e KG_EVAL_BYPASS_ALLOWED=1 \
       -e KG_BUILD_ENABLED=0 \
       -e KG_OVERLAY_ENABLED=0 \
       --entrypoint /bin/sh worker_knowledge_graph \
       -c 'umask 077; exec /opt/venv/bin/python "$@"' sh "$@"
   }

   for attempt in $(seq 1 60); do
     if docker logs "$KG_EVAL_WORKER_NAME" 2>&1 | grep -q ' ready\.'; then
       break
     fi
     if [ "$attempt" -eq 60 ]; then
       echo "isolated evaluation worker did not become ready" >&2
       exit 1
     fi
     sleep 1
   done

   kg_eval_python manage.py rebuild_knowledge_graph --collection <pk-a> --request-id <uuid-a> --eval-only
   kg_eval_python manage.py inspect_knowledge_graph --request-id <uuid-a> --wait --timeout-seconds 1800
   kg_eval_python manage.py rebuild_knowledge_graph --collection <pk-b> --request-id <uuid-b> --eval-only
   kg_eval_python manage.py inspect_knowledge_graph --request-id <uuid-b> --wait --timeout-seconds 1800
   kg_eval_python -m apps.knowledge_graph.evals.run_kg_eval --mode comparison --eval-only --collection <pk-a> --collection <pk-b> --output /app/artifacts/kg-eval-comparison.json
   kg_eval_python -m apps.knowledge_graph.evals.run_kg_eval --write-measured-gates --comparison-report /app/artifacts/kg-eval-comparison.json --runbook /app/docs/documents/operations/knowledge-graph-overlay-runbook.md
   kg_eval_python -m apps.knowledge_graph.evals.run_kg_eval --verify-gates --comparison-report /app/artifacts/kg-eval-comparison.json --runbook /app/docs/documents/operations/knowledge-graph-overlay-runbook.md

   cleanup_kg_eval
   ```

   The comparison must contain vector-only, one-hop, and shipping `ppr_v1` arms
   from the same authorized scope, fused seeds, and repeatable-read graph
   snapshot. The root `artifacts/` directory is restricted operator-only output:
   it is excluded from Git and the Docker build context, but operators must also
   limit host access, retain it only for the approved audit period, and securely
   delete it when that period ends. Never commit, publish, or copy the private
   scope/trace report into an image, public artifact store, log, or tool payload.
8. **Enable retrieval in development or staging.** Only after gate verification
   exits zero, set `KG_OVERLAY_ENABLED=1` in a non-production deployment and
   restart its readers. Do not change the measured shipping caps.
9. **Soak and monitor.** Observe latency, graph hit/miss/error/timeout statuses,
   build failures, stale artifacts, queue depth, disk growth, and citation
   coverage. Confirm error/miss requests exactly preserve the baseline.
10. **Enable production retrieval selectively.** Obtain the numeric-gate
    approver's sign-off, then enable a bounded production deployment or cohort.
    Expand only after the soak remains inside the approved gates. Do not use a
    client-supplied flag or the evaluation bypass.

Operator-wide backfill is a separate, high-impact action. The operations owner
must preview and explicitly confirm it:

```bash
docker compose --env-file .env -f deploy/compose/development.yml exec web /opt/venv/bin/python manage.py rebuild_knowledge_graph --all --dry-run
docker compose --env-file .env -f deploy/compose/development.yml exec web /opt/venv/bin/python manage.py rebuild_knowledge_graph --all --yes
```

## Measured retrieval gates

Task 20 atomically writes reviewed values from one comparison bundle. Until no
row remains pending and `--verify-gates` succeeds, retrieval must remain off.

| Gate | Required outcome | Current value | Status |
| --- | --- | --- | --- |
| Permission isolation | zero inaccessible chunks | `PENDING_MEASUREMENT` | `PENDING_MEASUREMENT` |
| Fail-open parity | exact baseline on graph miss/error | `PENDING_MEASUREMENT` | `PENDING_MEASUREMENT` |
| Identity precision | automatic links stricter than candidates | `PENDING_MEASUREMENT` | `PENDING_MEASUREMENT` |
| Retrieval quality | positive Recall@10 and nDCG movement on relationship/alias/cross-document/cross-collection cases | `PENDING_MEASUREMENT` | `PENDING_MEASUREMENT` |
| Multi-hop value | PPR no worse than one-hop and better on at least one distance-two case | `PENDING_MEASUREMENT` | `PENDING_MEASUREMENT` |
| Latency | graph p95 within the configured local-DB budget | `PENDING_MEASUREMENT` | `PENDING_MEASUREMENT` |
| Determinism | repeated PPR ranking is identical | `PENDING_MEASUREMENT` | `PENDING_MEASUREMENT` |
| Citations | 100% curated real-chunk evidence coverage | `PENDING_MEASUREMENT` | `PENDING_MEASUREMENT` |

The report is operator-only. It may contain the approved collection scope and
private evaluation trace, but operational logs, diagnostics, tool payloads, and
citations must never expose those values. Keep it only in the ignored,
access-controlled root `artifacts/` directory for the approved audit-retention
period, then securely delete it.

## Monitoring and inspection

Use `inspect_knowledge_graph`; do not diagnose by dumping entity or relation
tables. Its output is bounded and privacy-safe: opaque IDs, versions, lifecycle
statuses, and aggregate counts only.

```bash
docker compose --env-file .env -f deploy/compose/development.yml exec web /opt/venv/bin/python manage.py inspect_knowledge_graph --document <document-uuid>
docker compose --env-file .env -f deploy/compose/development.yml exec web /opt/venv/bin/python manage.py inspect_knowledge_graph --collection <positive-integer-pk>
docker compose --env-file .env -f deploy/compose/development.yml exec web /opt/venv/bin/python manage.py inspect_knowledge_graph --request-id <request-uuid>
docker compose --env-file .env -f deploy/compose/development.yml exec web /opt/venv/bin/python manage.py inspect_knowledge_graph --request-id <request-uuid> --wait --timeout-seconds 1800
```

### Resnapshot recovery

Inspection may report a durable `PARTIAL` request after authorized source scope
changes during publication:

- `resnapshot_pending` means successor reconciliation did not complete. Reissue
  the original rebuild command with the same scope and exact `--request-id`.
- `resnapshot_churn` means bounded reconciliation observed continued live scope
  drift. Let the move/deletion activity settle, inspect again, and reissue with
  the same `--request-id`.
- an operator-wide parent remains `RUNNING` while a child is waiting for either
  recovery. Resume the whole request with its original parent ID, then wait on
  that same parent:

  ```bash
  docker compose --env-file .env -f deploy/compose/development.yml exec web /opt/venv/bin/python manage.py rebuild_knowledge_graph --all --yes --request-id <original-parent-request-uuid>
  docker compose --env-file .env -f deploy/compose/development.yml exec web /opt/venv/bin/python manage.py inspect_knowledge_graph --request-id <original-parent-request-uuid> --wait --timeout-seconds 1800
  ```
- `scope_deleted` and `scope_ineligible` are final outcomes for the original
  scope. Do not loop the request; correct the source state and submit a new
  explicit request only if the scope becomes eligible again.

Never replace a resumable request with a fresh UUID or repair lifecycle rows by
hand. Same-ID reconciliation preserves the durable audit and idempotency fence.
Use the top-level privacy-safe `request_error_code` from inspection for this
decision; `builds[*].error_code` is run-level state and is not interchangeable.

Alert on sustained extraction queue depth, rebuild requests that do not become
terminal, failed build runs, stale artifact growth, graph timeout/error rate,
latency regression, disk growth, and citation-evidence regression. Graph
diagnostics may contain only `graph_ms`, seed/candidate counts, status, and
lowercase algorithm/version signatures. Never log queries, graph labels,
node/edge evidence, authorized scope, or inaccessible-neighbor counts.

## Rollback

Rollback order is mandatory and does not delete data:

1. Set `KG_OVERLAY_ENABLED=0` and restart every retrieval reader. Verify
   baseline vector/trigram/exact retrieval and citations.
2. Set `KG_BUILD_ENABLED=0` and restart ingestion/build publishers. In-flight
   durable state remains inspectable; do not mutate rows manually.
3. Stop the optional worker:

   ```bash
   docker compose --env-file .env -f deploy/compose/development.yml --profile knowledge-graph stop worker_knowledge_graph
   ```

Keep graph artifacts, build runs, and rebuild requests for diagnosis and
recovery. Retain the comparison report only in the ignored, access-controlled
operator directory and only for its approved audit period. A later rollout must
repeat the extractor check, representative rebuild, comparison, and gate
approval if code, ontology, resolver, model identity, or effective ranking
configuration changed.

## Retention and pruning

`prune_graph_artifacts_task` is a low-priority task routed to the configured
extraction queue. It is intentionally absent from the default beat schedule.
The deployment owner must add an explicit periodic schedule only after reviewing
the preview in that environment; retention maintenance is independent of
`KG_BUILD_ENABLED`.

Preview is the command default. Preview and execution use the same deterministic
bounded plan:

```bash
docker compose --env-file .env -f deploy/compose/development.yml exec web /opt/venv/bin/python manage.py prune_knowledge_graph --batch-size 100
docker compose --env-file .env -f deploy/compose/development.yml exec web /opt/venv/bin/python manage.py prune_knowledge_graph --execute --batch-size 100
```

Before execution, record the preview counts and confirm all of these invariants:

- active and building artifacts/runs are never candidates;
- the highest build generation for every logical scope/build kind is retained;
- the configured newest superseded generations are retained;
- rows with live leases, nonterminal runs or requests, canonical links, or
  pinned collection manifests are retained;
- a terminal successful request has no artifact foreign key; its immutable
  artifact/run/build/source/signature audit remains after terminal pruning;
- terminal age is derived from the status-specific immutable artifact timestamp
  and every exact attached terminal run, never merely `updated_at`;
- deletion is bounded, locked in the lifecycle order, and rechecked before the
  dependent graph rows cascade;
- source `TextChunk`, document, figure, collection, and object-storage data are
  outside the deletion boundary.

For emergency disk pressure, first apply the rollback order. Do not issue ad-hoc
SQL or remove active/building rows. The pruning owner may propose a lower
positive retention value or superseded keep count, obtain incident approval,
restart the maintenance process with that exact configuration, run and save a
fresh preview, and execute small batches while monitoring locks and disk. Rows
protected by high-water or live-reference rules remain protected. Rebuilding
from source is the recovery path for eligible derived artifacts, but audit-bound
references must not be broken to reclaim space.

## Ownership and approval

Assign named people or on-call roles for each responsibility before rollout.

| Responsibility | Required owner and authority |
| --- | --- |
| Ontology activation | Ontology owner; reviews the checksum-addressed YAML and may activate a new immutable ontology version. |
| Candidate identity review | Graph identity/data-quality reviewer; may review suppressed candidates but may not bypass automatic-link rules. |
| Numeric retrieval gates | Retrieval quality approver; reviews the atomic Task 20 report and approves written gate values. |
| Representative and operator-wide backfill | Knowledge-graph operations owner; selects explicit scopes, previews impact, monitors the dedicated queue, and alone authorizes `--all --yes`. |
| Pruning execution | Storage/lifecycle operations owner; reviews dry-run output, retention exceptions, live references, and executes bounded batches. |
| Production enablement/rollback | Service owner or incident commander; changes the two shipping flags in the required order and records the deployment/cohort. |

No single extractor result, candidate identity link, or unreviewed evaluation
report grants production activation authority.

## v1 non-goals

- Mem0 remains a separate conversation/user-memory system with separate storage,
  lifecycle, permissions, and product semantics.
- Canonical identities are not a user-enumerable deployment-wide knowledge graph.
- No graph visualization UI ships in v1.
- No automatic ontology generation or activation ships in v1.
- No external graph database, retrieval cache, online extractor, LLM traversal,
  or model/network call is added to the inference path.
- Graph expansion does not replace baseline retrieval, reranking, real-chunk
  evidence budgets, public payload shapes, or citation validation.
