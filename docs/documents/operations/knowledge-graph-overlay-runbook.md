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

   The evaluation is synthetic and must use a fresh database, broker, queue,
   Compose project, and cache volumes. It must never reuse, activate an ontology
   on, stop, or otherwise mutate a shared deployment. The reviewed environment
   file must pin the extractor, strict local embedding, and strict local
   reranker model/revisions and
   set `APP_EMBED_BASE_URL=http://vllm_embed:8000/v1`,
   `APP_EMBED_API_KEY=EMPTY`, `APP_EMBED_DIMS=1024`, and
   `APP_EMBED_ALLOW_DIMENSIONS_OVERRIDE=0`. It must also set
   `APP_RERANK_PROVIDER=local`,
   `APP_RERANK_BASE_URL=http://vllm_rerank:8000/v1`,
   `APP_RERANK_API_KEY=EMPTY`, and an immutable lowercase 40-hex
   `APP_RERANK_MODEL_REVISION`. The explicit
   `APP_RERANK_TOKENIZER_REVISION` and `APP_RERANK_CODE_REVISION` must equal
   that commit. Likewise, `APP_EMBED_MODEL_REVISION`,
   `APP_EMBED_TOKENIZER_REVISION`, and `APP_EMBED_CODE_REVISION` must be the
   same lowercase 40-hex commit. `APP_RERANK_MODEL`,
   `APP_RERANK_VLLM_MODEL`, and `APP_RERANK_TOKENIZER` must be the same exact
   bounded Hugging Face repository ID. Both sidecars must set their strict
   protected-argument fence to `1`, tokenizer/code revisions explicitly,
   `VLLM_TRUST_REMOTE_CODE=1`, tensor parallelism `1`, and the reviewed
   runner/dtype/resource envelope. The resolved reranker pooling task is
   `classify`, inferred from the exact `Qwen3VLForSequenceClassification`
   override; both strict sidecar argument vectors omit the removed `--task`.
   `VLLM_EXTRA_ARGS` contains only the reviewed non-protected canonical
   payload; `LMCACHE_ENABLED=0`. Before deploying the strict shipping
   embed/rerank services, migrate any legacy operator environment by removing
   `--model`, `--served-model-name`, `--tokenizer`, revision, runner, dtype,
   trust, task, tensor-parallel, GPU-utilization, max-length, API-key, and
   download-directory options from
   `MEM0_EMBED_VLLM_EXTRA_ARGS` and `APP_RERANK_VLLM_EXTRA_ARGS`; set their
   typed variables instead. Set `VLLM_STRICT_PROTECTED_ARGS=1`,
   `VLLM_API_KEY=EMPTY`, `VLLM_DOWNLOAD_DIR=/root/.cache/huggingface/hub`,
   and `VLLM_PYTHON_BIN=python3` on both strict sidecars. The strict wrapper
   rejects duplicates and any noncanonical remaining payload.
   Main, OCR, and transcription profiles remain compatibility-mode unless
   separately migrated and attested. It must route the shipping rerank cache only to the
   isolated broker with `DJANGO_CACHE_REDIS_URL=redis://redis:6379/1` and
   `RAG_CACHE_ENABLED=1`. The checked-in
   `deploy/compose/knowledge-graph-eval.yml` override replaces every inherited
   development service env file with that reviewed absolute path, explicitly
   maps the fresh PostgreSQL database/user/password, and replaces host caches
   with project-scoped named volumes. Its graph/extractor cache is mounted at
   neutral `/opt/kg-eval-hf-cache` with both `HF_HOME` and
   `KG_GLINER2_CACHE_DIR` set to that path. The seeder creates only clearly named
   synthetic rows and a manifest; it creates no graph request, artifact, run,
   or graph rows.

   Run this Bash procedure from the repository root. It canonicalizes the four
   exact collection/request pairs emitted by the manifest, verifies every
   request's terminal JSON, and checksum-binds cleanup. `vllm_embed` and
   `vllm_rerank` are started together with `--no-deps`; both must be healthy at
   once. A normal profile start is forbidden because it would also start
   unrelated vLLM services.

   ```bash
   set -euo pipefail
   : "${TASK21_ENV_FILE:?set TASK21_ENV_FILE to the absolute reviewed eval env file}"
   case "$TASK21_ENV_FILE" in /*) ;; *) echo "TASK21_ENV_FILE must be absolute" >&2; exit 64 ;; esac
   TASK21_ENV_FILE="$(realpath -- "$TASK21_ENV_FILE")"
   test -f "$TASK21_ENV_FILE"
   install -d -m 700 artifacts
   KG_EVAL_RUN_ID="$(python -c 'import uuid; print(uuid.uuid4().hex)')"
   KG_EVAL_PROJECT="aquillm-kg-eval-$KG_EVAL_RUN_ID"
   KG_EVAL_QUEUE="knowledge-graph-eval-$KG_EVAL_RUN_ID"
   KG_EVAL_WORKER_NAME="$KG_EVAL_PROJECT-worker"
   KG_EVAL_MANIFEST=/app/artifacts/kg-eval-fixture-manifest.json
   KG_EVAL_REPORT=/app/artifacts/kg-eval-comparison.json
   KG_EVAL_ONTOLOGY_CHECKSUM=eb8d0c6b512216db2592f16898cd59ab76a2c95e9151c5fabfcc3f1be87a9059
   KG_EVAL_WRAPPER_SHA256="$(sha256sum deploy/scripts/vllm_start.sh | awk '{print $1}')"
   KG_EVAL_PARSER_SHA256="$(sha256sum deploy/scripts/parse_vllm_extra_args.py | awk '{print $1}')"
   KG_EVAL_TEMPLATE_SHA256="$(sha256sum deploy/docker/vllm/chat_templates/qwen3_vl_reranker.jinja | awk '{print $1}')"
   KG_EVAL_DOCKER_ENV=(env -i "PATH=$PATH")
   for variable in HOME DOCKER_HOST DOCKER_CONTEXT DOCKER_CONFIG \
     DOCKER_TLS_VERIFY DOCKER_CERT_PATH XDG_RUNTIME_DIR; do
     if test -n "${!variable:-}"; then
       KG_EVAL_DOCKER_ENV+=("$variable=${!variable}")
     fi
   done

   kg_eval_compose() {
     "${KG_EVAL_DOCKER_ENV[@]}" \
       "TASK21_ENV_FILE=$TASK21_ENV_FILE" \
       "KG_EXTRACTION_QUEUE=$KG_EVAL_QUEUE" \
       DJANGO_DEBUG=1 KG_EVAL_BYPASS_ALLOWED=1 \
       KG_BUILD_ENABLED=0 KG_OVERLAY_ENABLED=0 \
       DJANGO_CACHE_REDIS_URL=redis://redis:6379/1 \
       RAG_CACHE_ENABLED=1 COHERE_KEY= \
       docker compose --env-file "$TASK21_ENV_FILE" -p "$KG_EVAL_PROJECT" \
         -f deploy/compose/development.yml \
         -f deploy/compose/knowledge-graph-eval.yml \
         --profile knowledge-graph --profile vllm "$@"
   }

   kg_eval_python() {
     kg_eval_compose run --rm --no-deps \
       --user "$(id -u):$(id -g)" -e COHERE_KEY= \
       -e PYTHONDONTWRITEBYTECODE=1 \
       --entrypoint /bin/sh worker_knowledge_graph \
       -c 'umask 077; cd /app/aquillm; exec /opt/venv/bin/python "$@"' sh "$@"
   }

   kg_eval_no_cache_python() {
     kg_eval_compose run --rm --no-deps \
       --user "$(id -u):$(id -g)" -e COHERE_KEY= \
       -e PYTHONDONTWRITEBYTECODE=1 -e RAG_CACHE_ENABLED=0 \
       --entrypoint /bin/sh worker_knowledge_graph \
       -c 'umask 077; cd /app/aquillm; exec /opt/venv/bin/python "$@"' sh "$@"
   }

   kg_eval_root_python() {
     kg_eval_compose run --rm --no-deps --user 0:0 \
       -e COHERE_KEY= -e KG_GLINER2_LOCAL_FILES_ONLY=0 \
       -e PYTHONDONTWRITEBYTECODE=1 \
       --entrypoint /bin/sh worker_knowledge_graph \
       -c 'umask 077; cd /app/aquillm; exec /opt/venv/bin/python "$@"' sh "$@"
   }

   stop_eval_worker() {
     case "$KG_EVAL_PROJECT" in aquillm-kg-eval-[0-9a-f][0-9a-f]*) ;; *) return 97 ;; esac
     test "$KG_EVAL_WORKER_NAME" = "$KG_EVAL_PROJECT-worker"
     local worker_container
     worker_container="$(docker ps -aq --filter "name=^/${KG_EVAL_WORKER_NAME}$")"
     if test -n "$worker_container"; then
       case "$worker_container" in *[!0-9a-f]*) return 99 ;; esac
       test "$(docker inspect --format '{{ index .Config.Labels "com.docker.compose.project" }}' "$worker_container")" = "$KG_EVAL_PROJECT"
       test "$(docker inspect --format '{{ index .Config.Labels "com.docker.compose.service" }}' "$worker_container")" = worker_knowledge_graph
       test "$(docker inspect --format '{{ index .Config.Labels "com.docker.compose.oneoff" }}' "$worker_container")" = True
       docker rm -fv "$worker_container" >/dev/null
     fi
   }

   purge_eval_queue() {
     kg_eval_compose run --rm --no-deps -e COHERE_KEY= \
       --entrypoint /opt/venv/bin/celery worker_knowledge_graph \
       -A aquillm purge -Q "$KG_EVAL_QUEUE" -f >/dev/null 2>&1 || true
   }

   cleanup_kg_eval() {
     trap - EXIT INT TERM
     local cleanup_status=0
     case "$KG_EVAL_PROJECT" in aquillm-kg-eval-[0-9a-f][0-9a-f]*) ;; *) return 97 ;; esac
     stop_eval_worker
     purge_eval_queue
     if test -n "${KG_EVAL_MANIFEST_CHECKSUM:-}"; then
       kg_eval_python manage.py seed_knowledge_graph_eval_fixture \
         --cleanup --fixture-manifest "$KG_EVAL_MANIFEST" \
         --expected-manifest-checksum "$KG_EVAL_MANIFEST_CHECKSUM" || cleanup_status=$?
     fi
     while IFS= read -r volume; do
       test -z "$volume" || case "$volume" in "$KG_EVAL_PROJECT"_*) ;; *) return 98 ;; esac
     done < <(docker volume ls -q --filter "label=com.docker.compose.project=$KG_EVAL_PROJECT")
     kg_eval_compose down --volumes --remove-orphans || cleanup_status=$?
     test -z "$(docker ps -aq --filter "label=com.docker.compose.project=$KG_EVAL_PROJECT")" || cleanup_status=$?
     test -z "$(docker volume ls -q --filter "label=com.docker.compose.project=$KG_EVAL_PROJECT")" || cleanup_status=$?
     return "$cleanup_status"
   }
   trap 'status=$?; cleanup_kg_eval; exit "$status"' EXIT
   trap 'cleanup_kg_eval; exit 130' INT
   trap 'cleanup_kg_eval; exit 143' TERM

   kg_eval_compose config --quiet
   kg_eval_compose build worker_knowledge_graph vllm_embed vllm_rerank
   kg_eval_compose up -d --wait --wait-timeout 300 db redis
   kg_eval_python manage.py migrate --noinput

   kg_eval_python manage.py activate_knowledge_graph_ontology \
     --path research-v1.yaml \
     --expected-checksum "$KG_EVAL_ONTOLOGY_CHECKSUM" --dry-run
   kg_eval_python manage.py activate_knowledge_graph_ontology \
     --path research-v1.yaml \
     --expected-checksum "$KG_EVAL_ONTOLOGY_CHECKSUM" --yes

   kg_eval_compose up -d --wait --wait-timeout 3600 --no-deps vllm_embed vllm_rerank
   KG_EVAL_EMBED_CONTAINER="$(kg_eval_compose ps -q vllm_embed)"
   KG_EVAL_RERANK_CONTAINER="$(kg_eval_compose ps -q vllm_rerank)"
   test -n "$KG_EVAL_EMBED_CONTAINER"
   test -n "$KG_EVAL_RERANK_CONTAINER"
   test "$(docker inspect --format '{{.State.Health.Status}}' "$KG_EVAL_EMBED_CONTAINER")" = healthy
   test "$(docker inspect --format '{{.State.Health.Status}}' "$KG_EVAL_RERANK_CONTAINER")" = healthy
   printf 'vllm_simultaneous_gpu_gate=ok\n'

   for service_container in \
     "vllm_embed:$KG_EVAL_EMBED_CONTAINER" \
     "vllm_rerank:$KG_EVAL_RERANK_CONTAINER"; do
     IFS=: read -r service container extra <<<"$service_container"
     test -z "${extra:-}"
     docker exec \
       -e "EXPECTED_VLLM_SERVICE=$service" \
       -e "EXPECTED_WRAPPER_SHA256=$KG_EVAL_WRAPPER_SHA256" \
       -e "EXPECTED_PARSER_SHA256=$KG_EVAL_PARSER_SHA256" \
       -e "EXPECTED_TEMPLATE_SHA256=$KG_EVAL_TEMPLATE_SHA256" \
       "$container" python3 -c '
   import hashlib
   import json
   import os
   import re
   import shlex
   from pathlib import Path
   argv = [item.decode("utf-8") for item in Path("/proc/1/cmdline").read_bytes().split(b"\0") if item]
   def flag(name):
       assert argv.count(name) == 1
       index = argv.index(name)
       return argv[index + 1]
   def file_sha256(path):
       return hashlib.sha256(Path(path).read_bytes()).hexdigest()
   assert file_sha256("/vllm_start.sh") == os.environ["EXPECTED_WRAPPER_SHA256"]
   assert file_sha256("/parse_vllm_extra_args.py") == os.environ["EXPECTED_PARSER_SHA256"]
   model = os.environ["VLLM_MODEL"]
   revision = os.environ["VLLM_REVISION"]
   assert re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*", model)
   assert re.fullmatch(r"[0-9a-f]{40}", revision)
   assert model == os.environ["VLLM_SERVED_MODEL_NAME"] == os.environ["VLLM_TOKENIZER"]
   assert revision == os.environ["VLLM_TOKENIZER_REVISION"]
   assert revision == os.environ["VLLM_CODE_REVISION"]
   assert os.environ["VLLM_STRICT_PROTECTED_ARGS"] == "1"
   assert os.environ["VLLM_TRUST_REMOTE_CODE"] == "1"
   assert os.environ["VLLM_API_KEY"] == "EMPTY"
   assert os.environ["VLLM_DOWNLOAD_DIR"] == "/root/.cache/huggingface/hub"
   assert os.environ["VLLM_PYTHON_BIN"] == "python3"
   assert os.environ["LMCACHE_ENABLED"] == "0"
   assert Path(argv[0]).name == "python3"
   assert flag("--model") == os.environ["VLLM_MODEL"]
   assert flag("--served-model-name") == os.environ["VLLM_SERVED_MODEL_NAME"]
   assert flag("--tokenizer") == os.environ["VLLM_TOKENIZER"]
   assert flag("--revision") == os.environ["VLLM_REVISION"]
   assert flag("--tokenizer-revision") == os.environ["VLLM_TOKENIZER_REVISION"]
   assert flag("--code-revision") == os.environ["VLLM_CODE_REVISION"]
   assert flag("--runner") == "pooling"
   assert flag("--dtype") == "float16"
   assert flag("--tensor-parallel-size") == "1"
   assert flag("--api-key") == "EMPTY"
   assert flag("--download-dir") == "/root/.cache/huggingface/hub"
   assert argv.count("--trust-remote-code") == 1
   service = os.environ["EXPECTED_VLLM_SERVICE"]
   if service == "vllm_embed":
       assert model == os.environ["APP_EMBED_MODEL"]
       assert revision == os.environ["APP_EMBED_MODEL_REVISION"]
       assert revision == os.environ["APP_EMBED_TOKENIZER_REVISION"]
       assert revision == os.environ["APP_EMBED_CODE_REVISION"]
       assert flag("--gpu-memory-utilization") == "0.20"
       assert flag("--max-model-len") == "2048"
       assert "--task" not in argv
       assert flag("--quantization") == "bitsandbytes"
       assert flag("--load-format") == "bitsandbytes"
       loader = json.loads(flag("--model-loader-extra-config"))
       assert loader == {
           "load_in_4bit": True,
           "bnb_4bit_compute_dtype": "float16",
           "bnb_4bit_quant_type": "nf4",
           "bnb_4bit_use_double_quant": True,
       }
       hf_overrides = json.loads(flag("--hf-overrides"))
       assert hf_overrides == {"matryoshka_dimensions": [1024]}
       expected_extra_args = [
           "--quantization", "bitsandbytes",
           "--load-format", "bitsandbytes",
           "--model-loader-extra-config",
           "{\"load_in_4bit\":true,\"bnb_4bit_compute_dtype\":\"float16\",\"bnb_4bit_quant_type\":\"nf4\",\"bnb_4bit_use_double_quant\":true}",
           "--hf-overrides", "{\"matryoshka_dimensions\":[1024]}",
       ]
   else:
       assert service == "vllm_rerank"
       assert model == os.environ["APP_RERANK_VLLM_MODEL"]
       assert model == os.environ["APP_RERANK_MODEL"]
       assert model == os.environ["APP_RERANK_TOKENIZER"]
       assert revision == os.environ["APP_RERANK_MODEL_REVISION"]
       assert revision == os.environ["APP_RERANK_TOKENIZER_REVISION"]
       assert revision == os.environ["APP_RERANK_CODE_REVISION"]
       assert "--task" not in argv
       assert flag("--gpu-memory-utilization") == "0.30"
       assert flag("--max-model-len") == "1024"
       assert flag("--chat-template") == "/templates/qwen3_vl_reranker.jinja"
       overrides = json.loads(flag("--hf-overrides"))
       assert overrides == {
           "architectures": ["Qwen3VLForSequenceClassification"],
           "classifier_from_token": ["no", "yes"],
           "is_original_qwen3_reranker": True,
       }
       assert file_sha256("/templates/qwen3_vl_reranker.jinja") == os.environ["EXPECTED_TEMPLATE_SHA256"]
       expected_extra_args = [
           "--chat-template", "/templates/qwen3_vl_reranker.jinja",
           "--hf-overrides",
           "{\"architectures\":[\"Qwen3VLForSequenceClassification\"],\"classifier_from_token\":[\"no\",\"yes\"],\"is_original_qwen3_reranker\":true}",
       ]
   assert shlex.split(os.environ["VLLM_EXTRA_ARGS"]) == expected_extra_args
   print(f"{service}_provenance=ok")'
   done

   kg_eval_no_cache_python -c '
   from aquillm.utils import get_strict_index_embeddings, strict_index_embedding_signature
   from apps.knowledge_graph.evals.fixture_manifest import canonical_embedding_sha256
   signature = strict_index_embedding_signature()
   rows, actual = get_strict_index_embeddings(
       ["Aquillm synthetic KG evaluation probe"],
       expected_model_signature=signature,
   )
   assert actual == signature and len(rows) == 1 and rows[0][0] == 0
   vector_sha = canonical_embedding_sha256(tuple(rows[0][1]))
   print(f"strict_embedding_signature={signature} vector_sha256={vector_sha}")'

   kg_eval_python manage.py seed_knowledge_graph_eval_fixture \
     --fixture-manifest "$KG_EVAL_MANIFEST"
   KG_EVAL_MANIFEST_CHECKSUM="$(kg_eval_python -c '
   import sys
   from pathlib import Path
   from apps.knowledge_graph.evals.fixture_manifest import fixture_manifest_checksum, load_fixture_manifest
   print(fixture_manifest_checksum(load_fixture_manifest(Path(sys.argv[1]))))
   ' "$KG_EVAL_MANIFEST")"
   test "${#KG_EVAL_MANIFEST_CHECKSUM}" -eq 64

   mapfile -t KG_EVAL_SCOPE < <(kg_eval_python -c '
   import sys
   from pathlib import Path
   from apps.knowledge_graph.evals.fixture_manifest import load_fixture_manifest
   manifest = load_fixture_manifest(Path(sys.argv[1]))
   for binding in manifest["authorized_scope"]:
       print(binding["collection_id"], binding["rebuild_request_id"])
   ' "$KG_EVAL_MANIFEST")
   test "${#KG_EVAL_SCOPE[@]}" -eq 4

   kg_eval_no_cache_python -c '
   from apps.documents.models import TextChunk
   from apps.documents.services.chunk_rerank import (
       _STRICT_EVALUATION_RERANK,
       _strict_local_rerank_chunks,
   )
   rows = tuple(TextChunk.objects.order_by("pk")[:2])
   assert len(rows) == 2
   first = _strict_local_rerank_chunks(
       TextChunk, "Aquillm synthetic relationship probe", rows, 2,
       _capability=_STRICT_EVALUATION_RERANK,
   )
   second = _strict_local_rerank_chunks(
       TextChunk, "Aquillm synthetic relationship probe", rows, 2,
       _capability=_STRICT_EVALUATION_RERANK,
   )
   assert tuple(row.pk for row in first) == tuple(row.pk for row in second)
   print("strict_local_reranker=ok")'

   kg_eval_root_python manage.py check_knowledge_graph_extractor
   kg_eval_compose run -d --name "$KG_EVAL_WORKER_NAME" --no-deps \
     -e PYTHONDONTWRITEBYTECODE=1 worker_knowledge_graph

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
   docker exec -e "EXPECTED_QUEUE=$KG_EVAL_QUEUE" "$KG_EVAL_WORKER_NAME" \
     /bin/sh -c 'test "$KG_EXTRACTION_QUEUE" = "$EXPECTED_QUEUE"'

   KG_EVAL_COMPARE_ARGS=()
   for binding in "${KG_EVAL_SCOPE[@]}"; do
     read -r collection_id request_id extra <<<"$binding"
     test -z "${extra:-}"
     KG_EVAL_COMPARE_ARGS+=(--collection "$collection_id" --rebuild-request "$request_id")
     kg_eval_python manage.py rebuild_knowledge_graph \
       --collection "$collection_id" --request-id "$request_id" --eval-only
     inspection="$(kg_eval_python manage.py inspect_knowledge_graph \
       --request-id "$request_id" --wait --timeout-seconds 1800)"
     kg_eval_python -c '
   import json, sys
   report = json.loads(sys.argv[1])
   assert report["request_id"] == sys.argv[2]
   assert report["effective_request_id"] == sys.argv[2]
   assert report["status"] == "succeeded"
   assert report["request_error_code"] == ""
   assert report["failure_count"] == 0 and report["truncated"] is False
   assert report["artifacts"] and report["builds"]
   assert all(row["evaluation_only"] is True for row in report["artifacts"])
   assert all(row["rebuild_request_id"] == sys.argv[2] for row in report["artifacts"])
   assert all(row["evaluation_only"] is True for row in report["builds"])
   assert all(row["rebuild_request_id"] == sys.argv[2] for row in report["builds"])
   assert all(row["status"] == "succeeded" for row in report["builds"])
   ' "$inspection" "$request_id"
   done

   stop_eval_worker
   purge_eval_queue
   kg_eval_compose run --rm --no-deps \
     --user 0:0 \
     -e "KG_EVAL_HOST_UID=$(id -u)" -e "KG_EVAL_HOST_GID=$(id -g)" \
     -e PYTHONDONTWRITEBYTECODE=1 \
     --entrypoint /bin/sh worker_knowledge_graph -c '
   set -eu
   test "$HF_HOME" = /opt/kg-eval-hf-cache
   test "$KG_GLINER2_CACHE_DIR" = /opt/kg-eval-hf-cache
   test -d "$KG_GLINER2_CACHE_DIR"
   chown -R "$KG_EVAL_HOST_UID:$KG_EVAL_HOST_GID" "$KG_GLINER2_CACHE_DIR"'
   kg_eval_python -c '
   import os
   from pathlib import Path
   cache = Path(os.environ["KG_GLINER2_CACHE_DIR"])
   assert cache == Path("/opt/kg-eval-hf-cache")
   probe = cache / ".task20-host-uid-probe"
   probe.write_bytes(b"task20")
   assert probe.read_bytes() == b"task20"
   probe.unlink()
   print("kg_eval_cache_host_uid=ok")'
   kg_eval_python manage.py check_knowledge_graph_extractor

   kg_eval_no_cache_python -m apps.knowledge_graph.evals.run_kg_eval \
     --mode comparison --eval-only "${KG_EVAL_COMPARE_ARGS[@]}" \
     --fixture-manifest "$KG_EVAL_MANIFEST" --output "$KG_EVAL_REPORT"
   kg_eval_python -c '
   import sys
   from pathlib import Path
   from apps.knowledge_graph.evals.run_kg_eval import _load_comparison_report
   _load_comparison_report(Path(sys.argv[1]))
   print("comparison_report=validated")
   ' "$KG_EVAL_REPORT"

   printf 'Designated approver: review the JSON and human table, then enter the manifest checksum: '
   read -r KG_EVAL_APPROVED_MANIFEST_CHECKSUM
   test "$KG_EVAL_APPROVED_MANIFEST_CHECKSUM" = "$KG_EVAL_MANIFEST_CHECKSUM"
   kg_eval_python -m apps.knowledge_graph.evals.run_kg_eval \
     --write-measured-gates --comparison-report "$KG_EVAL_REPORT" \
     --runbook /app/docs/documents/operations/knowledge-graph-overlay-runbook.md
   kg_eval_python -m apps.knowledge_graph.evals.run_kg_eval \
     --verify-gates --comparison-report "$KG_EVAL_REPORT" \
     --runbook /app/docs/documents/operations/knowledge-graph-overlay-runbook.md

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
