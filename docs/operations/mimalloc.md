# mimalloc deployment

AquiLLM's web, Celery, knowledge-graph, test, and vLLM images include mimalloc
3.5.3. This replaces native malloc at service startup to target CPU allocation
cost and resident memory. It does not replace CUDA allocation, change the
model, or establish a measured latency improvement for this deployment.

The shared builder pins upstream commit
`d4881d338125e1cb7c47ba4cfb398d6f7c0c8d45` and checks the source archive's SHA-256.
It builds on Ubuntu 22.04 for compatibility with both the Ubuntu vLLM and
Debian Bookworm application images. Architecture-specific optimizations are
disabled; compilation tools stay in the build stage. The runtime includes
the upstream license and `/opt/mimalloc/share/build-info`.

## Controls

All shipping Compose configurations default to:

```dotenv
AQUILLM_ALLOCATOR=mimalloc
AQUILLM_PYTHONMALLOC=default
```

The second setting preserves CPython's normal small-object allocator. To
experiment with routing Python object allocations through mimalloc as well,
set `AQUILLM_PYTHONMALLOC=malloc`. Python 3.12 does not support
`PYTHONMALLOC=mimalloc`; do not use that value in these images.

Every service can override both settings with its uppercase Compose name:

```dotenv
WEB_ALLOCATOR=system
WORKER_ALLOCATOR=mimalloc
WORKER_KNOWLEDGE_GRAPH_PYTHONMALLOC=malloc
KNOWLEDGE_GRAPH_QUERY_EXTRACTOR_ALLOCATOR=system
VLLM_ALLOCATOR=system
VLLM_EMBED_ALLOCATOR=mimalloc
VLLM_RERANK_PYTHONMALLOC=malloc
VLLM_TRANSCRIBE_ALLOCATOR=system
```

The same convention applies to schema/projection/memory workers, query gateways,
schedulers, OCR, and the optional Nemotron ASR image. Explicit service settings
take precedence over global settings. Pass `--env-file .env` to Compose so it
can interpolate these settings, including for isolated services without an
`env_file`. Existing database, Redis, Qdrant, Memgraph, MinIO, nginx, and browser
allocators are outside this change.

The image entrypoint applies preloading before the original command, including
Compose command overrides and spawned workers. It executes the command directly
so normal signal and exit-code handling is retained. The library and launcher
live outside `/app`, where bind mounts cannot hide them. Image builds do not run
under mimalloc.

Startup checks actual malloc symbol resolution and fails if the library is
missing, incompatible, or does not override malloc. A competing mimalloc,
jemalloc, or tcmalloc in `LD_PRELOAD` also fails startup; remove the competing
entry or select `system`. Other preloads are preserved. `system` removes this
launcher's own preload and leaves externally configured allocators alone.

## Development deployment and rollback

Build the changed images before recreating the selected services. For example,
for a web canary:

```sh
docker compose --env-file .env -f deploy/compose/development.yml build web
docker compose --env-file .env -f deploy/compose/development.yml up -d --no-deps --force-recreate web
docker compose --env-file .env -f deploy/compose/development.yml logs --tail=30 web
docker compose --env-file .env -f deploy/compose/development.yml exec web sh -c 'grep libmimalloc /proc/1/maps'
```

Look for `allocator=mimalloc`, the selected Python allocator, and
`mimalloc_version` in startup logs. Repeat build/recreate for each service being
rolled out; enable its usual profile (`vllm` or `knowledge-graph`) when required.
Build both ordinary vLLM and Genesis images when using both deployment profiles.
Preserve the current model, Genesis revision, flags, caches, and concurrency.

To roll back just web, set these values in the Compose interpolation environment
and repeat the recreate command (no rebuild needed):

```dotenv
WEB_ALLOCATOR=system
WEB_PYTHONMALLOC=default
```

For a global rollback set `AQUILLM_ALLOCATOR=system` and
`AQUILLM_PYTHONMALLOC=default`, remove any service-specific overrides, and recreate
the affected containers. A plain restart does not apply environment changes.
Commands launched with `docker exec` do not inherit entrypoint-only preloads;
inspect the running application's maps/logs rather than a fresh exec interpreter.

## Validation and performance comparison

The GPU-free CI images exercise real malloc/realloc/free interception, Python
allocation, subprocess inheritance, prefork worker recycling, command arguments,
exit status, missing libraries, competing preloads, and rollback. The builder
also runs mimalloc's own tests. Run locally with:

```sh
docker build -f deploy/docker/mimalloc/Dockerfile.test -t aquillm-mimalloc-smoke:local .
MIMALLOC_TEST_IMAGE=aquillm-mimalloc-smoke:local PYTHONPATH=aquillm python -m pytest -c /dev/null --noconftest aquillm/tests/integration/test_mimalloc_runtime.py aquillm/tests/integration/test_mimalloc_compose.py -q
```

Compare `system/default`, `mimalloc/default`, and `mimalloc/malloc` on the same
image, prompts, model, warmed caches, request rate, and worker counts. Change one
service at a time and alternate baseline/candidate runs. Measure p50/p95 latency,
throughput, CPU, steady-state/peak RSS or PSS, and memory after repeated ingestion
and worker recycling. Include sustained concurrency and long-running workers;
an allocator can trade lower latency for higher retained memory.

Separate inference TTFT from user-submission-to-visible-answer latency. Existing
`ttft_ms` stops on the first content, reasoning, or tool-call signal. Use the
existing RAG stage timings to locate changes and measure visible-answer latency
with an application replay. Report cold startup separately. Container tests
validate integration; GPU inference and workload performance still need a
development-host canary. Roll back a service if errors or repeatable latency or
memory regressions appear.
