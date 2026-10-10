# Mimalloc experiment reproduction sources

These are exact byte copies of the coordinator's private source ledger at
`.superpowers/sdd/2026-10-10-h100-turboquant-performance`, captured while the
model API experiment was running and the application replay was being prepared.
They preserve the experiment's scripts, rather than provide a portable launcher
or attest that either stage completed. No runtime outputs, credentials, cookies,
customer data, or unrelated historical scripts are included. `manifest.json`
records each source hash and the source checkout commit
`d9a8c25ecd2ef8b8cfffc06a885f7eb2d100064b`; private ledger scripts were not
tracked at that commit. The image build pins PR #240's source commit
`fedc29373c634fd47cf1f4429d133edaa2bc7e91`.

The application stage uses harness commit
`c1624504139e04bfdd737077bb62dba521b719a9`; the manifest records its corrected web
helper hash separately from the historical capture. Read-only live preflight
found that Compose used the implicit `compose-web` image and that the Linux
builder omitted legacy Windows `ArgsEscaped` metadata. The corrected helper
verifies that implicit reference against the exact base and normalizes only that
Linux metadata difference while retaining full pinned image configuration digests.
See the [OCI definition](https://github.com/opencontainers/image-spec/blob/main/config.md)
of `ArgsEscaped`. The archived web builder also uses `.get('Entrypoint')` because
Docker omits that key when empty; its manifest hash reflects the executed fix.

## Original locations and dependencies

All ten archived files were staged under `/tmp/` on the authorized development
host `aquillm-dev2`, retaining these basenames. Keep `run_prefill_blocks.py`
beside both allocator runners: they import only its `run` and `wait_ready`
functions, without running its prefill experiment entry point.

| Source | Original use |
| --- | --- |
| `build_mimalloc_experiment.py`, `Dockerfile.mimalloc-experiment` | Build the model overlay from the pinned prefill image and clean `/home/exouser/AquiLLM-mimalloc-validation` checkout. |
| `run_allocator_api.py`, `probe_serving_allocator.py` | Three serial system/mimalloc pairs, fixed serving shapes, first-pair strict/long quality, process mappings and memory. Probe copied into `compose-vllm-1:/tmp/`. |
| `build_web_allocator.py`, `Dockerfile.web-mimalloc-experiment` | Build the web overlay from the original web image and the model overlay's allocator files, after the API stage. |
| `run_allocator_chat.py`, `probe_web_allocator.py` | Two serial pairs switching both web and model; one warmup plus five measured requests per chat/RAG kind. Web probe copied into `compose-web-1:/tmp/`. |
| `allocator_fixture_helper.py` | Verified disposable principal, real HTTP numbered VTT ingestion, owned DB/provenance checks and scoped cleanup; copied into `compose-web-1:/tmp/`. |
| `run_prefill_blocks.py` | Shared Docker command and model-health helpers. |

The runners need Docker, host Python 3, and the matching harness checkout at
`/home/exouser/AquiLLM-h100`. Its `scripts/h100_performance/` supplies
`allocator_switch.py`, `web_allocator_switch.py`, `serve_bench.py`,
`quality_bench.py`, `long_quality_bench.py`, and `chat_replay.py`; their captured
hashes are also in the manifest. Existing switch baselines must be prepared and
verified separately. Model benchmark scripts are copied into
`compose-vllm-1:/tmp/h100_performance/`.

The dependency hashes cover captured checkout bytes, including line endings,
rather than Git's normalized blobs. Five dependencies used CRLF;
`long_quality_bench.py` used LF. A checkout with another line-ending policy can
therefore have different byte hashes with identical source text. The ten archived
reproduction files retain their original bytes and must match their hashes exactly.

Web execution uses `/opt/venv/bin/python`, installed Django/allauth/httpx/websockets,
the application mounted at `/app`, and its existing database, embedding, memory,
graph and worker services. The replay client is copied to `/tmp/chat_replay.py`;
helper `--repo-root /app --replay-script /tmp/chat_replay.py` avoids modifying
that bind mount. Fixture/state JSON use `/tmp/h100-allocator-fixture*.json` and
are mirrored to the host before web replacement. A generated password stays in
the parent runner's memory and is passed to children as stdin JSON.

## Probe correction and evidence limits

EngineCore's `setproctitle` can overwrite the original environment region exposed
by `/proc/<pid>/environ` even while Python's `os.environ` retains its values.
The serving probe therefore permits missing allocator keys for EngineCore,
requires exact keys for the API process, and verifies allocator library mappings
for both. It labels EngineCore environment evidence unreliable and requires an
API and an engine record. It never emits the complete environment or command.

The API runner writes its completion marker before its `finally` rollback;
that marker alone does not prove restoration. Require the successful rollback,
health checks and independent retained restoration evidence. This archive
preserves that behavior. The current chat runner gates its completion marker
on successful measurements, scoped final cleanup and both healthy restorations.

The helper proves configured route/model and retained message/answer state;
it does not attest actual request dispatch or allocator mappings. Separate
process probes supply mapping evidence. Its empty profile-fact and namespace
checks do not establish terminal queued profile-promotion/KG/Celery tasks.
Mem0 deletion history is deliberately retained. Aggregate event capture covers
the web log time window and is not per-conversation route evidence. Consult the
separate runtime evidence before drawing performance or correctness conclusions.
