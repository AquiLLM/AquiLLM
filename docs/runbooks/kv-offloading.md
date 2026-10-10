# KV storage and serving benchmark runbook

The capacity planner and benchmark client are locally testable. Storage plans are
**experimental and not launchable as a model-serving configuration**. The launcher
rejects `local`, `mooncake`, and `paged` execution. Mixed attention/GDN/MTP allocation
registration, active-page lifetime protection, and an active pager remain absent.
Setting a budget or starting a storage process does not establish KV traffic.

The user deferred development H100 testing because it is busy. No H100 endpoint,
checkpoint, or model was used for these benchmark tests. Four and eight users at
262144 tokens remain later qualification targets, including output reservation.
Kvarn/Neutrino performance work is independent.

## Invariants and capability boundary

All model weights stay GPU resident. Keep AWQ weights, `turboquant_k8v4`, MTP depth
4, Genesis correctness guards, TP1/PP1, and the existing model/revision. Do not add
CPU weight offload, replace packed KV with a different dtype, or disable the hybrid
manager to make an experiment launch. The approved model is
`hampsonw/Qwen3.6-27B-AWQ-BF16-INT4-mtp-bf16` at
`2d783431e303148fc6e16622fac5edac83a6b5c4`.

Capacity JSON is a payload estimate, excluding hybrid state, MTP, allocator,
scratch, compilation, and staging overhead. Explicit `off`/`resident` still opts
into capacity planning. Remove conflicting legacy `--max-num-seqs`,
`--max-model-len`, prefix/mamba, and connector arguments before selecting a
profile. With no `KV_CACHE_*` opt-in the existing startup behavior is preserved.
Budget and N/T resizing requires a controlled restart; no live-resize API exists.

Source pins: Genesis `34e269301cc3df71ae4b0da00a0a159b16b4e5d8`, LMCache
`05a013b29da78cf2321b9b46ec5039dde2fb0bb0` (0.5.5), Mooncake
`719735896c86b56fabec6cf3e825fb2ea640597a` (0.3.13.post1). Base digest:
`sha256:6a93ae4316826f3dd8a92bee5442cbed50184a9cbd688d310f9e56ecad1eabeb`.

## Produce capacity profiles without launching

Run from the repository root. These PowerShell commands scope environment changes
to a child process and write UTF-8 JSON. They do not load weights or contact an
endpoint. Clear inherited conflicting CLI settings when planning.

```powershell
rtk proxy powershell -NoProfile -Command '$env:KV_CACHE_TARGET_ACTIVE_SEQUENCES="4"; $env:KV_CACHE_RETAINED_CONTEXTS="4"; $env:VLLM_MAX_MODEL_LEN="262144"; $env:KV_CACHE_STORAGE_MODE="off"; $env:KV_CACHE_EXECUTION_MODE="resident"; python deploy/scripts/kv_cache_config.py --plan | Set-Content -Encoding utf8 capacity-4.json'
rtk proxy powershell -NoProfile -Command '$env:KV_CACHE_TARGET_ACTIVE_SEQUENCES="8"; $env:KV_CACHE_RETAINED_CONTEXTS="8"; $env:VLLM_MAX_MODEL_LEN="262144"; $env:KV_CACHE_STORAGE_MODE="off"; $env:KV_CACHE_EXECUTION_MODE="resident"; python deploy/scripts/kv_cache_config.py --plan | Set-Content -Encoding utf8 capacity-8.json'
```

The same public API is `resolve_profile(env, extra_args) -> dict | None` in
`deploy/scripts/kv_cache_config.py`. Ordinary resolution only accepts launchable
off/resident modes; `--plan` can serialize unsupported modes for inspection.
The benchmark consumes the serialized profile, deriving N/T from it.

For experimental storage, first capture the actual mixed `KVCacheConfig` group
layout and measure resource headroom into `captured-layout.json` and
`measured-limits.json`. See `kv_storage_layout.py`/`kv_storage_plan.py` for schemas.
The test layout's 16/32 token spans and 32-token chunk are synthetic, not measured
geometry or deployment defaults. Explicit RAM/SSD caps must cover L1, embedded
RealClient segment/buffer, SSD staging, per-engine staging, overhead, and the
metadata master's at least 512 MiB. `adapter_threads` is a shared thread count.

```powershell
rtk proxy powershell -NoProfile -Command '$env:KV_CACHE_TARGET_ACTIVE_SEQUENCES="4"; $env:KV_CACHE_RETAINED_CONTEXTS="4"; $env:VLLM_MAX_MODEL_LEN="262144"; $env:KV_CACHE_STORAGE_MODE="mooncake"; $env:KV_CACHE_EXECUTION_MODE="resident"; $env:KV_CACHE_RAM_GIB="8"; $env:KV_CACHE_SSD_GIB="16"; python deploy/scripts/kv_storage_plan.py --layout captured-layout.json --limits measured-limits.json | Set-Content -Encoding utf8 kv-storage-plan.json'
```

The 8/16 GiB values are example explicit caps; planning rejects insufficient
measured headroom. Do not interpret a plan as approval to execute its emitted argv.
Local mode is RAM only; Mooncake uses the embedded RealClient's own SSD path and
caps, not the master's `root_fs_dir`. One server/RealClient per IPC namespace is
supported because its embedded offload port is fixed at 50052.

## Native image and experimental infrastructure diagnostics

These are local Docker Desktop commands for a separately scheduled operator
session. Do not run a second native build while one is already running. The
Genesis tag must resolve to the expected locally built base image ID; a mutable
tag is not compatibility evidence. Compilation uses two jobs and one nvcc thread.

```powershell
rtk proxy docker --context desktop-linux image inspect aquillm-kv-genesis-local:task2 --format '{{.Id}}'
rtk proxy docker --context desktop-linux build --progress plain -f deploy/docker/vllm/Dockerfile.kv-storage --build-arg GENESIS_IMAGE=aquillm-kv-genesis-local:task2 -t aquillm-kv-storage-local:task2 .
rtk proxy docker --context desktop-linux run --rm --network none --entrypoint python3 aquillm-kv-storage-local:task2 /opt/kv-storage/kv_storage_preflight.py --base-identity
rtk proxy docker --context desktop-linux run --rm --network none --gpus all --entrypoint python3 aquillm-kv-storage-local:task2 /opt/kv-storage/kv_storage_preflight.py --native
rtk proxy docker --context desktop-linux run --rm --network none --gpus all --entrypoint python3 aquillm-kv-storage-local:task2 -m pip check
```

`--base-identity` verifies the inherited serving tuple. `--native` additionally
checks pinned source revisions and real `lmcache.cuda_ops` and C++ Mooncake imports;
it explicitly reports GPU round trips as not validated. Driver-less native load
may fail because `libcuda.so.1` is unavailable. The build-only CUDA link stub must
never enter runtime `LD_LIBRARY_PATH` or ELF RPATH/RUNPATH.

Retain image IDs, `/opt/kv-storage/base-manifest.json`, `python-lock.txt`,
`system-lock.txt`, `sdk-sha256.txt`, `wheel-sha256.txt`, and
`mooncake-include-paths.txt` alongside operator records. Do not invent a successful
`native-manifest.json`: run and retain the actual `--native` output.

The standalone `deploy/compose/kv-storage.yml` has profiles `kv-local` and
`kv-mooncake`, no model service, no published ports, and an internal network.
Its plan file and unique SSD directory must already exist. Set
`KV_STORAGE_PLAN_PATH`, `KV_STORAGE_SSD_PATH`, `KV_STORAGE_IMAGE`,
`KV_STORAGE_MEMORY_LIMIT`, and `KV_STORAGE_SHM_SIZE` to reviewed absolute paths
and explicit limits before infrastructure use. The Compose file does not inherit
the application's `.env`. Dry rendering is read-only:

```powershell
rtk proxy docker --context desktop-linux compose -f deploy/compose/kv-storage.yml --profile kv-local config --quiet
rtk proxy docker --context desktop-linux compose -f deploy/compose/kv-storage.yml --profile kv-mooncake config --quiet
```

Only after native/resource diagnostics pass may an operator start the experimental
infrastructure in an isolated scheduled session:

```powershell
rtk proxy docker --context desktop-linux compose -f deploy/compose/kv-storage.yml --profile kv-mooncake up -d mooncake lmcache
rtk proxy docker --context desktop-linux compose -f deploy/compose/kv-storage.yml --profile kv-mooncake run --rm kv-storage-preflight
```

Preflight reconstructs validated commands and checks current host/cgroup RAM,
disk headroom, and required TCP endpoints. It then **always refuses serving
readiness** (exit 64). TCP success is not an LMCache protocol handshake. This is
an expected capability gate, not a reason to remove the check.

## Token fixtures and reproducible measurements

Supply a UTF-8 JSON token fixture with `model`, `revision`, and `prompts`, where
`prompts` contains exactly N distinct lists of integer token IDs. Use an already
available tokenizer matching the exact model/revision to create these lists;
the runner neither downloads nor instantiates a tokenizer. Do not count characters
or words, fabricate a tokenization claim, or send production documents.
For full-capacity reservation with 256 output tokens, each prompt has exactly
261888 token IDs. Early truncation or a short prompt cannot qualify full T.

Add `identity` to the capacity profile from inspected runtime records, with all
of these nonempty string fields:

| Field | Required operator record |
| --- | --- |
| model / revision | Exact underlying model and checkpoint revision above |
| runtime | Exact Python/torch/CUDA/vLLM/transformers/HF hub versions, image/source hashes, TP1/PP1 and material runtime settings |
| quantization / mtp | `awq` / `depth-4` |
| hardware | Exact GPU model/count, VRAM, driver, CPU/RAM/storage and material hardware settings |
| genesis | Exact commit and correctness/optimization guard settings |
| weights | `gpu-resident`, based on inspected launch argv and allocations |

Store these fields in `runtime-identity.json`, then enrich a copied profile:

```powershell
rtk proxy python -c 'import json; from pathlib import Path; p=json.loads(Path("capacity-4.json").read_text(encoding="utf-8-sig")); p["identity"]=json.loads(Path("runtime-identity.json").read_text(encoding="utf-8-sig")); Path("resident-4.json").write_text(json.dumps(p,indent=2),encoding="utf-8")'
```

This is **operator-supplied, unattested metadata**. It does not prove server
identity or weight placement. Without complete compatible metadata, measured
rates can still be displayed but target/baseline verdicts remain unavailable.
Use `--model` for underlying tokenizer identity and `--served-model` for the
explicit endpoint alias configured by `VLLM_SERVED_MODEL_NAME`.

First validate every prompt and output reservation with no network calls:

```powershell
rtk proxy python scripts/benchmark_kv_offloading.py --profile resident-4.json --prompts prompts-4.json --url http://127.0.0.1:8000/v1/completions --model hampsonw/Qwen3.6-27B-AWQ-BF16-INT4-mtp-bf16 --served-model qwen3.6:27b-mtp-awq --output-tokens 256 --dry-run --output dry-run-4.json
```

The later live commands require the user-selected endpoint and separately
approved available hardware/session. The loopback URL below is an explicit
example, not permission to contact a busy H100. The runner never launches a model.
If authentication is needed, add `--api-key-env VLLM_API_KEY` using an existing
environment secret; do not place a key in the URL, argv, profile, or fixture.
Redirects are rejected, and responses/prompts/credentials are not stored in reports.

```powershell
rtk proxy python scripts/benchmark_kv_offloading.py --profile resident-4.json --prompts prompts-4.json --url http://127.0.0.1:8000/v1/completions --model hampsonw/Qwen3.6-27B-AWQ-BF16-INT4-mtp-bf16 --served-model qwen3.6:27b-mtp-awq --output-tokens 256 --timeout 1800 --run-id resident-4-repeat-1 --output resident-4-run.json
rtk proxy python scripts/benchmark_kv_offloading.py --profile candidate-4.json --prompts prompts-4.json --url http://127.0.0.1:8000/v1/completions --model hampsonw/Qwen3.6-27B-AWQ-BF16-INT4-mtp-bf16 --served-model qwen3.6:27b-mtp-awq --output-tokens 256 --timeout 1800 --run-id candidate-4-repeat-1 --baseline resident-4-run.json --output candidate-4-run.json
```

The candidate command is for a future capability-qualified runtime. A storage
plan alone cannot produce that runtime today. Repeat with N=8 and the matching
eight-prompt fixture/profile. Record cold and warmed runs separately, repeated
measurements, failures and full latency distributions. Keep workload fingerprints,
output reservation, generation settings, runtime and hardware identical across
each baseline pair. For optional spill sweeps, restart between reviewed VRAM/RAM/
SSD budgets and retained-context counts; retain one result per budget/repeat. A
changed configured spill budget is not evidence of actual page transfers.

## Interpret results and attach historical traces

Accepted token counts come only from final `usage.completion_tokens`; drafts,
text lengths and SSE events are excluded. `usage.prompt_tokens`, when supplied,
must match the supplied token-ID length. Invalid, missing, conflicting or failed
accounting never produces an output rate. Full exercise is separately reported as
supplied prompt length plus accepted output, with reserved length also recorded.

The conservative rate is accepted completion tokens divided by full request
elapsed time, including prefill and first-token latency. It is not a measurement
of steady decode speed. MTP may commit several tokens in the first chunk, so
decode-only rate is explicitly unavailable. TTFT, each client-visible nonempty
chunk gap, maximum gap and terminal wait are reported per request. Chunk gaps
are not per-token stalls. Each user must reach the 55 token/s lower threshold;
55–75 is the desired band. A fast aggregate cannot hide a slow user.
Rate and context-coverage verdicts describe each measured request; decoder
concurrency and paging are separate evidence fields and may remain unverified.
The timeout watchdog bounds connected HTTP exchanges, including slowly arriving
headers. OS hostname resolution is not cancellable by this standard-library
client; use a locally resolved, reviewed endpoint for controlled timing runs.

Baseline checks are independent: each request's total latency and TTFT must be
at most 1.10 times its matching resident baseline; 1.05 is preferred. Per-user
maximum-gap ratios are also reported for pause review. Missing/mismatched N/T,
model/revision/runtime/KV dtype/AWQ/MTP/Genesis/weights/hardware, workload hash,
prompt fingerprint or output reservation makes the comparison unavailable.
Incomplete or serial baseline runs also cannot pass a concurrent comparison.
Offline imports must retain exact usage provenance, integer accepted counts equal
to the reservation, consistent elapsed-time rates and context lengths, and
complete timing records. Analysis recomputes targets; cached verdicts never count
as evidence, and malformed/incomplete records yield unavailable comparisons.
CLI exit 0 means the workflow ran; it is not a performance acceptance verdict.
Request/protocol failures exit 2, input failures exit 64, and argparse misuse exits 2.

Client HTTP overlap reports only simultaneous outstanding requests. Without
run-correlated decoder intervals, N active decoders remains **unverified**. A
global running gauge, configured `max-num-seqs`, or connector flag proves neither
decoder concurrency nor active paging. Supply optional trace JSON after the run:

```json
{
  "run_id": "candidate-4-repeat-1",
  "scheduler_intervals": [
    {"start_seconds": 2.1, "end_seconds": 2.2, "phase": "decode",
     "request_ids": ["candidate-4-repeat-1:0", "candidate-4-repeat-1:1",
                     "candidate-4-repeat-1:2", "candidate-4-repeat-1:3"]}
  ],
  "transfers": [
    {"request_id": "candidate-4-repeat-1:0", "at_seconds": 2.15,
     "purpose": "active_kv", "direction": "h2d", "bytes": 4096}
  ]
}
```

Times must be aligned to the recorded client run origin and fall within the
corresponding request intervals. Request IDs are sent in `X-Request-ID`; instrument
the server to propagate them to scheduler/transfer traces. Transfers require
positive bytes and `d2h`/`h2d`; `prefix_restore` never counts as active paging.
These are operator-supplied historical records, not attested by the client. A
matching valid record yields `supported-by-supplied-trace`; missing or irrelevant
records remain unverified. The schema is an analysis input, not a claim that the
current runtime emits these traces or implements active paging.

```powershell
rtk proxy python scripts/benchmark_kv_offloading.py --analyze candidate-4-run.json --metrics candidate-4-trace.json --baseline resident-4-run.json --output candidate-4-analyzed.json
```

Analysis is offline and sends no requests. Raw metrics/source strings are not
copied into the report. Preserve the original run and external trace provenance
with the analyzed result.

## Failures, recovery and rollback

Keep the original serving image ID, complete sanitized launch settings, cache
directory ownership and resident baseline before experiments. On timeout, usage
mismatch, OOM, driver/import failure, disk exhaustion, or failed numerical
continuation, stop the experiment and preserve diagnostic records. Do not silently
fall back to a different dtype, bypass readiness, share another server's SSD path,
or delete model/cache volumes to make recovery work.

For this separate storage infrastructure, shutdown is scoped and keeps volumes:

```powershell
rtk proxy docker --context desktop-linux compose -f deploy/compose/kv-storage.yml --profile kv-mooncake down
```

Restore the original serving image and original sanitized settings, unset all
experimental `KV_CACHE_*` opt-ins, set `LMCACHE_ENABLED=0`, and remove experimental
connector/offload arguments. If deliberately retaining a resident capacity
profile, use `KV_CACHE_STORAGE_MODE=off` and `KV_CACHE_EXECUTION_MODE=resident`
instead; this still applies its N/T/budget and is not the untouched default.
Perform a controlled serving restart through the existing deployment workflow
only in its approved session, then rerun resident correctness/readiness checks.

## Checks actually exercised

The benchmark uses only the Python standard library. Local loopback HTTP/SSE tests
cover fragmentation, interleaved and serial requests, MTP chunks/drafts, exact and
missing usage, prompt-count mismatch, redacted errors, redirect refusal, timeout,
token-reservation validation, per-user rate failures, workload/runtime mismatch,
unverified gauges and prefix restores, and offline analysis. Small fixture N=2,
T=16 is explicitly synthetic; it is not a 256K serving measurement.
The final selection passed 68 tests; the focused dry-run/live CLI and deadline
selection passed 3 tests.

```powershell
rtk proxy python -m pytest -c pyproject.toml aquillm/tests/integration/test_kv_offloading_benchmark.py aquillm/tests/integration/test_kv_offloading_benchmark_contracts.py -q --tb=short
rtk proxy python -m pytest -c pyproject.toml aquillm/tests/integration/test_kv_offloading_benchmark.py::test_cli_dry_run_and_loopback_smoke -q --tb=short
```

The standalone local commands use `pyproject.toml` and run without the repository's
Django-specific pytest configuration or its unrelated warning.
Task 2 configuration/preflight/Compose checks passed locally; the earlier native
build failed CUDA SDK linkage, followed by a successful bounded allocator-only
link/load and synthetic 256-byte local RTX 3090 copy. That was not an LMCache
packed transfer or hybrid registration. The latest SDK build succeeded; the
candidate image build is still in progress and native/kernel results remain
pending separate confirmation at this writing. No serving, H100,
SSD round trip, numerical equivalence, sustained 55–75/user target, full 4/8-user
256K context, or active paging success is claimed.
