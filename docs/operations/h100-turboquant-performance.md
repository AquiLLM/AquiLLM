# H100 TurboQuant performance experiments

This workflow targets development host `149.165.150.254` (`aquillm-dev2`). Production is outside this rollout. Features are opt-in; checked-in deployment defaults remain unchanged.

## Runtime contract

Captured baseline image: `sha256:00441111dd81532d55310362a55b57b100fd48aa0cc589fb16b836ab042f885f`.

| Component | Captured value |
|---|---|
| GPU | H100 80GB HBM3, SM90, 132 SMs |
| vLLM | `0.23.1rc1.dev748+g2dfaae752` |
| PyTorch / Triton / FlashInfer | `2.11.0+cu130` / `3.6.0` / `0.6.13` |
| Genesis | `34e269301cc3df71ae4b0da00a0a159b16b4e5d8` |
| Model | `hampsonw/Qwen3.6-27B-AWQ-BF16-INT4-mtp-bf16` |
| Resolved model revision | `2d783431e303148fc6e16622fac5edac83a6b5c4` |
| Activation / KV | FP16 / `turboquant_k8v4` |
| Attention geometry | Q24, KV4, head dimension 256, physical pages 2128 tokens |
| Verifier | 15 committed splits, one raw speculative-tail slot, tile 32 |
| Serving | MTP depth 4, context 131072, scheduling budget 4096, one sequence, TP1 |
| Memory / prefix caching | GPU fraction 0.45 / disabled |

One-change comparisons preserve these settings and the other GPU services.

## Feature controls

| Control | Default | Experiment |
|---|---|---|
| `AQUILLM_H100_MTP_KERNEL` | `baseline` | `fused`; experimental, first serving sweep regressed short-context generation |
| `AQUILLM_H100_SPLIT_POLICY` | `baseline` | Adaptive remains inactive without a qualifying profile |
| `AQUILLM_H100_PREFILL` | `0` | Measured long-prefix path; serving qualification required |
| `AQUILLM_H100_GDN` | `baseline` | No enabled replacement on this pinned FP16 runtime |

The installed FlashInfer sampler was already enabled. The inspected GDN decode candidate rounds intermediate query/key and output values through BF16, so it does not preserve the captured FP16 behavior.

The prefill experiment combines attention over compressed historical KV with attention over the raw causal current chunk. Its complete kernel/caller benchmark improves approximately threefold at 32768–65536 cached tokens, but loses at 8192. Routing must remain within the measured GPU, model revision, precision, page geometry and length bounds. Unsupported requests retain the original PN401/P101/P38 routing. Raw-current semantics differ from the old P101 compressed-current branch and require full generation checks.

## Development image and switching

Use `/home/exouser/AquiLLM-h100`, tracking `origin/codex/h100-kernel-integration`. `deploy/docker/vllm/Dockerfile.h100-overlay` takes the captured local image ID as `BASE_IMAGE`. It adds the local package without upgrading serving dependencies. The hook preserves `sndr.plugin:register`, runs after Genesis, and rejects an unexpected plugin fingerprint.

Run these Python commands on the development host:

```sh
python3 scripts/h100_performance/dev_switch.py prepare --verify-current-baseline
python3 scripts/h100_performance/dev_switch.py switch --image IMAGE_ID --mtp baseline --prefill 1
python3 scripts/h100_performance/dev_switch.py rollback
```

Preparation requires the original image and verifies the running Compose hash using resolved JSON passed through stdin. State contains protected configuration digests, never credential values. Legacy state requires this explicit verification; it cannot reconstruct missing historical configuration.

Switching replaces only the main vLLM service. It checks image defaults and the protected environment, command, mounts and host settings before replacement, then verifies the resulting container. Feature-off comparisons use `--mtp baseline --prefill 0`. The sidecars remain running.

Health alone does not establish activation. Require an `AQUILLM_H100` installed log in the engine and a `route_exercised` record after an eligible request. Reject requested-but-inactive experiments. Record image ID, model revision, flags, GPU processes and activation logs with each comparison. Separate compilation/warmup from warmed latency.

## Rollback and configuration drift

`rollback` restores the captured original image and baseline feature flags with the same protected configuration. Verify health and a full-model smoke request afterward. This path was exercised during development testing.

Protected configuration drift blocks both switching and rollback. Digests cannot reconstruct changed credentials or mounts: restore the original protected configuration before retrying. A failed helper is not a completed rollback.

## Acceptance

`serve_bench.py` uses frozen token inputs; `quality_bench.py` supplies 32 exact checks; `long_quality_bench.py` adds long recall and tools. Completed streams require a terminal marker, valid finish reason, matching usage and output hashes. Multi-token MTP chunks are not token timestamps: aggregate decode time is elapsed time after first token divided by output tokens minus one.

`report.py` checks paired blocks, complete repeats, median/p95, MTP counter deltas, exact quality and separate activation/runtime/numerical/memory evidence. Missing evidence stays incomplete. Production qualification also requires an application replay and review of output changes. Development microbenchmarks do not authorize production deployment.

See `docs/audits/2026-10-10-h100-performance/` for measurements, decisions, limitations and the tested candidate image.
