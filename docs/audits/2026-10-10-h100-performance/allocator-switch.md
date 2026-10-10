# Main vLLM allocator comparison

`scripts/h100_performance/allocator_switch.py` is restricted to the authorized
`aquillm-dev2` development host and `compose-vllm-1`. It reuses the existing
development switch's Docker, Compose, digest, and mount-order primitives without
changing `dev_switch.py` or its original `baseline.json`.

Preparation requires the exact authorized prefill image
`sha256:f3f93409ed546b6438147b7b633aa4380573b8f547fc7758662be17700b6bdb7`,
prefill enabled, and MTP/split/GDN controls at baseline. The running container
and resolved Compose configuration must still match the protected original
H100 state. Preparation also verifies the running resolved Compose hash.

The thin experiment image must retain the prefill image's filesystem layers,
platform, commands, users, health checks, and all other image configuration and
environment defaults. The only allowed image configuration differences are
`Entrypoint=["/usr/local/bin/aquillm-allocator", "/genesis_entrypoint.sh"]`,
`AQUILLM_ALLOCATOR=system`, and an optional `PYTHONMALLOC=default` default.
The image builder owns the copied wrapper/library artifact provenance; Docker
inspection proves base-layer retention and configuration, not the behavior of
the newly copied binaries. Preparation pins the exact experiment image ID and
both image configuration digests in separate `allocator.json` state. It refuses
an existing allocator state rather than recapturing a candidate.

Use the coordinator-built thin image ID below. These are commands for the
authorized host; this worker has not executed them against a server.

```sh
python3 /home/exouser/AquiLLM/scripts/h100_performance/allocator_switch.py prepare \
  --image sha256:<full-thin-image-id>
python3 /home/exouser/AquiLLM/scripts/h100_performance/allocator_switch.py switch --allocator system
python3 /home/exouser/AquiLLM/scripts/h100_performance/allocator_switch.py switch --allocator mimalloc
python3 /home/exouser/AquiLLM/scripts/h100_performance/allocator_switch.py rollback
```

Both allocator arms use that same pinned thin image, bounded prefill enabled,
and `PYTHONMALLOC=default`. Rollback restores the exact prefill image and the
captured allocator environment key presence or absence; it does not roll back
to the earlier prefill-off H100 image. The original H100 rollback helper and
state remain available independently.

External `LD_PRELOAD`/`LD_AUDIT` settings, unexpected image IDs, allocator modes,
environment keys/values, commands, mounts, runtime fields, flags, or original
baseline-state changes block replacement and rollback. Only the known wrapper
Entrypoint and two public allocator keys are normalized before comparison with
the captured digests. Protected environment values stay in process memory;
state stores digests, and `allocator-next.json` stores references to temporary
captured environment variables. Secrets are neither printed nor persisted.

The helper replaces only `vllm` with `up -d --no-deps --no-build vllm`, then
checks the exact selected image, allocator values, H100 flags, protected
configuration, and resolved Compose hash. A post-switch failure is reported and
requires runtime inspection; it is not hidden by an automatic fallback.

Helper success proves configuration selection, not allocator activation in the
worker processes, model correctness, performance, or chat/RAG behavior. The
coordinator must verify allocator activation and run the authorized model API
and subsequent application checks in both arms.

CPU verification uses the scripts test boundary to avoid Django setup:

```sh
rtk python -m pytest -q -p no:django \
  -c deploy/vllm_plugins/h100_kernels/pyproject.toml \
  --confcutdir=scripts/h100_performance scripts/h100_performance/tests
```

The combined scripts suite passed 106 tests, including 46 allocator tests
covering preparation, image/default drift, both arms, rejection before
replacement, post-switch verification failures, and rollback key absence.
