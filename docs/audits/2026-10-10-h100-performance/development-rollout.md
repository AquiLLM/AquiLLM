# Authorized development rollout

The user explicitly requested merging the H100 work into `development` and deploying it on `149.165.150.254`, after reviewing the prefill gain, the mixed-workload throughput gain, and the decode/MTP regressions. This authorizes a development-only deployment of the bounded prefill path. Production remains outside scope.

The earlier [qualification report](results.md) remains unchanged: protected decode p95 and MTP acceptance failed their original gates, and application replay and complete greedy-output review remain missing. The development decision accepts those disclosed limitations; it does not convert the report to a pass or qualify production.

## Selected configuration

- Tested image: `sha256:f3f93409ed546b6438147b7b633aa4380573b8f547fc7758662be17700b6bdb7`.
- `AQUILLM_H100_PREFILL=1`; model, precision, geometry and length guards remain enforced.
- `AQUILLM_H100_MTP_KERNEL=baseline`, `AQUILLM_H100_SPLIT_POLICY=baseline`, `AQUILLM_H100_GDN=baseline`.
- Existing model, MTP depth, scheduler settings, cache geometry, dependencies and other services remain unchanged.
- Generic package defaults remain off. Development selection is explicit through the guarded `dev_switch.py` command and the running container configuration.

Use the merged checkout on the authorized host to recreate this selection:

```sh
python3 /home/exouser/AquiLLM/scripts/h100_performance/dev_switch.py switch \
  --image sha256:f3f93409ed546b6438147b7b633aa4380573b8f547fc7758662be17700b6bdb7 \
  --mtp baseline --prefill 1
```

The helper preserves the protected configuration and replaces only the main vLLM service. Docker restarts retain its image/environment; future Compose recreations must use the documented selection rather than assuming generic defaults enable the feature. Rollback remains `dev_switch.py rollback`, followed by health and a model-response check.

## Additional allocator experiment

The user also requested mimalloc testing for end-to-end latency. The first comparison isolates the main vLLM service with bounded prefill enabled in both arms. It measures client TTFT, generation time and full model-API response latency, with allocator activation verified in the serving processes. This does not establish full application/RAG latency. The broader mimalloc branch changes many services and is not included implicitly in the H100 merge.

Deployment and allocator measurements are pending the fresh checks recorded by this rollout.
