# Optional Nemotron ASR

CPU contract test dependencies are declared separately in
`deploy/vllm_plugins/nemotron_asr/requirements-test.txt`. Install them alongside
the application test dependencies before running the plugin tests. These do not
install vLLM or enable a GPU service; pinned-runtime tests remain opt-in.

The standard GPU Compose files retain Whisper, its 0.08 GPU allocation, and the
production Genesis image and 131072-token context. Nemotron is an explicit opt-in
override, not enabled by updating this checkout. Existing deployments with
transcription disabled should leave it disabled until GPU capacity is verified.

## Local GPU ASR (Nemotron 3.5)

The isolated transcription image serves
`nvidia/nemotron-3.5-asr-streaming-0.6b`, served as
`nemotron-3.5-asr-streaming-0.6b`, pinned at revision
`f3d333391852ba876df169dcc9ba902d25b6ab0b`. It uses vLLM 0.21.0,
Transformers 5.13.0 and librosa 0.11.0. The plugin is installed as a wheel and its
build probe validates the runtime and processor contract without model weights.

Supported API: `POST /v1/audio/transcriptions` returning `.text`, batch/offline
only, at most 390 seconds per request. The initial release uses
`--max-num-seqs 1` and makes no concurrency promise. It uses dtype float32 (FP32),
`VLLM_USE_V2_MODEL_RUNNER=0`, and `--enforce-eager`. Translations, diarization,
word timestamps, verbose output, and realtime WebSockets are outside this release.
Blank `INGEST_TRANSCRIBE_LANGUAGE` requests automatic language detection;
language tags such as `en-US` are normalized. Adaptation languages require
`NEMOTRON_ASR_ALLOW_ADAPTATION_LANGUAGES=1`.

Review the [model card](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/blob/f3d333391852ba876df169dcc9ba902d25b6ab0b/README.md)
and [OpenMDW license](https://openmdw.ai/license/1-1/) before downloading weights.
The model license is distinct from the AquiLLM source license; this repository
does not redistribute them.

The override uses 0.20 GPU utilization for weights, features, activations, and
runtime overhead. This replaces Whisper's 0.08 allocation and adds 12 percentage
points to a shared GPU budget. Chat remains 0.45, embedding 0.12, reranking 0.15;
their sum with Nemotron is 0.92, before other processes and optional OCR.
This is not a validated shared-GPU capacity guarantee. Measure available memory
and startup/inference peaks before activation; change coordinated allocations
deliberately if needed. `NEMOTRON_ASR_GPU_MEMORY_UTILIZATION` configures the
optional allocation. No GPU services are launched by the CPU tests.

Existing `.env` files are not rewritten. The override pins the complete Nemotron
service configuration and application model, so stale `TRANSCRIBE_VLLM_*`
Whisper values cannot create a mixed configuration. Preserve your existing
transcription API key and explicitly reconcile language and provider settings.

From the repository root, review the rendered configuration first:

```sh
docker compose --env-file .env -f deploy/compose/production.yml -f deploy/compose/nemotron-asr.yml --profile vllm config
docker compose --env-file .env -f deploy/compose/production.yml -f deploy/compose/nemotron-asr.yml --profile vllm build vllm_transcribe
docker compose --env-file .env -f deploy/compose/production.yml -f deploy/compose/nemotron-asr.yml --profile vllm up -d --no-deps --wait --wait-timeout 900 --force-recreate vllm_transcribe
docker compose --env-file .env -f deploy/compose/production.yml -f deploy/compose/nemotron-asr.yml --profile vllm up -d --no-deps --force-recreate web worker
```

Use the same override with `base.yml` or `development.yml` when appropriate.
The development service does not publish host port 8005; probe from its container
or network. `no_gpu_dev` continues to use hosted `whisper-1` and should not use
this GPU override. For production, inspect `http://localhost:8000/health` inside
the service, then `http://127.0.0.1:8005/v1/models` and
`http://127.0.0.1:8005/v1/audio/transcriptions` using the fixture
`tests/fixtures/audio/librispeech_1272-128104-0000.flac`.

CPU contracts: run `deploy/vllm_plugins/nemotron_asr/tests` and
`tests/unit/test_nemotron_verification_harness.py`. GPU verification is explicit:
the PowerShell harness `scripts/verify_nemotron_asr.ps1` supports `-SelfTest`
without GPU work, and separate opted-in runtime/profile checks. Live tests under
`tests/asr` require `RUN_ASR_RUNTIME=1` and `ASR_BASE_URL`; skipping those is not
evidence of successful inference or shared-GPU capacity.

## Whisper rollback

Remove `-f deploy/compose/nemotron-asr.yml` from every subsequent Compose command,
restore the prior `.env` transcription configuration (including disabled
transcription, if applicable), and recreate `vllm_transcribe`, `web`, and `worker`
with the original Compose files. Build the standard `vllm_transcribe` image before
recreation if it is no longer cached. Rollback is not automatic and is not a
second resident model. Existing production model pins and GPU defaults remain
unchanged throughout this opt-in workflow.
