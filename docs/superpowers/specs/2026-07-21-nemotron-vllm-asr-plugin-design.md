# Nemotron 3.5 ASR vLLM Plugin Design

**Date:** 2026-07-21

## Objective

Add batch-only NVIDIA Nemotron 3.5 ASR support to AquiLLM as an out-of-tree
vLLM plugin while preserving the existing OpenAI-compatible
`POST /v1/audio/transcriptions` contract used by media ingestion.

The first release must make `nvidia/nemotron-3.5-asr-streaming-0.6b` the local
transcription default, retain environment-only rollback to Whisper, and avoid
changing the Django ingestion contract. Realtime streaming is explicitly a
later milestone.

## Constraints and Decisions

- Target the repository's pinned `vllm/vllm-openai:v0.21.0` runtime.
- Package the integration inside AquiLLM rather than the external Genesis
  repository.
- Install the plugin only in the transcription image. Chat, OCR, embedding,
  and reranking services must not inherit its Transformers upgrade or model
  code.
- Use vLLM's documented `vllm.general_plugins` entry point and
  `ModelRegistry.register_model` hook.
- Preserve the existing OpenAI SDK call in
  `aquillm/aquillm/ingestion/media.py` and its plain `.text` result.
- Start in FP32, matching the published checkpoint. BF16 can be enabled only
  after transcript-parity testing.
- Run eagerly with one active sequence in the initial implementation. This
  avoids request-state corruption until an upstream-compatible request-scoped
  model-state hook is proven.
- Treat omitted language as automatic detection. An optional ingestion
  language setting may pass an explicit locale.
- Keep the no-GPU development profile on hosted `whisper-1`.
- Use operational rollback, not automatic dual-model retry. Automatic fallback
  would require another resident model and is outside this scope.

## Approaches Considered

### AquiLLM-owned out-of-tree model plugin — selected

Create a small Python package under `deploy/vllm_plugins/nemotron_asr/` and a
dedicated transcription Dockerfile. This uses vLLM's supported extension point,
keeps ASR versioning under AquiLLM control, and isolates the Transformers 5.13
requirement from other inference services.

### Extend the Genesis/TurboQuant patch repository

This would reuse an existing build-time plugin source, but it couples an ASR
model adapter to unrelated chat quantization patches and an external release
lifecycle. The current transcription service deliberately uses the plain vLLM
image, so this approach weakens service isolation.

### Overlay or fork vLLM source directly

Copying modified vLLM files into the image would provide maximum access to
internal hooks but is brittle across vLLM releases. A narrowly scoped
compatibility patch is acceptable only if vLLM 0.21 cannot run the single-
request plugin through its default model state.

## Architecture

### Plugin package

The package will expose a re-entrant entry point named
`aquillm_nemotron_asr` in the `vllm.general_plugins` group. Its registration
function lazily registers the checkpoint architecture:

```text
Nemotron3_5AsrForRNNT
  -> aquillm_vllm_nemotron_asr.model:Nemotron3_5AsrForRNNT
```

The package will contain focused modules:

- `__init__.py`: re-entrant model registration only.
- `processing.py`: vLLM multimodal processing metadata, dummy inputs, audio
  feature extraction, and language-prompt propagation.
- `model.py`: vLLM model wrapper, checkpoint loading, encoder invocation,
  greedy RNNT decoding, and forced-token replay.
- `state.py` if needed: isolated predecoded-token state helpers. The initial
  model may keep one request's state locally because `max-num-seqs=1` is a hard
  runtime constraint.

The package must not monkeypatch broad vLLM internals during import. If a
vLLM 0.21 compatibility hook is necessary, it will live in an explicit,
version-checked module and fail fast on unsupported vLLM versions.

### Transcription image

`deploy/docker/vllm/Dockerfile.transcribe` will derive from
`vllm/vllm-openai:v0.21.0`, install media libraries, install a pinned
Transformers 5.13 release, and install the local plugin package. Build-time
checks will verify:

1. the plugin entry point is discoverable;
2. Transformers exposes `Nemotron3_5AsrForRNNT` and its processor;
3. loading vLLM plugins registers the architecture.

The same image remains able to serve Whisper because registering an unused
model architecture is inert. Rollback therefore requires only environment
changes and recreation of `vllm_transcribe`.

### Compose and startup wiring

The `vllm_transcribe` service in base, development, and production Compose
files will use `Dockerfile.transcribe`. It will set
`VLLM_SERVICE_KIND=transcribe`, allowing `vllm_start.sh` to recover
`TRANSCRIBE_VLLM_EXTRA_ARGS` without matching a Whisper-specific model name.

Defaults will select:

```text
model:       nvidia/nemotron-3.5-asr-streaming-0.6b
served name: nemotron-3.5-asr-streaming-0.6b
tokenizer:   nvidia/nemotron-3.5-asr-streaming-0.6b
dtype:       auto (checkpoint resolves to FP32)
execution:   eager
sequences:   1
```

`INGEST_TRANSCRIBE_MODEL` must equal the served name. `.env.example` will
include a complete commented Whisper rollback configuration. Existing local
`.env` values remain operator-owned and will not be overwritten wholesale.

## Request and Decode Flow

1. AquiLLM sends a complete media file to
   `POST /v1/audio/transcriptions` through the OpenAI Python client.
2. vLLM decodes/downmixes the media waveform and invokes the plugin's
   multimodal processor.
3. The Nemotron processor resamples to 16 kHz and creates the checkpoint's
   128-bin log-mel features and attention mask.
4. The processor maps an explicit language code or locale to `prompt_ids`.
   Missing language maps to `auto` rather than the Transformers pipeline's
   English default.
5. The FastConformer encoder processes the complete utterance once.
6. The Transformers-compatible greedy RNNT algorithm emits the transcript
   sequence. Blank token `13087` advances the encoder frame; nonblank tokens
   remain on the frame, with the checkpoint's ten-symbol-per-frame cap.
7. The plugin stores that completed token sequence for the single active
   request and replays forced logits through vLLM's normal scheduler. This
   preserves vLLM's transcription response machinery without pretending the
   RNNT decoder is autoregressive.
8. Token decoding uses `skip_special_tokens=True` and does not CTC-collapse
   repeated emissions. The response remains `{ "text": "..." }`.

## API Behavior

- `language`: supported. Bare codes are normalized to a supported locale when
  necessary; `auto` selects prompt ID 101.
- omitted language: automatic detection.
- `temperature`: only greedy decoding is supported; zero is accepted.
- `prompt`: unsupported in the first release and must not silently alter
  output.
- translations: unsupported.
- response: normal JSON/text transcription is required. Word-level timestamps
  are outside the first release because the checkpoint exposes token durations,
  not native word segmentation.

AquiLLM's current client does not send optional parameters, so these limitations
do not change the product's existing ingestion path.

## Length, Resources, and Safety

The checkpoint contains approximately 638 million FP32 parameters in a 2.55 GB
safetensors file. Practical GPU memory use is higher due to features,
activations, and vLLM overhead.

The encoder has 5,000 post-subsampling positions. Based on 10 ms feature frames
and 8x subsampling, offline utterances approach a hard ceiling near 400 seconds.
The first release will document and enforce a conservative maximum near 390
seconds rather than silently truncating longer files.

The model weights are licensed under OpenMDW 1.1; the Transformers integration
code is Apache-2.0. AquiLLM will download rather than redistribute weights, and
documentation will identify the model license.

## Failure Handling and Rollback

- Unsupported plugin/runtime versions fail during image build or service
  startup, not on the first user upload.
- Unsupported language values return an actionable client error.
- Empty transcripts remain errors in AquiLLM's existing ingestion adapter.
- Sequential requests must clear forced-token state before processing the next
  utterance.
- The service health check gates dependent inference services as it does today.
- Operators can restore Whisper by changing the documented model, tokenizer,
  served name, model alias, and Whisper-specific extra arguments, then
  recreating `vllm_transcribe`.

## Verification Strategy

### Package tests

- Entry-point registration is re-entrant and lazily imports vLLM model code.
- The architecture is visible after plugin loading.
- Language mapping covers `auto`, bare codes, explicit locales, and invalid
  values.
- Greedy RNNT state handling covers blank emissions, repeated real tokens,
  maximum symbols per frame, and state reset between requests.
- Output cleanup removes blank and language-tag special tokens without
  collapsing valid repetition.
- Checkpoint weight names load without unexpected or missing tensors.

### Repository integration tests

- Only `vllm_transcribe` uses `Dockerfile.transcribe`.
- All three GPU Compose files agree on the Nemotron defaults and service kind.
- The no-GPU profile remains on hosted Whisper.
- The startup script selects transcription arguments by service kind.
- `INGEST_TRANSCRIBE_MODEL` matches the served alias.
- Optional ingestion language is passed only when configured, if that small
  adapter enhancement is included.

### Container and GPU verification

1. Build the transcription image successfully.
2. Start the service on the available RTX 3090 and wait for `/health`.
3. Verify `/v1/models` exposes the Nemotron served alias.
4. Transcribe a known short 16 kHz WAV through `/v1/audio/transcriptions`.
5. Compare plugin output with direct Transformers 5.13 output for the same
   audio and language prompt.
6. Submit two sequential utterances and verify no transcript state leaks.
7. Reconfigure the same image for Whisper and verify the rollback endpoint.

If the local Docker daemon is unavailable, static and Python-level tests may
proceed, but the work is not considered fully runtime-verified until the
container/GPU smoke test runs in a compatible Linux environment.

## Out of Scope

- vLLM `/v1/realtime` WebSocket support.
- Persistent FastConformer cache management across audio chunks.
- Concurrent RNNT request state beyond one active sequence.
- Automatic dual-model retry or keeping Whisper resident alongside Nemotron.
- Translation, diarization, and word-level timestamps.
- TurboQuant or other weight quantization for Nemotron.
- Changes to the existing WhisperX enhancement proposal beyond correcting
  statements that would otherwise claim Whisper is permanently mandatory.

## Completion Criteria

The feature is complete when the plugin is packaged and registered, all
repository tests pass, the transcription image serves Nemotron through the
unchanged OpenAI endpoint, a known WAV returns the expected text without state
leakage, the result matches direct Transformers within the agreed greedy decode
contract, and the documented Whisper rollback succeeds.
