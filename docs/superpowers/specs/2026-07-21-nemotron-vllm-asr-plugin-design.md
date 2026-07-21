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
- Require vLLM 0.21's default V1 runner (`VLLM_USE_V2_MODEL_RUNNER=0`). The V2
  runner's transcription state is Whisper-specific and is not part of this
  first release.
- Treat omitted language as automatic detection. An optional ingestion
  language setting may pass an explicit locale.
- Keep the no-GPU development profile on hosted `whisper-1`.
- Use operational rollback, not automatic dual-model retry. Automatic fallback
  would require another resident model and is outside this scope.
- Pin Transformers to `5.13.0` and the Hugging Face checkpoint revision to
  `f3d333391852ba876df169dcc9ba902d25b6ab0b`. Record the base image's Torch,
  tokenizers, and CUDA versions during the build so dependency drift is visible.

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
- `state.py`: isolated single-request predecoded-token state, deterministic
  position lookup, terminal-token handling, reset, and profiling guards.
- `compat.py`: a version-gated vLLM 0.21 compatibility surface for endpoint
  request validation or lifecycle hooks that cannot be implemented through the
  public model protocol. It must reject unsupported vLLM versions and may not
  apply unrelated monkeypatches.

The wrapper will explicitly inherit `SupportsTranscription` and
`SupportsMultiModal`. It will set `supports_transcription_only=True`, declare
ISO-639-1 `supported_languages`, and implement the complete vLLM 0.21 contract:

- `get_speech_to_text_config()`;
- `validate_language()`;
- `get_generation_prompt()`;
- `get_num_audio_tokens()`;
- `post_process_output()`;
- multimodal processor registration;
- `embed_multimodal()`, `embed_input_ids()`, `forward()`, `compute_logits()`,
  and `load_weights()`.

The build/startup probe must verify that vLLM discovers the `transcription`
generation task and initializes the transcription route, not merely that the
architecture name appears in `ModelRegistry`.

The package must not monkeypatch broad vLLM internals during import. If a
vLLM 0.21 compatibility hook is necessary, it will live in an explicit,
version-checked module and fail fast on unsupported vLLM versions.

### Transcription image

`deploy/docker/vllm/Dockerfile.transcribe` will derive from
`vllm/vllm-openai:v0.21.0`, install media libraries, install exactly
Transformers 5.13.0, and install the local plugin package. Build-time
checks will verify:

1. the plugin entry point is discoverable;
2. Transformers exposes `Nemotron3_5AsrForRNNT` and its processor;
3. loading vLLM plugins registers the architecture;
4. the model class satisfies `SupportsTranscription` and advertises the
   `transcription` task.

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
runner:      V1 (`VLLM_USE_V2_MODEL_RUNNER=0`)
GPU budget:  0.20 initially; finalized by RTX 3090 startup measurement
model len:   50000, covering the 390-second theoretical RNNT replay ceiling
```

Because this explicit length exceeds the checkpoint's encoder-position value,
the service will opt into vLLM's long-model-length override. The wrapper does
not use text-model positional embeddings or a decoder KV cache; its decoder
position is only a replay index. Startup tests must prove the override does not
allocate an unintended text KV cache.

`INGEST_TRANSCRIBE_MODEL` must equal the served name. `.env.example` will
include complete Nemotron and Whisper blocks covering model, revision,
tokenizer, served name, dtype, GPU utilization, model length, trust settings,
and extra arguments. Nemotron arguments must not contain Whisper's
bitsandbytes flags. Existing local `.env` values remain operator-owned and will
not be overwritten wholesale.

## Request and Decode Flow

1. AquiLLM sends a complete media file to
   `POST /v1/audio/transcriptions` through the OpenAI Python client.
2. vLLM decodes/downmixes the media waveform and invokes the plugin's
   multimodal processor.
3. The Nemotron processor resamples to 16 kHz and creates the checkpoint's
   128-bin log-mel features and attention mask.
4. `get_generation_prompt()` returns an `ExplicitEncoderDecoderPrompt` with
   an `encoder_prompt=TextPrompt(...)` that contains both
   `multi_modal_data={"audio": (waveform, 16000)}` and per-request
   `mm_processor_kwargs={"language": locale}`, plus blank token `13087` as the
   one-token `decoder_prompt=TokensPrompt(...)`. These fields are nested inside
   the encoder singleton prompt; `ExplicitEncoderDecoderPrompt` itself contains
   only `encoder_prompt` and `decoder_prompt`. The processor maps the locale to
   `prompt_ids`. Missing language maps to `auto` rather than the Transformers
   pipeline's English default.
5. The FastConformer encoder processes the complete utterance once.
6. The Transformers-compatible greedy RNNT algorithm emits the transcript
   sequence. Blank token `13087` advances the encoder frame; nonblank tokens
   remain on the frame, with the checkpoint's ten-symbol-per-frame cap.
7. `embed_multimodal()` derives each item's valid encoder-frame length from the
   post-encoder attention mask, crops the corresponding encoder tensor to that
   length, and returns only the cropped tensors accepted by vLLM 0.21's
   `MultiModalEmbeddings` interface. The real request's `forward()` prefill call
   receives the cropped encoder output whether vLLM computed it for this call
   or recovered it from its multimodal content-hash cache. That prefill performs
   RNNT greedy decoding, atomically replaces the previous sequence, and appends
   a dedicated terminal token. Profiling/dummy calls may create provisional
   state, but every real prefill must overwrite it atomically before any
   user-visible replay.
8. Decode positions are derived from vLLM's absolute `positions` tensor and
   the one-token decoder prompt, never from a mutable counter. Positions inside
   the transcript force the corresponding token; the next and all later
   positions force the terminal token.
9. The terminal token is Nemotron blank/decoder-start token `13087`, configured
   as EOS in an image-local pinned generation configuration and omitted by
   special-token decoding. Empty transcripts therefore replay token `13087`
   immediately. Tests must prove
   the request finishes at the first terminal token rather than running to
   `max_tokens`.
10. RNNT blank emissions are control flow and are never inserted into the
    replay sequence. Token decoding preserves repeated emitted token IDs and
    removes terminal/language special tokens. The response remains
    `{ "text": "..." }`.

`max-num-seqs=1` and V1 make model-local state safe only when combined with
these rules. Each real prefill atomically overwrites state, positions rather
than a call counter select replay tokens, and cancellation/abort is followed by
a fresh prefill before any subsequent replay. Duplicate audio must rebuild
replay state correctly even when the multimodal encoder output comes from
vLLM's content-hash cache. The plugin must fail fast if V2 is enabled. If vLLM
0.21 does not expose enough lifecycle information to prove these invariants,
`compat.py` will add the smallest version-gated state lifecycle hook; this is a
planned deliverable, not an untracked fallback.

## API Behavior

- `language`: vLLM's public API accepts ISO-639-1 codes. The plugin overrides
  validation to also accept documented model locales. Deterministic defaults
  are `en -> en-US`, `es -> es-US`, `fr -> fr-FR`, `pt -> pt-BR`, and
  `zh -> zh-CN`; other unambiguous codes map to their only production locale.
  Explicit supported locales pass through. Missing language becomes `auto`,
  which selects prompt ID 101. Unsupported/adaptation-only choices return an
  actionable 4xx unless explicitly enabled by configuration.
- omitted language: automatic detection.
- `temperature`, `top_p`, `seed`, and beam controls: output remains greedy and
  deterministic because the model exposes one forced token per step. Nonzero
  temperature, beam search, or `n != 1` will be rejected with 4xx to avoid
  implying unsupported behavior.
- `prompt` and `hotwords`: nonempty values are rejected with 4xx.
- translations: `get_speech_to_text_config()` remains startup-safe for the
  `translate` handler, while `get_generation_prompt()` rejects translation
  requests with a stable 4xx.
- response: normal JSON and text transcription are required. HTTP output
  streaming may be rejected with 4xx in this batch-only release; realtime
  audio streaming is not registered. `verbose_json` and timestamp granularities
  are rejected because the checkpoint exposes token durations rather than
  native word segmentation.

The vLLM model protocol cannot inspect every HTTP sampling/output field.
`compat.py` will therefore add one version-checked validation call at the start
of vLLM 0.21's speech-to-text request creation path, active only for the
Nemotron architecture. It will enforce the option matrix and the 390-second
limit before generation. Prompt and translation checks are repeated in
`get_generation_prompt()` as defense in depth.

AquiLLM's current client does not send optional parameters, so these limitations
do not change the product's existing ingestion path.

## Length, Resources, and Safety

The checkpoint contains approximately 638 million FP32 parameters in a 2.55 GB
safetensors file. Practical GPU memory use is higher due to features,
activations, and vLLM overhead.

The encoder has 5,000 post-subsampling positions. Based on 10 ms feature frames
and 8x subsampling, offline utterances approach a hard ceiling near 400 seconds.
`get_speech_to_text_config()` will return
`SpeechToTextConfig(sample_rate=16000, max_audio_clip_s=390,
min_energy_split_window_size=None)`, which disables vLLM chunking. A version-
gated request check will reject audio longer than 390 seconds rather than
silently processing an over-limit clip. Boundary tests cover 389, 390, and 391
seconds. `TRANSCRIBE_VLLM_MAX_MODEL_LEN=50000` is separately sized for the
worst-case replay: 390 seconds / 80 ms per encoder frame * at most 10 nonblank
emissions per frame, plus prompt and terminal tokens. Tests use a synthetic
maximum-emission decoder to prove the terminal token fits and that vLLM never
returns a partial length-finished transcript.

The model weights are licensed under OpenMDW 1.1; the Transformers integration
code is Apache-2.0. AquiLLM will download rather than redistribute weights, and
documentation will identify the model license.

## Failure Handling and Rollback

- Unsupported plugin/runtime versions fail during image build or service
  startup, not on the first user upload.
- Unsupported language values return an actionable client error.
- Unsupported request options and translations return stable 4xx errors while
  server startup remains healthy.
- Empty transcripts remain errors in AquiLLM's existing ingestion adapter.
- Sequential requests, cache hits, cancellation, preprocessing errors, and
  dummy profiling must not leak forced-token state into the next utterance.
- The service health check gates dependent inference services as it does today.
- Operators can restore Whisper by changing the documented model, tokenizer,
  served name, `INGEST_TRANSCRIBE_MODEL`, and Whisper-specific extra arguments,
  then recreating `vllm_transcribe`.

## Verification Strategy

### Package tests

- Entry-point registration is re-entrant and lazily imports vLLM model code.
- The architecture is visible after plugin loading, satisfies the full
  `SupportsTranscription` protocol, and exposes the `transcription` task.
- Rendered encoder-decoder prompts retain `input_features`, `attention_mask`,
  `prompt_ids`, and lookahead metadata as batched audio fields.
- Language mapping covers `auto`, bare codes, explicit locales, and invalid
  values.
- Greedy RNNT state handling covers blank emissions, repeated real tokens,
  maximum symbols per frame, first/last/empty replay, explicit termination,
  duplicate inputs, cancellation, profiling, and reset between requests.
- Processor/model tests prove encoder outputs are cropped to their
  attention-mask-derived valid frame counts before entering replay decode.
- A synthetic 390-second worst-case replay fits prompt, 48,750 emitted tokens,
  and terminal token under the configured model length; no test path may return
  a partial transcript with a length finish reason.
- Output cleanup removes blank and language-tag special tokens without
  collapsing valid repetition.
- Lightweight mapping tests cover checkpoint tensor names; the full 2.55 GB
  weight-load assertion runs in container/GPU verification.

### Repository integration tests

- Only `vllm_transcribe` uses `Dockerfile.transcribe`.
- All three GPU Compose files agree on the Nemotron defaults and service kind.
- The no-GPU profile remains on hosted Whisper.
- The startup script selects transcription arguments by service kind.
- `INGEST_TRANSCRIBE_MODEL` matches the served alias.
- `INGEST_TRANSCRIBE_LANGUAGE` is in scope. `media.py` passes it only when set;
  unset behavior exercises Nemotron automatic detection and remains compatible
  with Whisper/OpenAI.
- Checked-in validation tests cover prompt, hotwords, temperature, beam, `n`,
  HTTP streaming, verbose responses, translations, and overlength audio.

### Container and GPU verification

1. Build the transcription image successfully.
2. Start the service on the available RTX 3090 and wait for `/health`.
3. Verify `/v1/models` exposes the Nemotron served alias.
4. Transcribe the checked-in LibriSpeech-derived, attributed fixture
   `tests/fixtures/audio/librispeech_1272-128104-0000.flac` through
   `/v1/audio/transcriptions` and assert its recorded exact normalized text plus
   the OpenAI SDK `.text` property. The fixture's source and redistribution
   terms must be recorded beside it.
5. Compare exact normalized decoded text and filtered emitted token IDs with
   direct Transformers 5.13.0 using the identical model revision, audio,
   processor kwargs, prompt ID, and FP32 dtype.
6. Verify omitted-language auto detection, explicit language, invalid language,
   and the complete request-validation matrix through HTTP.
7. Submit different sequential utterances, duplicate audio through a confirmed
   multimodal encoder-cache hit, and an aborted request followed by a successful
   request; verify every real prefill rebuilds replay state and no transcript or
   language state leaks.
8. Prove 389/390-second acceptance and 391-second rejection without truncation.
9. Reconfigure the same image with the complete documented Whisper environment
   block and verify the rollback endpoint.
10. Start the intended inference profile on the RTX 3090 and confirm the final
    GPU-utilization allocation fits alongside required services. If the full
    profile cannot fit, document the measured compatible profile instead of
    publishing an untested default.

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
contract, unsupported requests fail predictably, audio limits are enforced,
the measured RTX 3090 deployment budget is documented, and the documented
Whisper rollback succeeds.
