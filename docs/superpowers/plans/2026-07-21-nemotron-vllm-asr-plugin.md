# Nemotron 3.5 ASR vLLM Plugin Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve `nvidia/nemotron-3.5-asr-streaming-0.6b` through AquiLLM's existing OpenAI-compatible `/v1/audio/transcriptions` path using an out-of-tree vLLM 0.21 plugin, with batch-only deterministic RNNT decoding and environment-only Whisper rollback.

**Architecture:** A transcription-image-only Python package registers `Nemotron3_5AsrForRNNT` through `vllm.general_plugins`, adapts Hugging Face preprocessing and RNNT decoding to vLLM's encoder-decoder multimodal protocol, and uses absolute-position forced-token replay under a single-sequence V1/eager runtime. A narrowly version-gated compatibility hook validates options and clip length before generation. Compose selects a dedicated image and Nemotron defaults while the Django client keeps the same OpenAI SDK contract.

**Tech Stack:** Python 3.12, PyTorch, Transformers 5.13.0, vLLM 0.21.0, pytest, Docker/Compose, OpenAI Python SDK, NVIDIA CUDA on RTX 3090.

**Design spec:** `docs/superpowers/specs/2026-07-21-nemotron-vllm-asr-plugin-design.md`

---

## Execution prerequisites

- [ ] **Create an isolated host test environment without modifying the operator's existing `.venv`**

```powershell
$AsrHostVenv = Join-Path $env:TEMP "aquillm-asr-host-venv"
py -3.12 -m venv $AsrHostVenv
$AsrHostPython = Join-Path $AsrHostVenv "Scripts\python.exe"
& $AsrHostPython -m pip install --upgrade pip
& $AsrHostPython -m pip install -e .
```

Expected: `$AsrHostPython` has the repository's Python dependencies, including pytest, OpenAI, NumPy, and soundfile. Reuse this variable for every host command below. The plugin's Torch/Transformers/vLLM tests remain container-only.

## Chunk 1: Plugin foundation and deterministic RNNT core

### Task 1: Scaffold the installable vLLM plugin and prove re-entrant registration

**Files:**

- Create: `deploy/vllm_plugins/nemotron_asr/pyproject.toml`
- Create: `deploy/vllm_plugins/nemotron_asr/src/aquillm_vllm_nemotron_asr/__init__.py`
- Create: `deploy/vllm_plugins/nemotron_asr/src/aquillm_vllm_nemotron_asr/constants.py`
- Create: `deploy/vllm_plugins/nemotron_asr/tests/test_registration.py`
- Create: `deploy/vllm_plugins/nemotron_asr/tests/conftest.py`
- Modify: `pytest.ini`

- [ ] **Step 1: Write the registration contract test**

Test by parsing `pyproject.toml` with `tomllib` that the package declares exactly one `vllm.general_plugins` entry point named `aquillm_nemotron_asr`. Separately test that importing the package does not import `.model`, the first `register()` call writes the expected lazy target, and the second identical call is a no-op:

```python
ARCHITECTURE = "Nemotron3_5AsrForRNNT"
MODEL_CLASS = "aquillm_vllm_nemotron_asr.model:Nemotron3_5AsrForRNNT"
```

Use a fake `vllm` module in the host-side test so this test does not require a CUDA-capable vLLM installation. Give the fake registry a public `models` mapping shaped like vLLM 0.21's registry. Add a conflict case where `ARCHITECTURE` is already mapped to a different module/class and assert `register()` raises `RuntimeError` without overwriting it. Installed entry-point metadata is verified after the wheel is installed in Task 8.

- [ ] **Step 2: Run the focused test and confirm it fails**

```powershell
& $AsrHostPython -m pytest deploy/vllm_plugins/nemotron_asr/tests/test_registration.py -q
```

Expected: FAIL because the package and entry point do not exist.

- [ ] **Step 3: Add the package metadata and lazy registration function**

Pin no runtime dependencies in the wheel itself; the dedicated image owns the exact vLLM/Transformers versions. Define:

```toml
[project]
name = "aquillm-vllm-nemotron-asr"
version = "0.1.0"
requires-python = ">=3.10"

[project.entry-points."vllm.general_plugins"]
aquillm_nemotron_asr = "aquillm_vllm_nemotron_asr:register"
```

`register()` imports `ModelRegistry` only inside the function and registers the string import path. Inspect `ModelRegistry.models[ARCHITECTURE]` when present: compare its `module_name` and `class_name` to the two parts of `MODEL_CLASS`, return without writing on an identical match, and raise on any conflict. This deliberately avoids vLLM 0.21's default overwrite behavior.

Register `container`, `gpu`, and `asr_runtime` markers in root `pytest.ini` now, before any later task collects marked tests under `--strict-markers`.

- [ ] **Step 4: Run the focused test and install-metadata probe**

```powershell
& $AsrHostPython -m pytest deploy/vllm_plugins/nemotron_asr/tests/test_registration.py -q
& $AsrHostPython -m pip wheel --no-deps deploy/vllm_plugins/nemotron_asr -w $env:TEMP\nemotron-wheel
```

Expected: PASS and one wheel is built.

- [ ] **Step 5: Commit the scaffold**

```powershell
git add deploy/vllm_plugins/nemotron_asr pytest.ini
git commit -m "feat(asr): scaffold Nemotron vLLM plugin"
```

### Task 2: Implement language normalization and request-policy validation

**Files:**

- Create: `deploy/vllm_plugins/nemotron_asr/src/aquillm_vllm_nemotron_asr/languages.py`
- Create: `deploy/vllm_plugins/nemotron_asr/src/aquillm_vllm_nemotron_asr/validation.py`
- Create: `deploy/vllm_plugins/nemotron_asr/tests/test_languages.py`
- Create: `deploy/vllm_plugins/nemotron_asr/tests/test_validation.py`

- [ ] **Step 1: Write table-driven language tests**

Cover omitted/blank -> `auto`, `en -> en-US`, `es -> es-US`, `fr -> fr-FR`, `pt -> pt-BR`, `zh -> zh-CN`, every unambiguous production ISO code, and explicit production locales. The exact production set from the pinned NVIDIA model card is:

```text
en-US en-GB es-US es-ES fr-FR fr-CA it-IT pt-BR pt-PT nl-NL de-DE
tr-TR ru-RU ar-AR hi-IN ja-JP ko-KR vi-VN uk-UA pl-PL sv-SE cs-CZ
nb-NO da-DK bg-BG fi-FI hr-HR sk-SK zh-CN hu-HU ro-RO et-EE
```

The exact adaptation-ready set is `el-GR lt-LT lv-LV mt-MT sl-SI he-IL th-TH nn-NO`. Reject unknown and adaptation-ready locales by default with an error naming the value and supported choices. `normalize_language(value, *, allow_adaptation=False)` remains pure; its caller supplies `allow_adaptation` from `NEMOTRON_ASR_ALLOW_ADAPTATION_LANGUAGES=1`. Test both explicit boolean paths and the thin environment parser separately.

- [ ] **Step 2: Write request-option policy tests**

Use a tiny request dataclass/fake with all vLLM 0.21 `TranscriptionRequest` fields and encode this complete policy:

- accept task `transcribe`, `response_format in {"json", "text"}`, empty prompt/hotwords, `temperature in {None, 0}`, `use_beam_search=False`, `n=1`, no timestamps/stream options, and duration `<= 390.0`;
- accept `top_p`, `top_k`, `min_p`, `seed`, frequency/repetition/presence penalties, and `length_penalty` because one finite forced logit keeps output deterministic; tests prove these cannot change the forced argmax;
- reject `to_language`, `vllm_xargs`, non-default `max_completion_tokens`, `include_stop_str_in_output`, nonzero temperature, beam search, `n != 1`, prompt, hotwords, `stream` or either stream option, `verbose_json`/SRT/VTT/timestamps, translation, and `duration_s > 390.0`.

Define `validate_request(request, *, task_type: str, duration_s: float) -> str`; it returns the normalized locale and raises stable local errors that name the invalid field.

- [ ] **Step 3: Run the tests and confirm failure**

```powershell
& $AsrHostPython -m pytest deploy/vllm_plugins/nemotron_asr/tests/test_languages.py deploy/vllm_plugins/nemotron_asr/tests/test_validation.py -q
```

Expected: FAIL because normalization and validation are absent.

- [ ] **Step 4: Implement pure, side-effect-free helpers**

Keep the enumerated production/adaptation locales explicit and cite the pinned NVIDIA README URL in a source comment. `normalize_language()` must return a model locale rather than a prompt ID; the Transformers processor remains the source of prompt-ID truth. `validate_request()` returns the normalized language and raises a typed local exception that `compat.py` can translate into vLLM's normal 4xx response path.

- [ ] **Step 5: Run and commit**

```powershell
& $AsrHostPython -m pytest deploy/vllm_plugins/nemotron_asr/tests/test_languages.py deploy/vllm_plugins/nemotron_asr/tests/test_validation.py -q
git add deploy/vllm_plugins/nemotron_asr
git commit -m "feat(asr): validate Nemotron languages and requests"
```

Expected: PASS.

### Task 3: Implement the greedy RNNT decoder as a testable pure core

**Files:**

- Create: `deploy/vllm_plugins/nemotron_asr/src/aquillm_vllm_nemotron_asr/decoding.py`
- Create: `deploy/vllm_plugins/nemotron_asr/tests/test_decoding.py`
- Create: `deploy/vllm_plugins/nemotron_asr/tests/test_transformers_parity.py`

- [ ] **Step 1: Write deterministic decoder tests with scripted logits**

Keep this module framework-agnostic so its tests run in AquiLLM's existing host venv without Torch or Transformers. Cover:

- blank `13087` advances a frame and is not emitted;
- nonblank stays on the current frame;
- repeated nonblank token IDs are preserved;
- decoder cache update follows the current decoder input: the initial blank initializes the cache, a nonblank input updates it, and a later blank input reuses it;
- the tenth nonblank is emitted, then the ten-symbol cap forces a frame advance;
- attention-mask-derived valid lengths stop padded frames;
- batch size other than one fails fast for this release;
- an empty utterance yields no transcript tokens.

Expose the algorithm behind a small adapter protocol so scripted tests do not load the 2.55 GB checkpoint.

- [ ] **Step 2: Run and observe failure**

```powershell
& $AsrHostPython -m pytest deploy/vllm_plugins/nemotron_asr/tests/test_decoding.py -q
```

Expected: FAIL because `greedy_rnnt_decode` is missing.

- [ ] **Step 3: Implement exact Transformers-compatible control flow**

Define the adapter boundary explicitly: `adapter.predict(frame, previous_emitted_token, cache) -> (predicted_token, candidate_cache)`, where `candidate_cache` is the decoder result for the current input token. Match this pinned Transformers 5.13 invariant:

```python
while frame_index < valid_frames:
    token, candidate_cache = adapter.predict(frame, previous_token, cache)
    blank = token == BLANK_TOKEN_ID
    cache = candidate_cache                    # result of current input token
    previous_token = token                    # prediction is the next input
    if not blank:
        emitted.append(token)                 # preserve the tenth token
        symbols_on_frame += 1
    if blank or symbols_on_frame >= MAX_SYMBOLS_PER_FRAME:
        frame_index += 1
        symbols_on_frame = 0
```

The implementation must be checked line-by-line against `ParakeetRNNTGenerationMixin._update_model_kwargs_for_generation()` and the Nemotron model's decoder-cache update in Transformers tag `v5.13.0`. `adapter.predict()` returns the effective cache after processing the current `previous_token`: the initial blank initializes it, later blank inputs return it unchanged, and nonblank inputs update it. The prediction always becomes the next input, even when blank. Lock the tenth-token and nonblank->blank->next-frame transitions in tests.

- [ ] **Step 4: Add a parity harness against pinned Transformers**

Do not create a copied “reference” algorithm. In the pinned transcription container, monkeypatch the actual Transformers 5.13 Nemotron model's encoder/decoder/joint modules with deterministic tiny fakes, invoke its real `generate()`, and compare filtered emitted IDs with the plugin core for identical scripted logits. Duration/frame-advance behavior stays in the explicit blank/ten-symbol control-flow tests unless the core later exposes a typed trace. Mark this `@pytest.mark.container`; Task 8 runs it. Mark full-checkpoint parity `@pytest.mark.gpu` for Chunk 4. Host Task 3 runs only `test_decoding.py`.

- [ ] **Step 5: Run and commit**

```powershell
& $AsrHostPython -m pytest deploy/vllm_plugins/nemotron_asr/tests/test_decoding.py -q
git add deploy/vllm_plugins/nemotron_asr
git commit -m "feat(asr): add deterministic RNNT decoding"
```

Expected: PASS with repeated tokens unchanged.

### Task 4: Implement absolute-position replay state and lifecycle safety

**Files:**

- Create: `deploy/vllm_plugins/nemotron_asr/src/aquillm_vllm_nemotron_asr/state.py`
- Create: `deploy/vllm_plugins/nemotron_asr/tests/test_state.py`

- [ ] **Step 1: Write replay-state tests**

Prove atomic replacement on every real prefill; positions 1..N select transcript tokens after the one-token decoder prompt; position N+1 and later return terminal `13087`; empty transcripts terminate at the first generated position; first and last tokens are not skipped; a profiling sequence never survives a real prefill; duplicate inputs replace state; a simulated future cached-tensor prefill also replaces state; abort followed by prefill cannot leak; and 48,750 tokens plus prompt and terminal fit under 50,000.

- [ ] **Step 2: Run and observe failure**

```powershell
& $AsrHostPython -m pytest deploy/vllm_plugins/nemotron_asr/tests/test_state.py -q
```

Expected: FAIL because replay state is missing.

- [ ] **Step 3: Implement a lock-protected single-sequence state**

Use an immutable tuple snapshot and integer positions, never a mutable decode counter or Torch dependency in the core:

```python
class ReplayState:
    def replace_real(self, token_ids: Sequence[int]) -> None: ...
    def replace_profiling(self, token_ids: Sequence[int]) -> None: ...
    def forced_ids(self, positions: Sequence[int]) -> list[int]: ...
    def reset(self) -> None: ...
```

`replace_profiling()` may initialize or replace profiling state but must never overwrite an existing real snapshot; `replace_real()` always atomically overwrites any prior state. For a one-token prompt, lookup is `index = absolute_position - 1`: position 0 is invalid, positions 1..N return transcript IDs, and every position >= N+1 returns terminal `13087`. Preserve input order for repeated/noncontiguous positions. The model adapter converts the returned list to a `torch.long` tensor on the caller's device. `forced_ids()` must be a pure snapshot lookup so retries and repeated forward calls are idempotent.

- [ ] **Step 4: Run and commit**

```powershell
& $AsrHostPython -m pytest deploy/vllm_plugins/nemotron_asr/tests/test_state.py -q
git add deploy/vllm_plugins/nemotron_asr
git commit -m "feat(asr): add safe forced-token replay state"
```

Expected: PASS.

## Chunk 2: vLLM multimodal model adapter and compatibility layer

### Task 5: Implement the Nemotron multimodal processor and transcription protocol

**Files:**

- Create: `deploy/vllm_plugins/nemotron_asr/src/aquillm_vllm_nemotron_asr/processing.py`
- Create: `deploy/vllm_plugins/nemotron_asr/src/aquillm_vllm_nemotron_asr/model.py` (protocol/task-discovery scaffold; Task 6 fills runtime methods)
- Create: `deploy/vllm_plugins/nemotron_asr/tests/test_processing.py`
- Create: `deploy/vllm_plugins/nemotron_asr/tests/test_protocol.py`

- [ ] **Step 1: Write processor contract tests against vLLM 0.21**

Run these tests in the transcription build environment. Assert:

- `NemotronProcessingInfo` reports mono 16 kHz, one audio item, and the checkpoint's audio-token bound;
- dummy input generation respects 390 seconds without allocating transcript replay;
- `_call_hf_processor()` returns `input_features`, `attention_mask`, and `prompt_ids`;
- `_call_hf_processor()` converts Transformers 5.13's scalar `num_lookahead_tokens` into a one-value-per-audio tensor before vLLM field parsing;
- `_get_mm_fields_config()` marks `input_features`, `attention_mask`, `prompt_ids`, and normalized `num_lookahead_tokens` as `MultiModalFieldConfig.batched("audio")`;
- `create_encoder_prompt()` returns the required singleton dummy token;
- `_get_prompt_updates()` returns a `PromptReplacement` from `[0]` to `[0] * get_num_audio_tokens()`;
- omitted language reaches the HF processor as `auto`;
- rendered `ExplicitEncoderDecoderPrompt` contains only `encoder_prompt` and `decoder_prompt`, with audio/language nested under the encoder `TextPrompt` and decoder token `[13087]`.

- [ ] **Step 2: Run inside the base vLLM image and confirm failure**

```powershell
docker run --rm -v "${PWD}:/workspace" -w /workspace vllm/vllm-openai:v0.21.0 bash -lc "pip install -q pytest==8.4.1 transformers==5.13.0 -e deploy/vllm_plugins/nemotron_asr && pytest deploy/vllm_plugins/nemotron_asr/tests/test_processing.py deploy/vllm_plugins/nemotron_asr/tests/test_protocol.py -q"
```

Expected: FAIL before the processor/model protocol exists (or record Docker-daemon unavailability and rerun when available). Installing pytest explicitly ensures the failure is the missing contract, not an absent test runner.

- [ ] **Step 3: Implement `EncDecMultiModalProcessor` integration**

Follow vLLM 0.21's Whisper processor shape, replacing preprocessing with `Nemotron3_5AsrProcessor`. Register the processor using `MULTIMODAL_REGISTRY.register_processor(...)`. Implement `create_encoder_prompt()`, the `[0]` `PromptReplacement`, scalar-lookahead normalization, and all four batched audio fields. Create the model class scaffold now so protocol/task discovery is testable before Task 6; it inherits `nn.Module`, `SupportsTranscription`, `SupportsMultiModal`, and `IsAttentionFree`, exposes required classmethods, and leaves weight-bearing runtime construction for Task 6. `supported_languages` must be an ISO-639-1-to-English-name mapping because vLLM calls `.keys()`:

```python
supports_transcription_only = True
supported_languages = {
    "en": "English", "es": "Spanish", "fr": "French", "it": "Italian",
    # ...every bare ISO code represented by the 32 production locales
}

@classmethod
def get_speech_to_text_config(cls, model_config, task_type):
    return SpeechToTextConfig(
        sample_rate=16_000,
        max_audio_clip_s=390,
        min_energy_split_window_size=None,
    )
```

`get_generation_prompt()` validates transcription/prompt/hotwords, normalizes the locale, and returns the exact prompt structure in the design. vLLM's detokenizer must be configured/tested to skip tokenizer special IDs before `post_process_output(text)` runs. Since that protocol receives text only, `post_process_output()` may strip a residual terminal `<xx-XX>` locale tag and whitespace but may not pretend to decode token IDs; repetition preservation is asserted before and after detokenization.

- [ ] **Step 4: Run processor/protocol tests in the pinned environment**

```powershell
docker run --rm -v "${PWD}:/workspace" -w /workspace vllm/vllm-openai:v0.21.0 bash -lc "pip install -q pytest==8.4.1 transformers==5.13.0 -e deploy/vllm_plugins/nemotron_asr && pytest deploy/vllm_plugins/nemotron_asr/tests/test_processing.py deploy/vllm_plugins/nemotron_asr/tests/test_protocol.py -q"
```

Expected: PASS; the model class is recognized as both `SupportsMultiModal` and `SupportsTranscription`, and task discovery includes `transcription`.

- [ ] **Step 5: Commit**

```powershell
git add deploy/vllm_plugins/nemotron_asr
git commit -m "feat(asr): adapt Nemotron processing to vLLM"
```

### Task 6: Implement the model wrapper, encoder cropping, loader, and forced logits

**Files:**

- Modify: `deploy/vllm_plugins/nemotron_asr/src/aquillm_vllm_nemotron_asr/model.py`
- Create: `deploy/vllm_plugins/nemotron_asr/tests/test_model.py`
- Create: `deploy/vllm_plugins/nemotron_asr/tests/test_weight_loading.py`

- [ ] **Step 1: Write model tests with a tiny injected HF-compatible model**

Cover `embed_multimodal()` encoder invocation and post-encoder attention-mask cropping; fresh and cached encoder outputs taking the same real-prefill path; atomic decode-state replacement; `embed_input_ids()` shape; wrapper translation `vLLM position 0 -> ReplayState generated position 1`; position 0 predicting the first transcript token, the last transcript position, and the next terminal; one-hot/forced logits whose argmax is the replay token; terminal EOS on empty/complete transcripts; output cleanup; `IsAttentionFree` recognition with zero attention/KV-cache specs; and a complete mapping of checkpoint parameter names without silently ignoring required tensors.

- [ ] **Step 2: Run in the pinned vLLM environment and observe failure**

```powershell
docker run --rm -v "${PWD}:/workspace" -w /workspace vllm/vllm-openai:v0.21.0 bash -lc "pip install -q pytest==8.4.1 transformers==5.13.0 -e deploy/vllm_plugins/nemotron_asr && pytest deploy/vllm_plugins/nemotron_asr/tests/test_model.py deploy/vllm_plugins/nemotron_asr/tests/test_weight_loading.py -q"
```

Expected: FAIL until the wrapper is implemented; this command is independent of the later production image build.

- [ ] **Step 3: Implement the wrapper around Transformers modules**

Construct only the HF encoder/RNNT submodules needed for checkpoint parity; do not call `from_pretrained()` inside model initialization. Let vLLM own weight loading. Inherit `IsAttentionFree` so the runner does not interpret the 24 encoder layers as text self-attention or allocate decoder KV cache. During initialization validate `vllm_config.scheduler_config.max_num_seqs == 1`, `vllm_config.model_config.enforce_eager`, TP size 1, V1 runner, and the 50,000 scheduler limits. `embed_multimodal()` returns cropped encoder tensors. On every real prefill, `forward(..., encoder_outputs=...)` calls `greedy_rnnt_decode()` and replaces state before serving positions. The wrapper calls `state.forced_ids((positions + 1).tolist())`, then returns a `torch.long` tensor on `positions.device`. `compute_logits()` creates logits with a single finite maximum per row without allocating `[50000, vocab]`; allocate only `[active_positions, vocab_size]`.

The checkpoint uses identity top-level prefixes `encoder.*`, `decoder.*`, `encoder_projector.*`, `prompt_projector.*`, and `joint.*`; do not add a `model.` prefix. Use vLLM's `default_weight_loader`, track every iterator name, and compare it to `named_parameters()`. Permit only explicitly documented nonpersistent buffers or Transformers-tied aliases in a named allowlist; fail on unknown checkpoint names, duplicate loads, or missing parameters. Return the exact loaded-name set.

- [ ] **Step 4: Run model tests and the 48,750-token stress test**

```powershell
docker run --rm --gpus all -v "${PWD}:/workspace" -w /workspace vllm/vllm-openai:v0.21.0 bash -lc "pip install -q pytest==8.4.1 transformers==5.13.0 -e deploy/vllm_plugins/nemotron_asr && pytest deploy/vllm_plugins/nemotron_asr/tests/test_model.py deploy/vllm_plugins/nemotron_asr/tests/test_weight_loading.py deploy/vllm_plugins/nemotron_asr/tests/test_state.py -q"
```

Expected: PASS without a length-finished partial transcript or large persistent KV allocation.

- [ ] **Step 5: Commit**

```powershell
git add deploy/vllm_plugins/nemotron_asr
git commit -m "feat(asr): implement Nemotron vLLM model adapter"
```

### Task 7: Add the narrow vLLM 0.21 compatibility hook

**Files:**

- Create: `deploy/vllm_plugins/nemotron_asr/src/aquillm_vllm_nemotron_asr/compat.py`
- Create: `deploy/vllm_plugins/nemotron_asr/tests/test_compat.py`
- Modify: `deploy/vllm_plugins/nemotron_asr/src/aquillm_vllm_nemotron_asr/__init__.py`

- [ ] **Step 1: Write compatibility tests**

Assert exact vLLM version `0.21.0` acceptance and rejection of every other version; re-entrant patch installation; no behavioral change for non-Nemotron models; outer request-option validation before generation; duration derived from decoded/resampled audio, with 389 and 390 accepted and 391 rejected; stable 400-series `ErrorResponse`; and original method preservation. Include translation rejection even though vLLM initializes its translation handler for every transcription model.

- [ ] **Step 2: Run and observe failure**

```powershell
docker run --rm -v "${PWD}:/workspace" -w /workspace vllm/vllm-openai:v0.21.0 bash -lc "pip install -q pytest==8.4.1 transformers==5.13.0 -e deploy/vllm_plugins/nemotron_asr && pytest deploy/vllm_plugins/nemotron_asr/tests/test_compat.py -q"
```

Expected: FAIL because the hook is absent.

- [ ] **Step 3: Implement a version-checked serving wrapper**

Pin private-hook support to exactly vLLM `0.21.0`. Install two narrow wrappers on `OpenAISpeechToText`: an outer `_create_speech_to_text` wrapper validates all request fields before calling the original and catches the typed local error to return `self.create_error_response(message, err_type="BadRequestError", status_code=400)`; an `_preprocess_speech_to_text` wrapper awaits the original, reads its resampled `duration_s`, validates the 390-second boundary, and returns the unchanged tuple. Gate both on `self.model_cls` being the plugin class, so Whisper and every other model execute the exact original methods. Store idempotence sentinels and originals; never patch protocol models, FastAPI globally, or non-ASR routes.

Plugin registration fails immediately for the wrong vLLM version or `VLLM_USE_V2_MODEL_RUNNER=1`. Model initialization, where `VllmConfig` exists, fails for `max_num_seqs != 1`, eager execution disabled, TP != 1, or inconsistent scheduler lengths.

- [ ] **Step 4: Run and commit**

```powershell
docker run --rm -v "${PWD}:/workspace" -w /workspace vllm/vllm-openai:v0.21.0 bash -lc "pip install -q pytest==8.4.1 transformers==5.13.0 -e deploy/vllm_plugins/nemotron_asr && pytest deploy/vllm_plugins/nemotron_asr/tests/test_compat.py -q"
git add deploy/vllm_plugins/nemotron_asr
git commit -m "feat(asr): enforce Nemotron vLLM compatibility"
```

Expected: PASS with non-Nemotron behavior byte-for-byte unchanged.

### Task 8: Build the dedicated transcription image with pinned runtime probes

**Files:**

- Create: `deploy/docker/vllm/Dockerfile.transcribe`
- Create: `deploy/docker/vllm/nemotron_generation_config/generation_config.json`
- Create: `deploy/docker/vllm/probe_nemotron_plugin.py`
- Create: `aquillm/tests/integration/test_vllm_transcribe_image.py`
- Modify: `.dockerignore`

- [ ] **Step 1: Add static image-definition tests**

Require base `vllm/vllm-openai:v0.21.0`, exact `transformers==5.13.0`, local wheel installation without runtime network source, copied pinned generation-config directory with EOS `13087`, the plugin probe, media libraries, and the existing start scripts. Assert `Dockerfile` and `Dockerfile.genesis` remain free of Nemotron/Transformers upgrades.

- [ ] **Step 2: Run and observe failure**

```powershell
& $AsrHostPython -m pytest aquillm/tests/integration/test_vllm_transcribe_image.py -q
```

Expected: FAIL because `Dockerfile.transcribe` does not exist.

- [ ] **Step 3: Implement the image and startup/build probe**

At build time print and verify vLLM, Torch, CUDA, Transformers, tokenizers, and safetensors versions. The probe must load general plugins, verify the entry point and architecture registration, import `Nemotron3_5AsrForRNNT`/processor from Transformers, confirm the wrapper satisfies `SupportsMultiModal`, `SupportsTranscription`, and `IsAttentionFree`, confirm task discovery includes `transcription`, instantiate `SchedulerConfig(max_model_len=50000, is_encoder_decoder=True, is_multimodal_model=True, max_num_batched_tokens=50000, max_num_seqs=1)`, resolve the deployed `--generation-config /opt/aquillm/nemotron-generation-config` directory, and assert EOS is exactly `13087`. Reject V2.

- [ ] **Step 4: Run static test, build, and run package tests**

```powershell
& $AsrHostPython -m pytest aquillm/tests/integration/test_vllm_transcribe_image.py -q
docker build -f deploy/docker/vllm/Dockerfile.transcribe -t aquillm-vllm-transcribe:test .
docker run --rm aquillm-vllm-transcribe:test python3 /probe_nemotron_plugin.py
docker run --rm -v "${PWD}:/workspace" -w /workspace aquillm-vllm-transcribe:test bash -lc "pip install -q pytest==8.4.1 && pytest deploy/vllm_plugins/nemotron_asr/tests -q -m 'not gpu and not asr_runtime'"
```

Expected: all PASS. If Docker is unavailable, record the exact daemon failure; do not mark runtime verification complete.

- [ ] **Step 5: Commit**

```powershell
git add .dockerignore deploy/docker/vllm aquillm/tests/integration/test_vllm_transcribe_image.py
git commit -m "build(asr): add pinned Nemotron transcription image"
```

## Chunk 3: AquiLLM integration, deployment defaults, fixture, and operator docs

### Task 9: Pass an optional ingestion language without changing the API contract

**Files:**

- Modify: `aquillm/aquillm/ingestion/media.py`
- Modify: `aquillm/apps/ingestion/tests/test_transcribe_provider_selection.py`
- Modify: `.env.example`

- [ ] **Step 1: Write OpenAI SDK call-shape tests**

With `INGEST_TRANSCRIBE_LANGUAGE` unset/blank, assert `client.audio.transcriptions.create()` receives only `model` and `file`. With it set, assert it also receives `language`. In both cases, verify the returned `.text` is stripped and existing empty/error behavior is unchanged.

- [ ] **Step 2: Run and observe failure**

```powershell
& $AsrHostPython -m pytest aquillm/apps/ingestion/tests/test_transcribe_provider_selection.py -q
```

Expected: the new language-present case FAILS.

- [ ] **Step 3: Implement optional kwargs**

Build the kwargs dictionary before the SDK call:

```python
request = {"model": model, "file": file_obj}
language = (getenv("INGEST_TRANSCRIBE_LANGUAGE") or "").strip()
if language:
    request["language"] = language
response = client.audio.transcriptions.create(**request)
```

Document the variable near `INGEST_TRANSCRIBE_MODEL`; empty means Nemotron auto-detection and remains compatible with Whisper/OpenAI.

- [ ] **Step 4: Run and commit**

```powershell
& $AsrHostPython -m pytest aquillm/apps/ingestion/tests/test_transcribe_provider_selection.py -q
git add aquillm/aquillm/ingestion/media.py aquillm/apps/ingestion/tests/test_transcribe_provider_selection.py .env.example
git commit -m "feat(ingestion): pass optional ASR language"
```

Expected: PASS.

### Task 10: Wire the dedicated image and Nemotron defaults through Compose/startup

**Files:**

- Modify: `deploy/compose/base.yml`
- Modify: `deploy/compose/development.yml`
- Modify: `deploy/compose/production.yml`
- Modify: `deploy/scripts/vllm_start.sh`
- Modify: `.env.example`
- Modify: `aquillm/tests/integration/test_compose_multimodal_services.py`
- Modify: `aquillm/tests/integration/test_vllm_extra_args_parser.py`

- [ ] **Step 1: Write deployment invariants first**

For all three GPU Compose files assert only `vllm_transcribe` uses `Dockerfile.transcribe`; model/tokenizer/revision/served alias, `VLLM_DTYPE=float32`, GPU utilization, 50,000 model length, `VLLM_USE_V2_MODEL_RUNNER=0`, `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`, service kind, and trust settings agree. `TRANSCRIBE_VLLM_EXTRA_ARGS` contains eager, max sequences 1, max batched tokens 50,000, and `--generation-config /opt/aquillm/nemotron-generation-config`, with no bitsandbytes/TurboQuant flags. Assert no-GPU remains `whisper-1`. Test that `vllm_start.sh` restores transcription args via `VLLM_SERVICE_KIND=transcribe`, not model-name matching.

- [ ] **Step 2: Run and observe failure**

```powershell
& $AsrHostPython -m pytest aquillm/tests/integration/test_compose_multimodal_services.py aquillm/tests/integration/test_vllm_extra_args_parser.py -q
```

Expected: FAIL on the new Nemotron/image/service-kind assertions.

- [ ] **Step 3: Update Compose and startup**

Map all of these service variables explicitly in every GPU Compose file:

```text
TRANSCRIBE_VLLM_REVISION -> VLLM_REVISION
TRANSCRIBE_VLLM_DTYPE -> VLLM_DTYPE
TRANSCRIBE_VLLM_ALLOW_LONG_MAX_MODEL_LEN -> VLLM_ALLOW_LONG_MAX_MODEL_LEN
VLLM_SERVICE_KIND=transcribe
VLLM_USE_V2_MODEL_RUNNER=0
```

The Nemotron default sets the long-length variable to `1`; the Whisper rollback block sets it to `0`. Make the start script pass `--revision "$VLLM_REVISION"` only when nonempty and supported by vLLM 0.21. Recover `TRANSCRIBE_VLLM_EXTRA_ARGS` before model-name dispatch. Include service-kind and revision wrapper variables in the final unset block, but leave `VLLM_ALLOW_LONG_MAX_MODEL_LEN` and `VLLM_USE_V2_MODEL_RUNNER` visible to vLLM itself.

In `.env.example`, provide an active, complete Nemotron block and a commented, complete Whisper rollback block. The rollback block must restore the Whisper model, tokenizer, served alias, blank revision, float16 dtype, 448 model length, 1,500 batch-token limit (vLLM 0.21 requires at least the 30-second audio encoder budget), `TRANSCRIBE_VLLM_ALLOW_LONG_MAX_MODEL_LEN=0`, unquantized loading (vLLM 0.21's bitsandbytes loader selects its broken 4-bit fused-QKV path for this checkpoint), and matching `INGEST_TRANSCRIBE_MODEL`.

- [ ] **Step 4: Render Compose and verify scheduler arguments**

```powershell
docker compose --env-file .env.example -f deploy/compose/base.yml config | Out-File $env:TEMP\aquillm-base-rendered.yml
docker compose --env-file .env.example -f deploy/compose/development.yml config | Out-File $env:TEMP\aquillm-dev-rendered.yml
docker compose --env-file .env.example -f deploy/compose/production.yml config | Out-File $env:TEMP\aquillm-prod-rendered.yml
& $AsrHostPython -m pytest aquillm/tests/integration/test_compose_multimodal_services.py aquillm/tests/integration/test_vllm_extra_args_parser.py -q
```

Expected: rendered configurations select the new image and tests PASS.

- [ ] **Step 5: Commit**

```powershell
git add deploy/compose deploy/scripts/vllm_start.sh .env.example aquillm/tests/integration
git commit -m "feat(deploy): make Nemotron the local ASR default"
```

### Task 11: Add the attributed audio fixture and HTTP contract tests

**Files:**

- Create: `tests/fixtures/audio/librispeech_1272-128104-0000.flac`
- Create: `tests/fixtures/audio/librispeech_1272-128104-0000.txt`
- Create: `tests/fixtures/audio/README.md`
- Create: `tests/fixtures/audio/LICENSE-CC-BY-4.0.txt`
- Create: `tests/asr/test_nemotron_http_contract.py`
- Create: `tests/asr/test_nemotron_runtime.py`
- Create: `deploy/vllm_plugins/nemotron_asr/tests/test_engine_lifecycle.py`

- [ ] **Step 1: Acquire and document the redistributable fixture**

Use the LibriSpeech **dev-clean** utterance mirrored byte-for-byte by the official Qwen-Audio repository:

```text
URL: https://raw.githubusercontent.com/QwenLM/Qwen-Audio/b50fb958438081d36e1a14e93dbbc2f329c7f10e/assets/audio/1272-128104-0000.flac
SHA-256: 4e25e22555cd16e90edb0a3b49fdcf1fe652b2a1250ab643634db33895c75b41
Transcript: MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL
```

Record that Qwen-Audio is a single-file mirror, while OpenSLR SLR12 is the authoritative LibriSpeech upstream. Document speaker 1272, chapter 128104, utterance ID, LibriVox public-domain recording origin, LibriSpeech CC BY 4.0 terms, both URLs, and the exact hash. Download only that file and fail acquisition if the hash differs; do not commit an archive.

- [ ] **Step 2: Write HTTP contract tests behind an explicit marker**

Verify the `container`, `gpu`, and `asr_runtime` markers registered in Task 1. All network tests require both `RUN_ASR_RUNTIME=1` and an explicit `ASR_BASE_URL`; otherwise skip at collection/setup so ordinary `pytest` never contacts localhost.

Against that URL, assert `/models`, OpenAI SDK `.text`, plain `text`, omitted language, explicit `en`, invalid language, 389/390 acceptance and 391 rejection using generated low-amplitude WAVs, two distinct sequential utterances, and no state leakage. Raw multipart cases must assert stable 4xx error parameters for: `prompt`, `hotwords`, `temperature=0.5`, `use_beam_search=true`, `n=2`, `stream=true`, each stream option, `response_format=verbose_json|srt|vtt`, `timestamp_granularities[]=word|segment`, `to_language`, `vllm_xargs`, `max_completion_tokens`, and `/v1/audio/translations`. Keep SDK-compatible calls separate from vLLM-only raw multipart fields.

Pinned vLLM 0.21's scheduling-only `EncoderDecoderCacheManager.check_and_update_cache()` always returns `False`, and model config also disables its multimodal processor cache for encoder-decoder models. In `test_engine_lifecycle.py`, assert the selected manager is `EncoderDecoderCacheManager`, its cache check returns false, and two identical audio requests each invoke `embed_multimodal()` and each real `forward()` prefill replaces replay state. In `test_model.py`, separately feed the same cached encoder tensor directly through two prefills to prove wrapper idempotence as a future-runner guard without claiming a vLLM 0.21 cache hit. Abort an in-flight request through the engine's request ID/abort API, then submit a different request and prove a fresh prefill/state. The HTTP runtime test separately cancels an `httpx.AsyncClient.stream()` task after request start, then verifies a different follow-up request succeeds; engine counters are the lifecycle proof.

- [ ] **Step 3: Run collection/static fixture checks**

```powershell
& $AsrHostPython -m pytest --collect-only tests/asr -q
& $AsrHostPython -c "from pathlib import Path; import hashlib; p=Path('tests/fixtures/audio/librispeech_1272-128104-0000.flac'); print(hashlib.sha256(p.read_bytes()).hexdigest())"
```

Expected: tests collect and the hash matches `README.md`.

- [ ] **Step 4: Commit fixture and tests**

```powershell
git add tests/fixtures/audio tests/asr deploy/vllm_plugins/nemotron_asr/tests/test_engine_lifecycle.py
git commit -m "test(asr): add attributed Nemotron API fixture"
```

### Task 12: Update operator and pending WhisperX documentation

**Files:**

- Modify: `README.md`
- Modify: `docs/specs/2026-03-30-whisperx-transcription-design.md`
- Modify: `docs/roadmap/plans/pending/2026-03-30-whisperx-transcription-implementation.md`
- Modify: `docs/roadmap/roadmap-status.md`
- Modify: `aquillm/tests/integration/test_vllm_transcribe_image.py`

- [ ] **Step 1: Add documentation assertions**

Extend an appropriate integration/static test to require documentation of the Nemotron default, exact checkpoint revision, OpenMDW 1.1 link, batch/390-second/single-sequence limitations, automatic language detection, explicit language setting, GPU sizing caveat, health/smoke commands, and complete environment-only Whisper rollback.

- [ ] **Step 2: Correct Whisper-only baseline claims**

Describe WhisperX as an optional enhancer whose baseline input may come from the configured OpenAI-compatible ASR service (Nemotron default or Whisper rollback). Do not redesign or implement WhisperX.

- [ ] **Step 3: Run documentation/deployment tests and commit**

```powershell
& $AsrHostPython -m pytest aquillm/tests/integration/test_compose_multimodal_services.py aquillm/tests/integration/test_vllm_transcribe_image.py -q
git add README.md docs aquillm/tests/integration/test_vllm_transcribe_image.py
git commit -m "docs(asr): document Nemotron operation and rollback"
```

Expected: PASS and no document claims Whisper is permanently mandatory.

## Chunk 4: Container/GPU parity, rollback, and completion audit

### Task 13: Verify the full checkpoint against direct Transformers on RTX 3090

**Files:**

- Modify: `deploy/vllm_plugins/nemotron_asr/tests/test_transformers_parity.py`
- Modify: `deploy/vllm_plugins/nemotron_asr/tests/test_weight_loading.py`
- Create: `scripts/verify_nemotron_asr.ps1`
- Create: `docs/operations/nemotron-asr-rtx3090-verification.md`

- [ ] **Step 1: Add full-weight and exact parity tests**

Load revision `f3d333391852ba876df169dcc9ba902d25b6ab0b` in FP32 through two independent sequential paths so both copies are never resident together:

1. Direct Transformers 5.13.0 uses the pinned `AutoProcessor` and `AutoModelForRNNT.generate()` only. It may not import `greedy_rnnt_decode()` or any plugin cleanup helper. Save raw generated IDs, direct `skip_special_tokens=True` text, and config metadata to a temporary JSON file; unload the model, run GC, and empty CUDA cache.
2. The plugin/vLLM path loads the same revision and fixture, saves raw replay IDs before terminal plus final text, and compares to that JSON.

Assert all checkpoint tensors load exactly once, no required parameter is missing, filtered emitted IDs match before text normalization, normalized texts match separately, and repeated IDs are preserved.

- [ ] **Step 2: Enforce GPU isolation, then run lifecycle and full-checkpoint parity**

`verify_nemotron_asr.ps1 -AssertGpuIdle` inventories `nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory` and inspects every running Docker container's `HostConfig.DeviceRequests`. It aborts with both inventories if any GPU process/container remains after the verification project is down. It never kills unrelated work. An operator may explicitly pass repeatable `-AllowGpuPid <pid>` values, which are recorded in the verification document; there is no implicit allowlist.

```powershell
docker compose --project-name aquillm-asr-verification -f deploy/compose/base.yml down --remove-orphans
.\scripts\verify_nemotron_asr.ps1 -AssertGpuIdle
docker run --rm --gpus all -e VLLM_GPU_MEMORY_UTILIZATION=0.05 -v "${PWD}:/workspace" -w /workspace aquillm-vllm-transcribe:test bash -lc "pip install -q pytest==8.4.1 && pytest deploy/vllm_plugins/nemotron_asr/tests/test_engine_lifecycle.py deploy/vllm_plugins/nemotron_asr/tests/test_model.py -q"
docker run --rm --gpus all -v "${PWD}:/workspace" -w /workspace aquillm-vllm-transcribe:test bash -lc "pip install -q pytest==8.4.1 && pytest deploy/vllm_plugins/nemotron_asr/tests/test_transformers_parity.py deploy/vllm_plugins/nemotron_asr/tests/test_weight_loading.py -q -m gpu"
```

Expected: the tiny engine lifecycle suite runs with explicit 0.05 budget and exits before checkpoint loading; it proves duplicate recomputation, acknowledged abort, fresh prefill, terminal ID/count, and EOS finish reason. Then direct and plugin checkpoint paths run sequentially and exact token/text parity passes with no unallowlisted GPU process resident.

- [ ] **Step 3: Generate isolated runtime config and start only transcription**

`verify_nemotron_asr.ps1` creates temporary Nemotron and Whisper env files from the documented blocks plus a temporary Compose override setting `image: aquillm-vllm-transcribe:test`. It never edits `.env`. Service-level `environment` overrides `env_file: ../../.env`; the script renders `docker compose config` first and aborts unless every model/tokenizer/revision/dtype/length value matches the selected temporary env.

```powershell
.\scripts\verify_nemotron_asr.ps1 -PrepareEnvironments
docker compose --project-name aquillm-asr-verification --env-file $env:NEMOTRON_ASR_ENV -f deploy/compose/base.yml -f $env:NEMOTRON_ASR_OVERRIDE up -d --no-deps --wait --wait-timeout 900 vllm_transcribe
```

On timeout, print `docker compose ... ps` and `docker compose ... logs --no-color vllm_transcribe` with the identical project/env/file arguments, then fail. Immediately require `/v1/models` to contain exactly `nemotron-3.5-asr-streaming-0.6b`; otherwise stop rather than accidentally testing the operator's Whisper values.

- [ ] **Step 4: Run HTTP suites against the isolated service**

```powershell
$env:RUN_ASR_RUNTIME="1"
$env:ASR_BASE_URL="http://127.0.0.1:8005/v1"
& $AsrHostPython -m pytest tests/asr -q
```

Expected: correct `.text`, validation 4xxs, sequential/duplicate isolation, post-cancellation recovery, and 389/390/391 boundaries pass without launching a second engine beside the service.

- [ ] **Step 5: Measure scheduler, KV, and GPU behavior**

The probe and engine test expose/assert V1, eager, max sequences 1, model/batch tokens 50,000, `IsAttentionFree`, zero attention-layer KV specs, and zero decoder-attention KV bytes/blocks. Sample `nvidia-smi --query-compute-apps=used_gpu_memory --format=csv,noheader,nounits` before start, every second during load/request, and for 30 seconds afterward; record baseline, peak, and steady values. Assert engine output ends on EOS with exactly N+1 generated tokens. If utilization 0.20 is unsafe or materially wasteful, update it to the measured safe value and rerun Compose/runtime checks.

Write `docs/operations/nemotron-asr-rtx3090-verification.md` with timestamp, GPU, driver, CUDA, vLLM/Transformers/Torch versions, image ID/digest, revision, scheduler config, KV allocation, baseline/peak/steady VRAM, active services, final utilization, parity hashes/results, and exact verified command.

- [ ] **Step 6: Commit automation and measured evidence**

```powershell
git add scripts/verify_nemotron_asr.ps1 docs/operations/nemotron-asr-rtx3090-verification.md deploy/vllm_plugins/nemotron_asr .env.example deploy/compose
git commit -m "test(asr): verify Nemotron checkpoint parity"
```

### Task 14: Prove same-image Whisper rollback and intended-profile compatibility

**Files:**

- Modify: `scripts/verify_nemotron_asr.ps1`
- Modify: `README.md` only if measured instructions differ
- Modify: `docs/operations/nemotron-asr-rtx3090-verification.md`

- [ ] **Step 1: Recreate the same image with the documented Whisper block**

Record `docker image inspect aquillm-vllm-transcribe:test --format '{{.Id}}'`. Stop/remove only the verification project's transcription container, then start it with `$env:WHISPER_ASR_ENV`, the same override/image, `--no-deps --wait --wait-timeout 900`, and no rebuild. Render config first and assert model, tokenizer, blank revision, served alias, dtype, 448 model length, 1,500 batch-token limit, long-length flag 0, absence of unsupported bitsandbytes args, and `INGEST_TRANSCRIBE_MODEL=whisper-large-v3-turbo` switch together despite root `.env`. Require the Whisper alias and SDK transcript. Send a nonempty prompt and require success to prove the Nemotron validator is inert. Assert the image ID after recreation is unchanged.

- [ ] **Step 2: Restore Nemotron and test the intended inference profile**

Stop/remove Whisper and recreate Nemotron explicitly from `$env:NEMOTRON_ASR_ENV`; never infer “default” from `.env`. Re-assert alias and image ID.

Then test base Compose's required `vllm` profile services, excluding optional OCR: `vllm` at utilization 0.45, `vllm_transcribe` at measured/default 0.20, `vllm_embed` at 0.12, and `vllm_rerank` at 0.08. Start each with `--no-deps --wait --wait-timeout 900` in that order, probe `http://localhost:8000/health` inside each container via `docker compose exec -T`, and sample VRAM. On failure, collect project `ps` and service logs, stop the newest service to recover, and document the largest passing subset/order. Shut the verification project down when finished. Do not include `vllm_ocr` unless the separate `ocr-sidecar` profile is requested.

- [ ] **Step 3: Run rollback assertions**

```powershell
.\scripts\verify_nemotron_asr.ps1 -VerifyWhisperRollback -VerifyProfile
```

Expected: both models use the identical image ID; rollback needs environment changes only; the validator is inert for Whisper; and the profile harness truthfully records either the complete passing order or the largest passing prefix. The local RTX 3090 result may remain inconclusive when system RAM cannot load the main checkpoint; the user runs the same `-VerifyProfile` command on the H100/256 GB report server for the authoritative full-profile result.

- [ ] **Step 4: Commit any verification/doc correction**

```powershell
git add scripts/verify_nemotron_asr.ps1 README.md .env.example docs/operations/nemotron-asr-rtx3090-verification.md
git commit -m "test(asr): prove Whisper rollback path"
```

### Task 15: Run the complete audit and independent review

**Files:**

- Modify only files required by findings.

- [ ] **Step 1: Run all host-side regression tests**

```powershell
& $AsrHostPython -m pytest aquillm/apps/ingestion/tests/test_transcribe_provider_selection.py aquillm/tests/integration tests/unit -q
```

Expected: PASS.

- [ ] **Step 2: Run all plugin/container/runtime checks**

```powershell
docker build -f deploy/docker/vllm/Dockerfile.transcribe -t aquillm-vllm-transcribe:test .
docker run --rm aquillm-vllm-transcribe:test python3 /probe_nemotron_plugin.py
docker run --rm -v "${PWD}:/workspace" -w /workspace aquillm-vllm-transcribe:test bash -lc "pip install -q pytest==8.4.1 && pytest deploy/vllm_plugins/nemotron_asr/tests -q -m 'not gpu and not asr_runtime'"
.\scripts\verify_nemotron_asr.ps1 -VerifyNemotron -VerifyWhisperRollback -VerifyProfile
```

The verification script begins with `-AssertGpuIdle`, runs the lifecycle suite with the explicit 0.05 budget, then parity and service tests in the isolated order from Tasks 13-14. Expected: PASS with runtime evidence from the RTX 3090. A missing Docker daemon, dirty/unallowlisted GPU, or incompatible Linux/CUDA runtime is an explicit outstanding blocker, not a pass.

- [ ] **Step 3: Inspect exact deployment output and git hygiene**

```powershell
docker compose --env-file .env.example -f deploy/compose/base.yml config --quiet
docker compose --env-file .env.example -f deploy/compose/development.yml config --quiet
docker compose --env-file .env.example -f deploy/compose/production.yml config --quiet
$BaseSha = git rev-parse development
git diff --check
git diff --cached --check
git diff --check "$BaseSha...HEAD"
git status --short
git log --oneline "$BaseSha..HEAD"
```

Expected: valid Compose, no whitespace errors, no generated model weights/cache files, and small task-oriented commits.

- [ ] **Step 4: Request independent code review**

Use `superpowers:requesting-code-review` against the implementation base. Review specifically for vLLM 0.21 protocol conformance, RNNT parity, request-state leakage, option-validation coverage, image isolation, rollback accuracy, and fixture licensing. Fix every correctness finding and rerun the affected focused and full suites.

- [ ] **Step 5: Finish the branch**

Use `superpowers:verification-before-completion`, then `superpowers:finishing-a-development-branch` to present merge/PR options. Do not claim completion until the GPU smoke, Transformers parity, sequential/cache/abort tests, length boundaries, and Whisper rollback have actual passing evidence.
