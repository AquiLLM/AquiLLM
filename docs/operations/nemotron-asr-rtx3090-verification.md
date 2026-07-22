# Nemotron ASR RTX 3090 verification

Verified 2026-07-21 19:24 PDT (America/Los_Angeles) on one NVIDIA GeForce RTX
3090 (24,576 MiB), Windows driver 591.86. The host reports CUDA 13.1; the
container uses Torch 2.11.0+cu130 with CUDA 13.0. No chat model or other Docker
GPU container was running during any ASR phase. Windows WDDM desktop processes
were visible to `nvidia-smi` with `N/A` memory and were recorded rather than
treated as numeric compute allocations. This driver exposes no reliable
read-only `pmon`/type-C per-process memory signal for those WDDM rows, so that
remains a visibility limitation. The harness pairs the inventory with Docker
GPU `DeviceRequests`; any numeric compute allocation or GPU-requesting
container still fails isolation.

## Immutable runtime inputs

- Image: `aquillm-vllm-transcribe:test`, image ID and local repository digest
  `sha256:912270f4cd22011f2a4d985d4d41cd5f667a325f14ef982da87f46e341699d20`.
  The script also records the complete inspection JSON under its temporary
  artifact directory.
- vLLM 0.21.0, Transformers 5.13.0, Torch 2.11.0+cu130, librosa 0.11.0,
  tokenizers 0.22.2, safetensors 0.8.0.
- Installed plugin distribution 0.1.0 loaded from
  `/usr/local/lib/python3.12/dist-packages/aquillm_vllm_nemotron_asr/__init__.py`;
  no editable worktree package was used by the server.
- Model: `nvidia/nemotron-3.5-asr-streaming-0.6b` at revision
  `f3d333391852ba876df169dcc9ba902d25b6ab0b`.
- `model.safetensors`: 2,552,062,944 bytes, SHA-256
  `9eebdd6590289cb3030f310858f3df93256600a800a3e8200c5993d5f967e174`.
- Checkpoint inventory: 655 tensors and 637,997,088 parameters. Prefix counts
  were encoder 636, decoder 11, prompt projector 4, encoder projector 2, and
  joint 2. The plugin loaded every named parameter exactly once in FP32 with no
  unknown, duplicate, or missing names.
- Canonical audio SHA-256:
  `4e25e22555cd16e90edb0a3b49fdcf1fe652b2a1250ab643634db33895c75b41`.

The cache prefetch downloads exactly six files (config, generation config,
weights, processor config, tokenizer, and tokenizer config) into an isolated
named volume, verifies the weight bytes/hash/inventory, and all later parity
phases run with both Hugging Face and Transformers offline modes enabled.

## Direct Transformers and plugin parity

Direct Transformers and the plugin ran in separate sequential GPU containers;
neither process held the other model. Both used the same FP32 checkpoint,
processor outputs (`input_features` shape `1x586x128`, boolean attention mask
`1x586`, prompt ID 101), and complete 16 kHz fixture. Native
`AutoModelForRNNT.generate()` was called without `max_new_tokens` and without
importing plugin decoding or cleanup code.

- Native raw sequence: 125 IDs, including RNN-T blanks.
- Filtered native sequence: 50 IDs after removing only actual config blank
  13087. Its canonical JSON SHA-256 is
  `d20e566bff86fb909dfdb39a3663d9e5e4b7f1d7497914b99f717008b102ecce`.
- Native text: `Mr. Quilter is the apostle of the midle classes, and we are
  glad to welcome his gospel.`
- Corpus reference: `MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND
  WE ARE GLAD TO WELCOME HIS GOSPEL`.
- The native result has word error rate 0.117647 against that reference. Live
  correctness tests therefore require a nonempty transcript with WER no worse
  than 0.25, while duplicate requests must reproduce the identical normalized
  result. The model's spelling errors are not hardcoded as expected truth.
- Plugin replay before terminal matched all 50 filtered IDs exactly, including
  repetitions and order. Position 51 forced terminal blank 13087, giving the
  required N+1 output count and EOS/`stop`, never a length finish.
- Direct and plugin decoded raw and normalized text matched exactly.
- Captured acoustic joint logits had shape `124x13088` and `.npy` SHA-256
  `62a9cab709197fcd9588cd8d63eeb621e2a92760fcfe2b2eb9a233e282d8d269`.
  Argmax IDs matched exactly and every FP32 logit passed `rtol=1e-5,
  atol=1e-5`. Both processes disable TF32 and nondeterministic CUDA algorithms;
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` makes the cross-process comparison stable.
- Forced replay logits contained exactly one finite value and selected the
  expected replay ID. They were deliberately not compared with acoustic joint
  logits.

Immediately before the direct phase, the harness deletes both oracle outputs
and records its UTC start. The verified run started at
`2026-07-22T02:21:31.6328866Z`; `direct-joint-logits.npy` (6,491,776 bytes) and
`direct-transformers.json` (5,193 bytes) were rewritten at
`2026-07-22T02:21:43.5730963Z` and `2026-07-22T02:21:43.6099626Z`, respectively.
The harness required the direct pytest `2 passed` token, nonempty files, and
write timestamps no earlier than phase start before plugin parity could run.
These facts are persisted in `direct-phase-summary.json`.

## Deployed scheduler and memory

The standalone rendered runtime and all repository base/development/production
Compose profiles resolved the exact model, revision, and explicit
`VLLM_DTYPE=float32`. This explicit value is required: measurement proved that
vLLM 0.21 resolves this checkpoint's `auto` dtype to BF16. The actual server log
then proved `dtype=torch.float32`.

The engine ran V1 (`VLLM_USE_V2_MODEL_RUNNER=0`), eager, tensor parallel 1,
maximum sequences 1, model length 50,000, maximum batched tokens 50,000, and
maximum encoder input tokens 50,000. The wrapper satisfies `IsAttentionFree`,
contains no vLLM attention layers, and vLLM 0.21's exact
`get_kv_cache_groups(runtime, {})` result was empty. Consequently there were
zero attention KV groups/blocks/bytes; live metrics remained at 0.0% GPU KV
cache use. Encoder-decoder multimodal processor caching was also disabled by
the pinned runtime.

GPU memory was sampled from host `nvidia-smi` once per sampler cycle (Windows
WDDM makes each call slower than one second) across startup, all requests, and
the post-request steady window:

| Measurement | Overall GPU memory | Above WDDM baseline |
| --- | ---: | ---: |
| Baseline | 4,459 MiB | 0 MiB |
| Peak (FP32 long-boundary requests) | 20,733 MiB | 16,274 MiB |
| Post-request steady average | 20,720.2 MiB | 16,261.2 MiB |

The baseline was captured at `2026-07-21T19:22:14.5939211-07:00`. The HTTP
suite completed at `2026-07-21T19:23:38.8541956-07:00`, and the post-request
window ended at `2026-07-21T19:24:13.8725910-07:00`, 35.018 seconds later.
The steady average uses only the eight timestamped samples inside that window,
from `2026-07-21T19:23:39.1012829-07:00` through
`2026-07-21T19:24:12.0399745-07:00`; it does not include startup or request
samples. There were 24 samples across the full run. The resident FP32 model
therefore remained near the measured peak after requests completed.

`TRANSCRIBE_VLLM_GPU_MEMORY_UTILIZATION=0.20` started successfully and all
requests completed. It is retained because this attention-free model allocates
no decoder attention KV cache; the measured activation peak, rather than a KV
reservation, is the binding capacity fact. Do not colocate an unmeasured chat
engine with this FP32 ASR service on a 24 GiB card.

## Live OpenAI-compatible endpoint

The temporary Compose project started only `vllm_transcribe`, exposed exactly
`nemotron-3.5-asr-streaming-0.6b` from `/v1/models`, and passed all 29 tests in
`tests/asr` in 18.36 seconds of pytest runtime (27.902 seconds including host
runner setup/teardown). This covered SDK `.text`, normal JSON and text output, omitted and
explicit language, every request-validation 4xx, translation rejection,
sequential and duplicate state isolation, cancellation recovery, and 389/390
second acceptance plus 391 second rejection. The project and service were
removed in `finally`; the immutable Hugging Face cache volume was preserved.

The exact full command is:

```powershell
.\scripts\verify_nemotron_asr.ps1 -VerifyNemotron
```

The final full run was invoked from the system temporary directory to prove
that its absolute Dockerfile and build-context paths do not depend on the
caller's current directory. Before the host HTTP tests, the harness snapshots
all seven process environment variables it overrides and restores their exact
set/unset state and values in `finally`. The verification wrapper supplied a
mix of set and unset sentinel values and confirmed an identical state after
the full run.

Useful preparation/isolation-only commands are:

```powershell
.\scripts\verify_nemotron_asr.ps1 -PrepareEnvironments
.\scripts\verify_nemotron_asr.ps1 -AssertGpuIdle
.\scripts\verify_nemotron_asr.ps1 -SelfTest
```

Preparation creates standalone temporary Compose files plus complete Nemotron,
Whisper, and required-profile environment files without creating or changing
repository `.env`. It prints an activation script exposing their paths.

## Same-image Whisper rollback

Rollback was verified at 2026-07-21 19:49 PDT with the exact transcription
image ID above; the harness did not rebuild it. The live process command line
from `/proc/1/cmdline` used `openai/whisper-large-v3-turbo`, served alias
`whisper-large-v3-turbo`, FP16, model length 448, maximum batched tokens 1,500,
and one 30-second audio item. It contained no revision, Nemotron generation
configuration, or bitsandbytes/load-format/loader-extra flags. The 1,500-token
batch budget is required because vLLM 0.21 represents a 30-second Whisper audio
item as 1,500 encoder tokens even though the decoder model length remains 448.

The OpenAI SDK received a nonempty transcript while sending the nonempty prompt
`MISTER QUILTER`, proving the Nemotron-only request validator was inert. The
same image and Hugging Face cache were then recreated with the explicit
Nemotron environment, `/proc/1/cmdline` contained the pinned generation config
and no Whisper quantization flags, `/v1/models` exposed only the Nemotron alias,
and a second SDK transcription succeeded. Before and after image IDs were both
`sha256:912270f4cd22011f2a4d985d4d41cd5f667a325f14ef982da87f46e341699d20`.
`pip check` reported no broken requirements. Both services and their harness
network were removed, and the final GPU/container isolation check passed.

The initial documented rollback was corrected from 448 to 1,500 batched tokens
after vLLM rejected the smaller encoder budget. An attempted bitsandbytes
`load_in_8bit` configuration was also removed: vLLM 0.21 did not consume that
loader-extra setting for this unquantized checkpoint and entered its broken
4-bit Whisper fused-QKV path. Plain FP16 is the measured rollback configuration.

Run the complete environment-only rollback proof with:

```powershell
.\scripts\verify_nemotron_asr.ps1 -VerifyWhisperRollback
```

## Required multi-model profile

The required order is main chat (0.45), Nemotron transcription (0.20), embedding
(0.12), and reranking (0.08); optional OCR is explicitly excluded. The desktop
RTX 3090 run is inconclusive and is not a passing compatibility result. Its
retained warm-start log records the main checkpoint as 26.86 GiB while the
desktop Linux VM exposed only 26.81 GiB of available RAM; the user stopped the
follow-up run before it could prove a healthy first service. No profile service
is therefore recorded as passing, and the H100/256 GB report server remains the
authoritative target for the full measurement.

On that server, from the repository root with Docker/GPU access and no unrelated
GPU container, run:

```powershell
.\scripts\verify_nemotron_asr.ps1 -VerifyProfile
```

The harness generates a standalone Compose project, starts services in the
order above with in-container health and model probes, records VRAM after every
success, stops the newest service on failure, writes `profile-summary.json`
under the printed temporary artifact root, and removes only its own project.
