"""Pinned-Transformers parity checks for the framework-independent RNNT core.

This suite is deliberately container-only: the dedicated transcription image owns
Torch and the exact Transformers release.  The host test environment skips it.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
import unicodedata
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

BLANK_TOKEN_ID = 13_087
MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"
MODEL_REVISION = "f3d333391852ba876df169dcc9ba902d25b6ab0b"
_FULL_PHASE = os.environ.get("ASR_FULL_PARITY_PHASE", "").strip()
_FIXTURE = (
    Path(__file__).resolve().parents[4]
    / "tests"
    / "fixtures"
    / "audio"
    / "librispeech_1272-128104-0000.flac"
)
_SIDECAR = _FIXTURE.with_suffix(".txt")


def _artifact_paths() -> tuple[Path, Path]:
    root = Path(os.environ["ASR_PARITY_ARTIFACT_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    return root / "direct-transformers.json", root / "direct-joint-logits.npy"


def _cache_dir() -> str:
    return os.environ.get("ASR_HF_CACHE_DIR", str(Path(os.environ["HF_HOME"]) / "hub"))


def _configure_cuda_determinism(torch: object) -> None:
    torch.backends.cuda.matmul.allow_tf32 = False  # type: ignore[union-attr]
    torch.backends.cudnn.allow_tf32 = False  # type: ignore[union-attr]
    torch.backends.cudnn.benchmark = False  # type: ignore[union-attr]
    torch.use_deterministic_algorithms(True)  # type: ignore[union-attr]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    lexical = "".join(
        character
        if character.isalnum() or character.isspace() or character == "'"
        else " "
        for character in text
    )
    return " ".join(lexical.split())


def _processor_inputs(processor: object, torch: object) -> dict[str, object]:
    import soundfile as sf

    audio, sample_rate = sf.read(_FIXTURE, dtype="float32")
    assert sample_rate == 16_000
    return processor(  # type: ignore[operator]
        audio=audio,
        sampling_rate=sample_rate,
        language="auto",
        is_streaming=False,
        is_first_audio_chunk=True,
        return_tensors="pt",
    )


def _input_metadata(inputs: dict[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in inputs.items():
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            result[name] = {
                "shape": list(value.shape),  # type: ignore[union-attr]
                "dtype": str(value.dtype),  # type: ignore[union-attr]
                "values": value.reshape(-1).tolist()  # type: ignore[union-attr]
                if name in {"prompt_ids", "num_lookahead_tokens"}
                else None,
            }
    return result


class _RecordingJoint:
    """Delegating module populated lazily after Torch is available."""

    @staticmethod
    def wrap(torch: object, joint: object) -> tuple[object, list[np.ndarray]]:
        records: list[np.ndarray] = []

        class RecordingJoint(torch.nn.Module):  # type: ignore[union-attr]
            def __init__(self) -> None:
                super().__init__()
                self.inner = joint

            def forward(self, *args: object, **kwargs: object) -> object:
                output = self.inner(*args, **kwargs)
                records.append(output.detach().float().cpu().numpy())
                return output

        return RecordingJoint(), records


@dataclass
class _ScriptedAdapter:
    predictions: deque[tuple[int, object]]

    def predict(
        self, frame: object, previous_emitted_token: int, cache: object | None
    ) -> tuple[int, object]:
        del frame, previous_emitted_token, cache
        return self.predictions.popleft()


@pytest.mark.container
def test_pinned_nemotron_generate_matches_the_pure_rnnt_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the real pinned generation mixin with tiny deterministic modules.

    Task 8 runs this inside the transcription image.  It intentionally invokes
    ``Nemotron3_5AsrForRNNT.generate`` instead of duplicating the Hugging Face
    generation algorithm in the test.
    """

    torch = pytest.importorskip("torch")
    from aquillm_vllm_nemotron_asr.decoding import greedy_rnnt_decode

    transformers = pytest.importorskip("transformers")
    if transformers.__version__ != "5.13.0":
        pytest.skip("requires the pinned Transformers 5.13.0 transcription image")

    try:
        from transformers import Nemotron3_5AsrConfig, Nemotron3_5AsrForRNNT
        from transformers.models.nemotron3_5_asr import modeling_nemotron3_5_asr
    except ImportError:
        pytest.skip(
            "Nemotron 3.5 ASR classes are unavailable in this Transformers build"
        )

    hidden_size = 4
    vocab_size = BLANK_TOKEN_ID + 1
    scripted_tokens = (91, BLANK_TOKEN_ID, 91, BLANK_TOKEN_ID)

    class TinyEncoder(torch.nn.Module):
        main_input_name = "input_features"

        def forward(self, input_features, attention_mask=None, **kwargs):
            del kwargs
            return SimpleNamespace(
                last_hidden_state=input_features,
                attention_mask=attention_mask,
                hidden_states=None,
                attentions=None,
                past_key_values=None,
                padding_cache=None,
            )

    class TinyDecoder(torch.nn.Module):
        """Use the pinned cache object's real blank/nonblank update semantics."""

        def forward(self, input_ids, cache=None):
            decoder_output = torch.zeros(
                input_ids.shape[0],
                input_ids.shape[1],
                hidden_size,
                device=input_ids.device,
            )
            if cache is None:
                return decoder_output

            blank_mask = input_ids[:, -1] == BLANK_TOKEN_ID
            if cache.is_initialized and blank_mask.all():
                return cache.cache

            was_initialized = cache.is_initialized
            if not was_initialized:
                cache.lazy_initialization(decoder_output)
            states = torch.zeros(
                1, input_ids.shape[0], hidden_size, device=input_ids.device
            )
            cache.update(
                decoder_output,
                states,
                states,
                mask=~blank_mask if was_initialized else None,
            )
            return cache.cache

    class ScriptedJoint(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._tokens = deque(scripted_tokens)

        def forward(self, decoder_hidden_states, encoder_hidden_states):
            del decoder_hidden_states
            token = self._tokens.popleft()
            logits = torch.full(
                (*encoder_hidden_states.shape[:-1], vocab_size),
                float("-inf"),
                device=encoder_hidden_states.device,
            )
            logits[..., token] = 0
            return logits

    # Avoid allocating the real FastConformer while still constructing the real
    # Nemotron class and invoking its inherited Transformers generation path.
    monkeypatch.setattr(
        modeling_nemotron3_5_asr.AutoModel,
        "from_config",
        staticmethod(lambda config: TinyEncoder()),
    )
    config = Nemotron3_5AsrConfig(
        vocab_size=vocab_size,
        decoder_hidden_size=hidden_size,
        num_decoder_layers=1,
        num_prompts=2,
        prompt_intermediate_size=hidden_size,
        default_prompt_id=0,
        decoder_start_token_id=BLANK_TOKEN_ID,
        eos_token_id=None,
    )
    config.encoder_config.hidden_size = hidden_size
    model = Nemotron3_5AsrForRNNT(config).eval()
    model.encoder = TinyEncoder()
    model.decoder = TinyDecoder()
    model.joint = ScriptedJoint()

    generated = (
        model.generate(
            input_features=torch.zeros((1, 2, hidden_size)),
            attention_mask=torch.ones((1, 2), dtype=torch.long),
            prompt_ids=torch.zeros((1,), dtype=torch.long),
            do_sample=False,
            max_new_tokens=len(scripted_tokens),
            return_dict_in_generate=True,
        )
        .sequences[0]
        .tolist()
    )
    transformers_ids = [token for token in generated if token != BLANK_TOKEN_ID]

    pure_adapter = _ScriptedAdapter(
        deque((token, object()) for token in scripted_tokens)
    )
    assert transformers_ids == greedy_rnnt_decode(
        [["frame-0", "frame-1"]], pure_adapter
    )


@pytest.mark.gpu
@pytest.mark.skipif(_FULL_PHASE != "direct", reason="set ASR_FULL_PARITY_PHASE=direct")
def test_full_checkpoint_direct_transformers_export() -> None:
    """Export an independent native-Transformers FP32 parity oracle."""
    assert not any(
        name.startswith("aquillm_vllm_nemotron_asr") for name in sys.modules
    ), "direct oracle process imported the plugin"
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    assert transformers.__version__ == "5.13.0"
    assert torch.cuda.is_available()
    _configure_cuda_determinism(torch)

    from transformers import AutoModelForRNNT, AutoProcessor

    artifact_json, artifact_logits = _artifact_paths()
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=_cache_dir(),
        local_files_only=True,
    )
    inputs = _processor_inputs(processor, torch)
    model = (
        AutoModelForRNNT.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=_cache_dir(),
            local_files_only=True,
            dtype=torch.float32,
        )
        .eval()
        .cuda()
    )
    assert int(model.config.blank_token_id) == BLANK_TOKEN_ID
    assert int(processor.tokenizer.pad_token_id) == BLANK_TOKEN_ID
    assert int(processor.tokenizer.convert_tokens_to_ids("<blank>")) == 13_088
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())

    model.joint, joint_records = _RecordingJoint.wrap(torch, model.joint)
    cuda_inputs = {
        name: value.cuda() if isinstance(value, torch.Tensor) else value
        for name, value in inputs.items()
    }
    with torch.inference_mode():
        generated = model.generate(**cuda_inputs)
    raw_ids = [int(token) for token in generated.sequences[0].tolist()]
    filtered_ids = [token for token in raw_ids if token != BLANK_TOKEN_ID]
    direct_text = processor.tokenizer.decode(raw_ids, skip_special_tokens=True)
    normalized_text = _canonical_text(direct_text)
    expected_text = _canonical_text(_SIDECAR.read_text(encoding="utf-8"))

    logits = np.concatenate(
        [record.reshape(-1, record.shape[-1]) for record in joint_records], axis=0
    ).astype(np.float32, copy=False)
    np.save(artifact_logits, logits, allow_pickle=False)
    artifact = {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "blank_token_id": BLANK_TOKEN_ID,
        "raw_sequence_ids": raw_ids,
        "filtered_ids": filtered_ids,
        "direct_text": direct_text,
        "normalized_text": normalized_text,
        "reference_normalized_text": expected_text,
        "matches_reference": normalized_text == expected_text,
        "input_metadata": _input_metadata(inputs),
        "audio_sha256": _sha256(_FIXTURE),
        "logits_shape": list(logits.shape),
        "logits_sha256": _sha256(artifact_logits),
        "config": {
            "vocab_size": int(model.config.vocab_size),
            "blank_token_id": int(model.config.blank_token_id),
            "decoder_start_token_id": getattr(
                model.config, "decoder_start_token_id", None
            ),
            "eos_token_id": getattr(model.config, "eos_token_id", None),
            "default_prompt_id": int(model.config.default_prompt_id),
        },
        "generation_config": model.generation_config.to_dict(),
    }
    artifact_json.write_text(
        json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8"
    )

    del generated, cuda_inputs, logits, model, processor
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.gpu
@pytest.mark.skipif(_FULL_PHASE != "plugin", reason="set ASR_FULL_PARITY_PHASE=plugin")
def test_full_checkpoint_plugin_matches_direct_export(monkeypatch) -> None:
    """Load the real wrapper strictly and compare its acoustic decode oracle."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    monkeypatch.setenv("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
    torch = pytest.importorskip("torch")
    assert torch.cuda.is_available()
    _configure_cuda_determinism(torch)

    from aquillm_vllm_nemotron_asr.model import Nemotron3_5AsrForRNNT
    from transformers import AutoProcessor
    from vllm.config import (
        DeviceConfig,
        LoadConfig,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        VllmConfig,
    )
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
    from vllm.utils.torch_utils import set_default_torch_dtype

    artifact_json, artifact_logits = _artifact_paths()
    direct = json.loads(artifact_json.read_text(encoding="utf-8"))
    assert direct["model_id"] == MODEL_ID
    assert direct["revision"] == MODEL_REVISION
    cache_dir = _cache_dir()

    model_config = ModelConfig(
        model=MODEL_ID,
        tokenizer=MODEL_ID,
        revision=MODEL_REVISION,
        tokenizer_revision=MODEL_REVISION,
        dtype="float32",
        enforce_eager=True,
        max_model_len=50_000,
        trust_remote_code=True,
        served_model_name="nemotron-3.5-asr-streaming-0.6b",
        generation_config="/opt/aquillm/nemotron-generation-config",
    )
    scheduler_config = SchedulerConfig(
        max_model_len=50_000,
        is_encoder_decoder=model_config.is_encoder_decoder,
        max_num_batched_tokens=50_000,
        max_num_seqs=1,
        enable_chunked_prefill=False,
        is_multimodal_model=model_config.is_multimodal_model,
    )
    load_config = LoadConfig(load_format="auto", download_dir=cache_dir)
    vllm_config = VllmConfig(
        model_config=model_config,
        scheduler_config=scheduler_config,
        parallel_config=ParallelConfig(tensor_parallel_size=1),
        device_config=DeviceConfig(device="cuda"),
        load_config=load_config,
    )
    loader = DefaultModelLoader(load_config)
    with set_default_torch_dtype(torch.float32), torch.device("cuda"):
        model = Nemotron3_5AsrForRNNT(vllm_config=vllm_config).eval()

    checkpoint_names: list[str] = []

    def tracked_weights():
        for name, tensor in loader.get_all_weights(model_config, model):
            checkpoint_names.append(name)
            yield name, tensor

    loaded_names = model.load_weights(tracked_weights())
    named_parameters = dict(model.named_parameters())
    assert len(checkpoint_names) == 655
    assert len(set(checkpoint_names)) == 655
    assert loaded_names == set(checkpoint_names) == set(named_parameters)
    assert (
        sum(parameter.numel() for parameter in named_parameters.values()) == 637_997_088
    )
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    prefix_counts = {
        prefix: sum(name.startswith(f"{prefix}.") for name in checkpoint_names)
        for prefix in (
            "encoder",
            "decoder",
            "prompt_projector",
            "encoder_projector",
            "joint",
        )
    }
    assert prefix_counts == {
        "encoder": 636,
        "decoder": 11,
        "prompt_projector": 4,
        "encoder_projector": 2,
        "joint": 2,
    }

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=cache_dir,
        local_files_only=True,
    )
    inputs = _processor_inputs(processor, torch)
    assert _input_metadata(inputs) == direct["input_metadata"]
    cuda_inputs = {
        name: value.cuda() if isinstance(value, torch.Tensor) else value
        for name, value in inputs.items()
    }
    model.joint, joint_records = _RecordingJoint.wrap(torch, model.joint)
    with torch.inference_mode():
        encoder_outputs = model.embed_multimodal(**cuda_inputs)
        first_hidden = model.forward(
            input_ids=torch.tensor([BLANK_TOKEN_ID], device="cuda"),
            positions=torch.tensor([0], device="cuda"),
            encoder_outputs=encoder_outputs,
        )

    expected_ids = [int(token) for token in direct["filtered_ids"]]
    replay = model.replay_state.forced_ids(range(1, len(expected_ids) + 2))
    assert replay[:-1] == expected_ids
    assert replay[-1] == BLANK_TOKEN_ID
    assert len(replay) == len(expected_ids) + 1
    assert first_hidden.reshape(-1).to(dtype=torch.long).tolist() == replay[:1]
    plugin_text = processor.tokenizer.decode(replay[:-1], skip_special_tokens=True)
    assert plugin_text == direct["direct_text"]
    assert _canonical_text(plugin_text) == direct["normalized_text"]

    plugin_logits = np.concatenate(
        [record.reshape(-1, record.shape[-1]) for record in joint_records], axis=0
    ).astype(np.float32, copy=False)
    direct_logits = np.load(artifact_logits, allow_pickle=False)
    assert plugin_logits.shape == direct_logits.shape
    np.testing.assert_array_equal(
        plugin_logits.argmax(axis=-1), direct_logits.argmax(axis=-1)
    )
    np.testing.assert_allclose(plugin_logits, direct_logits, rtol=1e-5, atol=1e-5)

    forced_logits = model.compute_logits(first_hidden)
    assert forced_logits.shape == (1, 13_088)
    assert torch.isfinite(forced_logits).sum().item() == 1
    assert int(forced_logits.argmax(dim=-1).item()) == replay[0]
