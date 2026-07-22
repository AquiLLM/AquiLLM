"""Pinned vLLM 0.21 in-process lifecycle proof for replay state."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from types import MethodType

import pytest

os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip(
        "CUDA is required for the in-process vLLM lifecycle test",
        allow_module_level=True,
    )

np = pytest.importorskip("numpy")
pytest.importorskip("vllm")

from aquillm_vllm_nemotron_asr import model as model_module  # noqa: E402
from aquillm_vllm_nemotron_asr.decoding import BLANK_TOKEN_ID  # noqa: E402
from vllm.engine.arg_utils import EngineArgs  # noqa: E402
from vllm.engine.llm_engine import LLMEngine  # noqa: E402
from vllm.inputs import (  # noqa: E402
    ExplicitEncoderDecoderPrompt,
    TextPrompt,
    TokensPrompt,
)
from vllm.sampling_params import SamplingParams  # noqa: E402
from vllm.v1.core.encoder_cache_manager import (  # noqa: E402
    EncoderDecoderCacheManager,
)
from vllm.v1.engine.core_client import InprocClient  # noqa: E402

pytestmark = pytest.mark.gpu

_MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"
_REVISION = "f3d333391852ba876df169dcc9ba902d25b6ab0b"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_GENERATION_CONFIG = (
    _REPOSITORY_ROOT / "deploy" / "docker" / "vllm" / "nemotron_generation_config"
)


def _tiny_hf_overrides() -> dict[str, object]:
    """Keep the checkpoint protocol while shrinking only compute dimensions."""
    return {
        "architectures": ["Nemotron3_5AsrForRNNT"],
        "vocab_size": 13_088,
        "blank_token_id": BLANK_TOKEN_ID,
        "decoder_hidden_size": 8,
        "num_decoder_layers": 1,
        "num_prompts": 128,
        "prompt_intermediate_size": 16,
        "default_prompt_id": 101,
        "is_encoder_decoder": True,
        "encoder_config": {
            "model_type": "nemotron_asr_streaming_encoder",
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "intermediate_size": 16,
            "hidden_act": "silu",
            "attention_bias": False,
            "convolution_bias": False,
            "conv_kernel_size": 9,
            "subsampling_factor": 8,
            "subsampling_conv_channels": 8,
            "num_mel_bins": 128,
            "subsampling_conv_kernel_size": 3,
            "subsampling_conv_stride": 2,
            "dropout": 0.0,
            "dropout_positions": 0.0,
            "layerdrop": 0.0,
            "activation_dropout": 0.0,
            "attention_dropout": 0.0,
            "max_position_embeddings": 5_000,
            "scale_input": False,
            "initializer_range": 0.02,
            "sliding_window": 57,
            "default_num_lookahead_tokens": 3,
            "supported_num_lookahead_tokens": [3, 0, 6, 13],
        },
    }


@pytest.fixture(scope="module")
def engine() -> Iterator[LLMEngine]:
    args = EngineArgs(
        model=_MODEL_ID,
        tokenizer=_MODEL_ID,
        revision=_REVISION,
        tokenizer_revision=_REVISION,
        trust_remote_code=True,
        dtype="float32",
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=50_000,
        max_num_batched_tokens=50_000,
        max_num_seqs=1,
        gpu_memory_utilization=0.05,
        skip_mm_profiling=True,
        limit_mm_per_prompt={"audio": 1},
        load_format="dummy",
        hf_overrides=_tiny_hf_overrides(),
        generation_config=str(_GENERATION_CONFIG),
        disable_log_stats=True,
    )
    result = LLMEngine.from_engine_args(args, enable_multiprocessing=False)
    try:
        yield result
    finally:
        result.engine_core.shutdown()


def _prompt(sample: int) -> ExplicitEncoderDecoderPrompt:
    waveform = np.full(1_600, sample, dtype=np.float32)
    return ExplicitEncoderDecoderPrompt(
        encoder_prompt=TextPrompt(
            prompt="",
            multi_modal_data={"audio": (waveform, 16_000)},
            mm_processor_kwargs={"language": "en-US"},
        ),
        decoder_prompt=TokensPrompt(prompt_token_ids=[BLANK_TOKEN_ID]),
    )


def _run_to_completion(engine: LLMEngine, request_id: str, sample: int):
    engine.add_request(
        request_id,
        _prompt(sample),
        SamplingParams(temperature=0, max_tokens=8),
    )
    final = None
    while engine.has_unfinished_requests():
        for output in engine.step():
            if output.request_id == request_id:
                final = output
    assert final is not None and final.finished
    return final


def test_inproc_engine_recomputes_prefills_and_recovers_after_abort(
    engine: LLMEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert isinstance(engine.engine_core, InprocClient)
    core = engine.engine_core.engine_core
    cache_manager = core.scheduler.encoder_cache_manager
    assert isinstance(cache_manager, EncoderDecoderCacheManager)
    assert cache_manager.check_and_update_cache(object(), 0) is False
    multimodal_config = engine.vllm_config.model_config.multimodal_config
    assert multimodal_config is not None
    assert multimodal_config.mm_processor_cache_gb == 0.0

    worker = engine.model_executor.driver_worker.worker
    model = worker.model_runner.model
    counters = {"embed": 0, "prefill": 0, "replace": 0}
    decode_sequences = iter(([41], [41], [60, 61, 62], [90]))

    original_embed = model.embed_multimodal
    original_forward = model.forward
    original_replace = model.replay_state.replace_real

    def counted_embed(self, **kwargs):
        counters["embed"] += 1
        return original_embed(**kwargs)

    def counted_forward(self, input_ids, positions, encoder_outputs=None, **kwargs):
        if encoder_outputs is not None:
            counters["prefill"] += 1
        return original_forward(
            input_ids,
            positions,
            encoder_outputs=encoder_outputs,
            **kwargs,
        )

    def counted_replace(token_ids):
        counters["replace"] += 1
        return original_replace(token_ids)

    monkeypatch.setattr(model, "embed_multimodal", MethodType(counted_embed, model))
    monkeypatch.setattr(model, "forward", MethodType(counted_forward, model))
    monkeypatch.setattr(model.replay_state, "replace_real", counted_replace)
    monkeypatch.setattr(
        model_module,
        "greedy_rnnt_decode",
        lambda *args, **kwargs: list(next(decode_sequences)),
    )

    first = _run_to_completion(engine, "duplicate-1", sample=1)
    second = _run_to_completion(engine, "duplicate-2", sample=1)
    assert counters == {"embed": 2, "prefill": 2, "replace": 2}
    assert first.outputs[0].token_ids == second.outputs[0].token_ids

    engine.add_request(
        "abort-after-prefill",
        _prompt(sample=2),
        SamplingParams(temperature=0, max_tokens=8),
    )
    for _ in range(4):
        engine.step()
        if counters["prefill"] == 3:
            break
    assert counters == {"embed": 3, "prefill": 3, "replace": 3}
    assert engine.has_unfinished_requests()
    engine.abort_request(["abort-after-prefill"])
    assert not engine.has_unfinished_requests()

    fresh = _run_to_completion(engine, "fresh-after-abort", sample=3)
    assert counters == {"embed": 4, "prefill": 4, "replace": 4}
    assert fresh.finished
    assert fresh.outputs[0].finish_reason == "stop"
    assert fresh.outputs[0].finish_reason != "length"
    assert fresh.outputs[0].token_ids[-1] == BLANK_TOKEN_ID
    assert model.replay_state.forced_ids([2]) == [BLANK_TOKEN_ID]
