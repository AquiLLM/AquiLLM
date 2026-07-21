"""Runtime contract tests for the pinned vLLM Nemotron adapter."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from aquillm_vllm_nemotron_asr import model as model_module  # noqa: E402
from aquillm_vllm_nemotron_asr.decoding import BLANK_TOKEN_ID  # noqa: E402
from aquillm_vllm_nemotron_asr.state import ReplayState  # noqa: E402


class _Encoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, object]] = []
        self.post_attention_mask: torch.Tensor | None = None

    def forward(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        features = kwargs["input_features"]
        batch, frames = features.shape[:2]
        hidden = torch.arange(batch * frames * 2, dtype=torch.float32).reshape(
            batch, frames, 2
        )
        return SimpleNamespace(
            last_hidden_state=hidden,
            attention_mask=self.post_attention_mask,
        )


class _Identity(torch.nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values


class _PromptProjector(torch.nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values[..., :2]


def _bare_model() -> model_module.Nemotron3_5AsrForRNNT:
    result = object.__new__(model_module.Nemotron3_5AsrForRNNT)
    torch.nn.Module.__init__(result)
    result.config = SimpleNamespace(
        num_prompts=3,
        default_prompt_id=0,
        vocab_size=13088,
        blank_token_id=BLANK_TOKEN_ID,
    )
    result.encoder = _Encoder()
    result.encoder_projector = _Identity()
    result.prompt_projector = _PromptProjector()
    result.decoder = _Identity()
    result.joint = _Identity()
    result.replay_state = ReplayState()
    return result


def test_embed_multimodal_runs_audio_path_and_crops_post_encoder_mask() -> None:
    model = _bare_model()
    model.encoder.post_attention_mask = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0]])
    output = model.embed_multimodal(
        input_features=torch.zeros((2, 4, 3)),
        attention_mask=torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]]),
        prompt_ids=torch.tensor([1, 2]),
        num_lookahead_tokens=torch.tensor([0, 0]),
    )

    assert [item.shape for item in output] == [(1, 2), (2, 2)]
    assert torch.equal(output[0], torch.tensor([[0.0, 1.0]]))
    assert len(model.encoder.calls) == 1
    assert set(model.encoder.calls[0]) == {
        "input_features",
        "attention_mask",
        "num_lookahead_tokens",
    }
    assert model.encoder.calls[0]["num_lookahead_tokens"] == 0


def test_embed_multimodal_preserves_a_zero_length_post_encoder_item() -> None:
    model = _bare_model()
    model.encoder.post_attention_mask = torch.tensor([[0, 0], [1, 0]])

    output = model.embed_multimodal(
        input_features=torch.zeros((2, 2, 3)),
        attention_mask=torch.tensor([[1, 1], [1, 1]]),
        prompt_ids=torch.tensor([1, 2]),
    )

    assert [item.shape for item in output] == [(0, 2), (1, 2)]


def test_forward_replaces_replay_state_for_fresh_and_future_cached_encoder_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _bare_model()
    decoded: list[object] = []

    def fake_decode(frames: object, adapter: object, **kwargs: object) -> list[int]:
        decoded.append(frames)
        return [42] if len(decoded) == 1 else [99, 100]

    monkeypatch.setattr(model_module, "greedy_rnnt_decode", fake_decode)
    positions = torch.tensor([0, 1, 3])

    fresh = model.forward(
        torch.tensor([BLANK_TOKEN_ID]), positions, encoder_outputs=[torch.ones((2, 2))]
    )
    cached = model.forward(
        torch.tensor([BLANK_TOKEN_ID]), positions, encoder_outputs=[torch.ones((2, 2))]
    )

    assert decoded and len(decoded) == 2
    assert fresh.tolist() == [[42], [BLANK_TOKEN_ID], [BLANK_TOKEN_ID]]
    assert cached.tolist() == [[99], [100], [BLANK_TOKEN_ID]]
    assert model.replay_state.forced_ids([1, 2, 3]) == [99, 100, BLANK_TOKEN_ID]


def test_forward_translates_vllm_positions_and_logits_force_one_id_per_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _bare_model()
    monkeypatch.setattr(
        model_module, "greedy_rnnt_decode", lambda *args, **kwargs: [7, 8]
    )

    forced_ids = model.forward(
        torch.tensor([BLANK_TOKEN_ID]),
        torch.tensor([0, 1, 2, 1]),
        encoder_outputs=[torch.ones((1, 2))],
    )
    logits = model.compute_logits(forced_ids)

    assert forced_ids.dtype == torch.long
    assert forced_ids.tolist() == [[7], [8], [BLANK_TOKEN_ID], [8]]
    assert logits.shape == (4, 13088)
    assert logits.argmax(dim=-1).tolist() == forced_ids.flatten().tolist()
    assert torch.isfinite(logits).sum(dim=-1).tolist() == [1, 1, 1, 1]


def test_empty_transcript_forces_terminal_blank_without_large_logits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _bare_model()
    monkeypatch.setattr(model_module, "greedy_rnnt_decode", lambda *args, **kwargs: [])

    forced_ids = model.forward(
        torch.tensor([BLANK_TOKEN_ID]),
        torch.tensor([0, 7]),
        encoder_outputs=[torch.empty((0, 2))],
    )

    assert forced_ids.tolist() == [[BLANK_TOKEN_ID], [BLANK_TOKEN_ID]]
    assert model.compute_logits(forced_ids).shape == (2, 13088)


def test_embed_input_ids_returns_active_token_embeddings_on_input_device() -> None:
    model = _bare_model()
    input_ids = torch.tensor([1, 2, 3])

    embedded = model.embed_input_ids(input_ids)

    assert embedded.shape == (3, 1)
    assert embedded.device == input_ids.device
    assert embedded.dtype == torch.float32


def test_init_rehomes_only_checkpoint_identity_modules_without_pretrained_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TinyHfModel:
        def __init__(self, config: object) -> None:
            self.encoder = torch.nn.Linear(2, 2, bias=False)
            self.decoder = torch.nn.Linear(2, 2, bias=False)
            self.encoder_projector = torch.nn.Linear(2, 2, bias=False)
            self.prompt_projector = torch.nn.Linear(2, 2, bias=False)
            self.joint = torch.nn.Linear(2, 2, bias=False)

    monkeypatch.setattr(model_module, "HfNemotron3_5AsrForRNNT", TinyHfModel)
    config = SimpleNamespace(num_prompts=3, default_prompt_id=0, vocab_size=13088)
    runtime = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            max_num_seqs=1,
            max_num_batched_tokens=50_000,
            max_num_encoder_input_tokens=50_000,
        ),
        model_config=SimpleNamespace(
            enforce_eager=True, max_model_len=50_000, hf_config=config
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )

    model = model_module.Nemotron3_5AsrForRNNT(vllm_config=runtime)

    assert {name.split(".", 1)[0] for name, _ in model.named_parameters()} == {
        "encoder",
        "decoder",
        "encoder_projector",
        "prompt_projector",
        "joint",
    }
    assert not hasattr(model, "model")


def test_init_rejects_the_pinned_v2_model_runner_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            max_num_seqs=1,
            max_num_batched_tokens=50_000,
            max_num_encoder_input_tokens=50_000,
        ),
        model_config=SimpleNamespace(
            enforce_eager=True,
            max_model_len=50_000,
            hf_config=SimpleNamespace(vocab_size=13088),
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )
    monkeypatch.setattr(model_module.vllm_envs, "VLLM_USE_V2_MODEL_RUNNER", True)

    with pytest.raises(ValueError, match="V2"):
        model_module.Nemotron3_5AsrForRNNT(vllm_config=runtime)


def test_init_rejects_vocab_size_drift_before_constructing_hf_modules() -> None:
    runtime = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            max_num_seqs=1,
            max_num_batched_tokens=50_000,
            max_num_encoder_input_tokens=50_000,
        ),
        model_config=SimpleNamespace(
            enforce_eager=True,
            max_model_len=50_000,
            hf_config=SimpleNamespace(vocab_size=4),
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )

    with pytest.raises(ValueError, match="vocab_size=13088"):
        model_module.Nemotron3_5AsrForRNNT(vllm_config=runtime)


def test_attention_free_adapter_has_no_vllm_kv_cache_groups() -> None:
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
    from vllm.v1.core.kv_cache_utils import get_kv_cache_groups

    model = _bare_model()
    runtime = SimpleNamespace(
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False)
    )

    assert not any(isinstance(module, AttentionLayerBase) for module in model.modules())
    assert get_kv_cache_groups(runtime, {}) == []


@pytest.mark.parametrize(
    ("attribute", "value", "match"),
    [
        (
            "scheduler_config",
            SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=50_000),
            "max_num_seqs",
        ),
        (
            "model_config",
            SimpleNamespace(
                enforce_eager=False, max_model_len=50_000, hf_config=SimpleNamespace()
            ),
            "eager",
        ),
        ("parallel_config", SimpleNamespace(tensor_parallel_size=2), "tensor_parallel"),
        (
            "scheduler_config",
            SimpleNamespace(max_num_seqs=1, max_num_batched_tokens=4),
            "max_num_batched_tokens",
        ),
        (
            "scheduler_config",
            SimpleNamespace(
                max_num_seqs=1,
                max_num_batched_tokens=50_000,
                max_num_encoder_input_tokens=1,
            ),
            "max_num_encoder_input_tokens",
        ),
        (
            "model_config",
            SimpleNamespace(
                enforce_eager=True, max_model_len=4, hf_config=SimpleNamespace()
            ),
            "max_model_len",
        ),
    ],
)
def test_init_rejects_invalid_vllm_runtime_config(
    attribute: str, value: object, match: str
) -> None:
    values = {
        "scheduler_config": SimpleNamespace(
            max_num_seqs=1,
            max_num_batched_tokens=50_000,
            max_num_encoder_input_tokens=50_000,
        ),
        "model_config": SimpleNamespace(
            enforce_eager=True, max_model_len=50_000, hf_config=SimpleNamespace()
        ),
        "parallel_config": SimpleNamespace(tensor_parallel_size=1),
    }
    values[attribute] = value

    with pytest.raises(ValueError, match=match):
        model_module.Nemotron3_5AsrForRNNT(vllm_config=SimpleNamespace(**values))
