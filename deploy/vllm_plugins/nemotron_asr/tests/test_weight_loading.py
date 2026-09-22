"""Strict checkpoint-to-wrapper weight mapping tests."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from aquillm_vllm_nemotron_asr import model as model_module  # noqa: E402

MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"
MODEL_REVISION = "f3d333391852ba876df169dcc9ba902d25b6ab0b"
MODEL_WEIGHT_SHA256 = "9eebdd6590289cb3030f310858f3df93256600a800a3e8200c5993d5f967e174"


def _weight_model() -> model_module.Nemotron3_5AsrForRNNT:
    result = object.__new__(model_module.Nemotron3_5AsrForRNNT)
    torch.nn.Module.__init__(result)
    for prefix in (
        "encoder",
        "decoder",
        "encoder_projector",
        "prompt_projector",
        "joint",
    ):
        setattr(result, prefix, torch.nn.Linear(2, 2, bias=False))
    return result


def _weights(model: torch.nn.Module) -> list[tuple[str, torch.Tensor]]:
    return [
        (name, torch.full_like(parameter, index + 1))
        for index, (name, parameter) in enumerate(model.named_parameters())
    ]


def test_load_weights_requires_an_exact_complete_identity_prefix_mapping() -> None:
    model = _weight_model()
    weights = _weights(model)

    loaded = model.load_weights(iter(weights))

    assert loaded == {name for name, _ in weights}
    assert all(not name.startswith("model.") for name in loaded)
    assert all(
        parameter.eq(index + 1).all()
        for index, parameter in enumerate(model.parameters())
    )


@pytest.mark.parametrize(
    ("weights", "match"),
    [
        (
            lambda model: _weights(model) + [("unknown.weight", torch.ones((2, 2)))],
            "unknown",
        ),
        (
            lambda model: (
                [_weights(model)[0], _weights(model)[0]] + _weights(model)[1:]
            ),
            "duplicate",
        ),
        (lambda model: _weights(model)[:-1], "missing"),
    ],
)
def test_load_weights_rejects_unknown_duplicate_and_missing_names(
    weights: object, match: str
) -> None:
    model = _weight_model()

    with pytest.raises(ValueError, match=match):
        model.load_weights(iter(weights(model)))  # type: ignore[operator]


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("ASR_FULL_PARITY_PHASE", "").strip() != "plugin",
    reason="set ASR_FULL_PARITY_PHASE=plugin",
)
def test_full_checkpoint_safetensors_inventory_is_exact() -> None:
    """Prove the cached immutable checkpoint has the reviewed tensor inventory."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    weight_path = hf_hub_download(
        repo_id=MODEL_ID,
        filename="model.safetensors",
        revision=MODEL_REVISION,
        cache_dir=os.environ.get(
            "ASR_HF_CACHE_DIR", str(Path(os.environ["HF_HOME"]) / "hub")
        ),
        local_files_only=True,
    )
    digest = hashlib.sha256()
    with open(weight_path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    assert os.path.getsize(weight_path) == 2_552_062_944
    assert digest.hexdigest() == MODEL_WEIGHT_SHA256

    with safe_open(weight_path, framework="pt", device="cpu") as checkpoint:
        names = list(checkpoint.keys())
        tensor_count = len(names)
        parameter_count = sum(checkpoint.get_tensor(name).numel() for name in names)

    assert tensor_count == 655
    assert parameter_count == 637_997_088
    assert {
        prefix: sum(name.startswith(f"{prefix}.") for name in names)
        for prefix in (
            "encoder",
            "decoder",
            "prompt_projector",
            "encoder_projector",
            "joint",
        )
    } == {
        "encoder": 636,
        "decoder": 11,
        "prompt_projector": 4,
        "encoder_projector": 2,
        "joint": 2,
    }
