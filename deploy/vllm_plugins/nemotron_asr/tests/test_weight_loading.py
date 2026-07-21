"""Strict checkpoint-to-wrapper weight mapping tests."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from aquillm_vllm_nemotron_asr import model as model_module  # noqa: E402


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
