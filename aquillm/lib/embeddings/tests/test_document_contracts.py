from dataclasses import replace

import pytest

from lib.embeddings import document_contracts as contracts


def profile(**changes):
    engine = dict(
        weights_revision="weights-v1",
        tokenizer_revision="tok-v1",
        compute_precision="fp16",
        quantization_algorithm="none",
        quantization_revision="none",
        pooling_policy="mean",
        query_template_sha256=contracts.digest("{text}"),
        document_template_sha256=contracts.digest("{text}"),
        image_processor_revision="verified-png-v1",
        dimension_policy="native-1024-v1",
        normalization="none",
        engine_binary_sha256="a" * 64,
        engine_config_sha256="b" * 64,
        response_route="openai-embeddings",
    )
    engine.update(changes)
    return contracts.declare_space(
        engine,
        provider="local-openai",
        model="synthetic",
        query_template="{text}",
        document_template="{text}",
        permitted_modalities=("text", "image", "caption_only"),
        max_input_tokens=8192,
        max_batch_tokens=32768,
    )


def test_precision_quantization_pooling_and_templates_are_space_identity():
    base = profile()
    assert (
        len(
            {
                base.space_key,
                profile(compute_precision="bf16").space_key,
                profile(
                    quantization_algorithm="4bit", quantization_revision="q1"
                ).space_key,
                profile(pooling_policy="last").space_key,
            }
        )
        == 4
    )
    with pytest.raises(ValueError):
        contracts.validate_space(replace(base, document_prep_key="foreign"))


def test_missing_engine_evidence_is_not_none():
    with pytest.raises(ValueError):
        profile(quantization_algorithm=None)


def test_build_budget_validates_relative_timeout_and_keeps_120_seconds():
    clock = [0.0]
    budget = contracts.build_budget(120000, clock=lambda: clock[0])
    clock[0] = 16
    assert budget.remaining_seconds() == 104
    for invalid in (True, 0, -1, 120001, float("nan")):
        with pytest.raises(ValueError):
            contracts.build_budget(invalid)
