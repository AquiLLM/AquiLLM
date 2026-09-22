from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


def test_strict_embedding_adapter_rejects_raw_dimension_mismatch(monkeypatch):
    from aquillm import utils

    monkeypatch.setenv("APP_EMBED_MODEL_REVISION", "rev")
    monkeypatch.setattr(
        utils,
        "get_local_embed_config",
        lambda: ("http://local", "key", "embed-model"),
    )
    monkeypatch.setattr(utils, "get_target_dims", lambda: 1024)
    monkeypatch.setattr(
        utils,
        "get_strict_indexed_embeddings_via_local_openai",
        lambda _queries: [(0, [1.0, 2.0, 3.0])],
    )

    signature = utils.strict_index_embedding_signature()
    with pytest.raises(RuntimeError, match="invalid vector|1024"):
        utils.get_strict_index_embeddings(["Atlas"], expected_model_signature=signature)


def test_strict_local_embedding_adapter_rejects_served_model_drift(monkeypatch):
    from lib.embeddings import local

    requests = []

    class Embeddings:
        def create(self, **kwargs):
            requests.append(kwargs)
            return SimpleNamespace(
                model="different-model",
                data=[SimpleNamespace(index=0, embedding=[0.0] * 1024)],
            )

    monkeypatch.setattr(
        local,
        "get_local_embed_config",
        lambda: ("https://embeddings.example.test/v1", "secret", "embed-model"),
    )
    monkeypatch.setattr(
        local,
        "_get_local_openai_client",
        lambda _base_url, _api_key: SimpleNamespace(embeddings=Embeddings()),
    )
    monkeypatch.setattr(local, "_dims_kwargs", lambda: {})

    with pytest.raises(RuntimeError, match="model|identity"):
        local.get_strict_indexed_embeddings_via_local_openai(["Atlas"])
    assert requests == [
        {"model": "embed-model", "input": ["Atlas"], "dimensions": 1024}
    ]


def test_strict_embedding_signature_requires_an_immutable_model_revision(
    monkeypatch,
):
    from aquillm import utils

    monkeypatch.delenv("APP_EMBED_MODEL_REVISION", raising=False)
    monkeypatch.setattr(
        utils,
        "get_local_embed_config",
        lambda: ("http://local", "key", "embed-model"),
    )
    monkeypatch.setattr(utils, "get_target_dims", lambda: 1024)

    with pytest.raises(RuntimeError, match="revision|digest|immutable"):
        utils.strict_index_embedding_signature()


def test_strict_embedding_signature_binds_normalized_provider_endpoint(monkeypatch):
    from aquillm import utils

    monkeypatch.setenv("APP_EMBED_MODEL_REVISION", "rev")
    monkeypatch.setattr(utils, "get_target_dims", lambda: 1024)
    monkeypatch.setattr(
        utils,
        "get_local_embed_config",
        lambda: ("HTTPS://Embeddings.Example.test/v1/", "secret", "embed-model"),
    )
    normalized = utils.strict_index_embedding_signature()
    monkeypatch.setattr(
        utils,
        "get_local_embed_config",
        lambda: ("https://embeddings.example.test/v1", "other-secret", "embed-model"),
    )
    assert utils.strict_index_embedding_signature() == normalized

    monkeypatch.setattr(
        utils,
        "get_local_embed_config",
        lambda: ("https://other.example.test/v1", "secret", "embed-model"),
    )
    drifted = utils.strict_index_embedding_signature()
    assert drifted != normalized
    with pytest.raises(RuntimeError, match="signature drift"):
        utils.get_strict_index_embeddings(
            ["Atlas"], expected_model_signature=normalized
        )


def test_embedding_revision_is_documented_and_passed_fail_closed_to_compose():
    repository = Path(__file__).resolve().parents[4]
    env_example = (repository / ".env.example").read_text(encoding="utf-8")
    runbook = (
        repository / "docs/documents/operations/knowledge-graph-overlay-runbook.md"
    ).read_text(encoding="utf-8")

    assert "APP_EMBED_MODEL_REVISION=" in env_example
    assert "immutable" in env_example.lower()
    assert "APP_EMBED_ALLOW_DIMENSIONS_OVERRIDE=0" in env_example
    assert "provider-side dimensions=1024" in env_example.lower()
    assert "APP_EMBED_MODEL_REVISION" in runbook
    assert "fail" in runbook.lower()
    for compose_path in (
        "deploy/compose/base.yml",
        "deploy/compose/development.yml",
        "deploy/compose/production.yml",
        "deploy/compose/no_gpu_dev.yml",
    ):
        compose = (repository / compose_path).read_text(encoding="utf-8")
        assert "APP_EMBED_MODEL_REVISION" in compose
        assert "${APP_EMBED_MODEL_REVISION:-}" in compose
    no_gpu = (repository / "deploy/compose/no_gpu_dev.yml").read_text(encoding="utf-8")
    assert "APP_EMBED_ALLOW_DIMENSIONS_OVERRIDE: 1" in no_gpu
    for compose_path in (
        "deploy/compose/base.yml",
        "deploy/compose/development.yml",
        "deploy/compose/production.yml",
    ):
        compose = (repository / compose_path).read_text(encoding="utf-8")
        assert "VLLM_REVISION=${APP_EMBED_MODEL_REVISION:-}" in compose
        assert "VLLM_MODEL=${APP_EMBED_MODEL:-" in compose
        assert "VLLM_SERVED_MODEL_NAME=${APP_EMBED_MODEL:-" in compose
        assert "VLLM_TOKENIZER=${APP_EMBED_MODEL:-" in compose
