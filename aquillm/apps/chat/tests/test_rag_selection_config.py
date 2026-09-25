"""Safe, bounded rollout configuration for adaptive evidence selection."""

import pytest

from apps.chat.services.rag_config import (
    direct_rag_top_k,
    evidence_selection_config,
    evidence_token_budget,
    max_snippets_per_doc,
)

KEYS = (
    "RAG_EVIDENCE_SELECTION_MODE",
    "RAG_EVIDENCE_SELECTION_SCORE_TIMEOUT_MS",
    "RAG_EVIDENCE_SELECTION_SHADOW_SCORING",
)


@pytest.fixture(autouse=True)
def clean_selection_environment(monkeypatch):
    for key in KEYS:
        monkeypatch.delenv(key, raising=False)


def test_defaults_preserve_legacy_without_added_scoring():
    config = evidence_selection_config()
    assert (config.mode, config.score_timeout_ms, config.shadow_scoring) == (
        "legacy",
        3000,
        False,
    )
    assert config.error is None


@pytest.mark.parametrize("mode", ["legacy", "shadow", "adaptive"])
@pytest.mark.parametrize("timeout", [100, 1500, 3000])
def test_supported_modes_and_bounded_timeouts(monkeypatch, mode, timeout):
    monkeypatch.setenv(KEYS[0], mode)
    monkeypatch.setenv(KEYS[1], str(timeout))
    config = evidence_selection_config()
    assert (config.mode, config.score_timeout_ms) == (mode, timeout)
    assert config.error is None


@pytest.mark.parametrize(
    ("key", "value"),
    [
        (KEYS[0], "unknown"),
        (KEYS[1], "99"),
        (KEYS[1], "3001"),
        (KEYS[1], "nan"),
        (KEYS[2], "maybe"),
    ],
)
def test_invalid_configuration_falls_back_as_one_unit(monkeypatch, key, value):
    monkeypatch.setenv(KEYS[0], "adaptive")
    monkeypatch.setenv(key, value)
    config = evidence_selection_config()
    assert (config.mode, config.score_timeout_ms, config.shadow_scoring) == (
        "legacy",
        3000,
        False,
    )
    assert config.error == "invalid_evidence_selection_configuration"


def test_shadow_extra_scoring_requires_explicit_flag(monkeypatch):
    monkeypatch.setenv(KEYS[0], "shadow")
    assert evidence_selection_config().shadow_scoring is False
    monkeypatch.setenv(KEYS[2], "1")
    assert evidence_selection_config().shadow_scoring is True


def test_adaptive_mode_does_not_override_existing_evidence_limits(monkeypatch):
    monkeypatch.setenv(KEYS[0], "adaptive")
    monkeypatch.setenv("RAG_DIRECT_TOP_K", "2")
    monkeypatch.setenv("RAG_MAX_SNIPPETS_PER_DOC", "1")
    monkeypatch.setenv("RAG_EVIDENCE_TOKEN_BUDGET", "256")
    assert evidence_selection_config().mode == "adaptive"
    assert (direct_rag_top_k(), max_snippets_per_doc(), evidence_token_budget()) == (
        2,
        1,
        256,
    )
