"""Requested arm labels must describe the effective production configuration."""

import pytest

from apps.chat.evals.evidence_quality_live import mode_environment
from apps.chat.evals.evidence_quality_runtime import comparison_controls


def test_invalid_inherited_shadow_rejects_live_arm_and_restores_environment(
    monkeypatch,
):
    monkeypatch.setenv("RAG_EVIDENCE_SELECTION_SHADOW_SCORING", "invalid")
    monkeypatch.setenv("RAG_EVIDENCE_SELECTION_MODE", "shadow")
    with pytest.raises(ValueError, match="effective"):
        with mode_environment("combined"):
            pytest.fail("invalid effective arm reached execution")
    import os

    assert os.environ["RAG_EVIDENCE_SELECTION_MODE"] == "shadow"


@pytest.mark.parametrize(
    "setting,value",
    [
        ("RAG_EVIDENCE_SELECTION_SCORE_TIMEOUT_MS", "1200"),
        ("APP_RERANK_PROVIDER", "different"),
        ("APP_RERANK_TIMEOUT_SECONDS", "7"),
        ("APP_RERANK_SCORE_CONCURRENCY", "2"),
    ],
)
def test_consumed_controls_change_snapshot(monkeypatch, setting, value):
    before = comparison_controls()
    monkeypatch.setenv(setting, value)
    assert comparison_controls() != before


def test_outer_limit_uses_consumed_constant_not_later_environment(monkeypatch):
    from apps.chat.consumers import chat as utils

    before = comparison_controls()
    monkeypatch.setattr(utils, "CHAT_MAX_FUNC_CALLS", utils.CHAT_MAX_FUNC_CALLS + 1)
    assert comparison_controls() != before


def test_four_treatments_share_invariant_controls():
    from apps.chat.evals.evidence_effective_config import resolved_treatment

    controls = []
    for mode in ("baseline", "selection", "preservation", "combined"):
        with mode_environment(mode):
            assert resolved_treatment(mode)["mode"] == mode
            controls.append(comparison_controls())
    assert all(c == controls[0] for c in controls)
