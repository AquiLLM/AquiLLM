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


@pytest.mark.parametrize("value,enabled", [(None, False), ("off", False), (" YES ", True)])
def test_support_hint_snapshot_uses_the_consumed_gate(monkeypatch, value, enabled):
    if value is None:
        monkeypatch.delenv("RAG_SELECTION_SUPPORT_HINTS_ENABLED", raising=False)
    else:
        monkeypatch.setenv("RAG_SELECTION_SUPPORT_HINTS_ENABLED", value)
    assert comparison_controls().get("selection_support_hints") == {
        "enabled": enabled, "policy": "assessment-support-hints-v1",
    }


def test_invalid_support_hint_control_cannot_be_snapshotted(monkeypatch):
    from django.core.exceptions import ImproperlyConfigured

    monkeypatch.setenv("RAG_SELECTION_SUPPORT_HINTS_ENABLED", "sometimes")
    with pytest.raises(ImproperlyConfigured, match="RAG_SELECTION_SUPPORT_HINTS_ENABLED"):
        comparison_controls()


@pytest.mark.parametrize("change", ["enabled", "policy"])
def test_support_hint_control_change_invalidates_review_and_matched_arm(monkeypatch, change):
    from copy import deepcopy

    from apps.chat.evals.evidence_quality_eval import MODES, digest, join_arms
    from apps.chat.evals.evidence_review_subject import review_subject, validated_review
    from apps.chat.tests.test_evidence_review_subject import reviewed_case
    from apps.chat.services import rag_selection_hints

    monkeypatch.setenv("RAG_SELECTION_SUPPORT_HINTS_ENABLED", "0")
    _, row, review = reviewed_case()
    row["comparison_controls"] = comparison_controls()
    row["snapshot"]["configuration"] = digest(row["comparison_controls"])
    review["subject"] = review_subject(row)
    assert validated_review(row, review) is review
    arms = {mode: [{**deepcopy(row), "mode": mode}] for mode in MODES}
    assert len(join_arms(arms)) == 1
    if change == "enabled":
        monkeypatch.setenv("RAG_SELECTION_SUPPORT_HINTS_ENABLED", "1")
    else:
        monkeypatch.setattr(rag_selection_hints, "POLICY", "changed-hint-policy")
    changed = arms["combined"][0]
    changed["comparison_controls"] = comparison_controls()
    changed["snapshot"]["configuration"] = digest(changed["comparison_controls"])
    assert validated_review(changed, review) is None
    with pytest.raises(ValueError, match="snapshot drift"):
        join_arms(arms)
