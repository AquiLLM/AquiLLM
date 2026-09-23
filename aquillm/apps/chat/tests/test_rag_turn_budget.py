"""Pilot retrieval budget and opt-in configuration contracts."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from lib.retrieval.turn_budget import TurnBudget, TurnLimits
from apps.chat.services.rag_preservation_config import preservation_config


@pytest.fixture(autouse=True)
def clear_preservation_environment(monkeypatch):
    for name in (
        "RAG_RERANK_TEXT_MODE", "RAG_RERANK_SHADOW_SCORING_ENABLED",
        "RAG_EVIDENCE_TEXT_MODE", "RAG_DOCUMENT_CAPACITY_MODE",
        "RAG_DOCUMENT_HARD_CAP", "RAG_FOLLOWUP_EVIDENCE_ENABLED",
        "RAG_ITERATIVE_RETRIEVAL_ENABLED",
    ):
        monkeypatch.delenv(name, raising=False)


def test_pair_allowances_are_shared_and_final_pairs_reserved():
    budget = TurnBudget(TurnLimits(), clock=lambda: 0.0)
    assert budget.reserve_pairs(90, phase="acquisition")
    assert not budget.reserve_pairs(1, phase="acquisition")
    assert budget.reserve_pairs(45, phase="final")
    assert not budget.reserve_pairs(1, phase="final")
    budget.close("cancelled")
    assert not budget.can_publish()


def test_actions_charge_retries_and_repeated_signatures_cannot_expand():
    budget = TurnBudget(TurnLimits(), clock=lambda: 0.0)
    assert budget.reserve_action("search:a")
    assert not budget.reserve_action("search:a")
    assert budget.reserve_action("search:b")
    assert budget.reserve_action("search:c")
    assert not budget.reserve_action("search:d")
    assert budget.reserve_pairs(89, phase="acquisition")
    assert budget.reserve_pairs(1, phase="acquisition")  # retry
    assert not budget.reserve_pairs(1, phase="acquisition")


def test_unique_sources_include_revision_and_are_cumulative():
    budget = TurnBudget(TurnLimits(), clock=lambda: 0.0)
    assert all(budget.admit_source((i, "rev-a")) for i in range(45))
    assert budget.admit_source((0, "rev-a"))
    assert not budget.admit_source((0, "rev-b"))


def test_materialization_and_tokenization_codepoint_limits_are_separate():
    budget = TurnBudget(TurnLimits(), clock=lambda: 0.0)
    assert budget.reserve_text(250_000, kind="materialized")
    assert not budget.reserve_text(1, kind="materialized")
    assert budget.reserve_text(1_000_000, kind="tokenized")
    assert not budget.reserve_text(1, kind="tokenized")


def test_deadline_closes_late_publication_and_optional_work_has_completion_reserve():
    now = [0.0]
    budget = TurnBudget(TurnLimits(), clock=lambda: now[0])
    assert budget.can_start_optional(8_000, 4_000)  # 8 + 4 + 3 final <= 15 seconds
    assert not budget.can_start_optional(8_001, 4_000)
    now[0] = 14.9
    assert budget.remaining_ms() <= 100
    now[0] = 15.0
    assert budget.remaining_ms() == 0
    assert not budget.reserve_action("late")
    assert not budget.can_publish()
    assert budget.stop_reason == "deadline"


def test_closed_ledger_rejects_late_reservations_but_keeps_reason():
    budget = TurnBudget(TurnLimits(), clock=lambda: 0.0)
    budget.close("cancelled")
    assert budget.stop_reason == "cancelled"
    assert not budget.reserve_action("late")
    assert not budget.admit_source((1, "r"))
    assert not budget.reserve_pairs(1, phase="final")
    assert not budget.reserve_text(1, kind="materialized")


def test_concurrent_reservations_cannot_exceed_allowance():
    budget = TurnBudget(TurnLimits(), clock=lambda: 0.0)
    with ThreadPoolExecutor(max_workers=16) as pool:
        granted = list(pool.map(lambda _: budget.reserve_pairs(1, phase="acquisition"), range(200)))
    assert sum(granted) == 90


def test_nested_limits_only_narrow_parent_allowances():
    parent = TurnLimits(actions=2, acquisition_pairs=60, final_pairs=20)
    nested = TurnLimits().clamped_to(parent)
    assert (nested.actions, nested.acquisition_pairs, nested.final_pairs) == (2, 60, 20)
    with pytest.raises(ValueError):
        TurnLimits(acquisition_pairs=100, final_pairs=45)


@pytest.mark.parametrize("override", [
    {"actions": 4}, {"unique_sources": 46},
    {"materialized_codepoints": 250_001}, {"tokenized_codepoints": 1_000_001},
    {"acquisition_pairs": 91}, {"final_pairs": 46},
    {"in_flight_pairs": 7}, {"planner_calls": 3},
    {"retrieval_ms": 15_001}, {"final_scoring_ms": 3_001},
    {"actions": True}, {"actions": -1},
])
def test_pilot_limits_cannot_be_raised_or_malformed(override):
    with pytest.raises(ValueError):
        TurnLimits(**override)


@pytest.mark.parametrize("method, kwargs", [
    ("reserve_pairs", {"count": 0, "phase": "acquisition"}),
    ("reserve_text", {"count": -1, "kind": "materialized"}),
    ("reserve_pairs", {"count": 1, "phase": "other"}),
    ("reserve_text", {"count": 1, "kind": "other"}),
])
def test_invalid_reservations_raise(method, kwargs):
    with pytest.raises(ValueError):
        getattr(TurnBudget(TurnLimits(), clock=lambda: 0.0), method)(**kwargs)


def test_default_configuration_is_off_and_rejects_contradictions(monkeypatch):
    for name in ("RAG_RERANK_TEXT_MODE", "RAG_EVIDENCE_TEXT_MODE", "RAG_DOCUMENT_CAPACITY_MODE",
                 "RAG_DOCUMENT_HARD_CAP", "RAG_FOLLOWUP_EVIDENCE_ENABLED", "RAG_ITERATIVE_RETRIEVAL_ENABLED"):
        monkeypatch.delenv(name, raising=False)
    assert preservation_config().active is False
    monkeypatch.setenv("RAG_FOLLOWUP_EVIDENCE_ENABLED", "1")
    assert preservation_config().error is not None
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "source")
    assert preservation_config().error is None
    monkeypatch.setenv("RAG_DOCUMENT_HARD_CAP", "-1")
    assert preservation_config().error is not None


@pytest.mark.parametrize("key,value", [
    ("RAG_RERANK_TEXT_MODE", "future"),
    ("RAG_EVIDENCE_TEXT_MODE", "yes"),
    ("RAG_DOCUMENT_CAPACITY_MODE", "many"),
    ("RAG_DOCUMENT_HARD_CAP", "nan"),
    ("RAG_FOLLOWUP_EVIDENCE_ENABLED", "maybe"),
    ("RAG_ITERATIVE_RETRIEVAL_ENABLED", "2"),
    ("RAG_RERANK_SHADOW_SCORING_ENABLED", "maybe"),
])
def test_invalid_opt_in_values_are_rejected(monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    assert preservation_config().error is not None


def test_shadow_scoring_needs_separate_opt_in(monkeypatch):
    monkeypatch.setenv("RAG_RERANK_TEXT_MODE", "shadow")
    assert preservation_config().shadow_scoring is False
    monkeypatch.setenv("RAG_RERANK_SHADOW_SCORING_ENABLED", "1")
    assert preservation_config().shadow_scoring is True
    monkeypatch.setenv("RAG_RERANK_TEXT_MODE", "windowed")
    assert preservation_config().error is not None


def test_preservation_requires_available_shared_selector(monkeypatch):
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "source")
    assert preservation_config(shared_selector_available=False).error is not None
    assert preservation_config(shared_selector_available=True).error is None


def test_budgeted_document_cap_uses_final_limit_and_explicit_ceiling(monkeypatch):
    monkeypatch.setenv("RAG_DOCUMENT_CAPACITY_MODE", "budgeted")
    assert preservation_config().document_cap(final_passage_limit=10, legacy_per_doc_limit=3) == 10
    monkeypatch.setenv("RAG_DOCUMENT_HARD_CAP", "5")
    assert preservation_config().document_cap(final_passage_limit=10, legacy_per_doc_limit=3) == 5
    monkeypatch.setenv("RAG_DOCUMENT_CAPACITY_MODE", "legacy")
    assert preservation_config().document_cap(final_passage_limit=10, legacy_per_doc_limit=3) == 3
