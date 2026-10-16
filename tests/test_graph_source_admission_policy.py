"""Deterministic admission operates on references, never an alternate ledger."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "aquillm"))

from apps.documents.services.graph_source_admission_policy import (
    allocation,
    charged_sources,
    fair_prefix,
)


def test_existing_twenty_graph_cap_derives_twenty_five_baseline_share():
    assert allocation(45, 20, 3) == (25, 20)
    assert allocation(7, 20, 3) == (3, 4)
    assert allocation(2, 20, 3) == (2, 0)
    assert allocation(45, 5, 3) == (40, 5)


def test_dense_cannot_monopolize_initial_body_admission():
    channels = {"dense": tuple(range(1, 46)), "exact": (101, 102),
                "trigram": (201, 202), "full_text": (301, 302)}
    assert fair_prefix(channels, 8) == (1, 101, 201, 301, 2, 102, 202, 302)


def test_overlapping_and_exhausted_channels_do_not_waste_slots():
    assert fair_prefix({"dense": (1, 2, 3), "exact": (1, 4),
                        "trigram": (4, 5)}, 5) == (1, 4, 5, 2, 3)


def test_only_explicit_protected_exact_gets_first_admission():
    channels = {"dense": (1, 2), "exact": (3, 4)}
    assert fair_prefix(channels, 2) == (1, 3)
    assert fair_prefix(channels, 2, protected=(4,)) == (4, 1)


@pytest.mark.parametrize("remaining,cap,channels", [(-1, 20, 3), (46, 20, 3),
                                                  (45, 21, 3), (45, 20, 5)])
def test_invalid_plan_cannot_raise_existing_limits(remaining, cap, channels):
    with pytest.raises(ValueError):
        allocation(remaining, cap, channels)


def test_reading_original_source_ledger_neither_reserves_nor_refunds():
    from lib.retrieval.turn_budget import TurnBudget, TurnLimits

    budget = TurnBudget(TurnLimits())
    assert budget.admit_source((7, "revision"))
    assert charged_sources(budget) == frozenset({(7, "revision")})
    assert budget.sources_used == 1
    assert budget.admit_source((7, "revision"))
    assert budget.sources_used == 1
