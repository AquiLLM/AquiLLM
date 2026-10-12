"""Atomic, bounded admission of authenticated immutable index records only."""

from asyncio import CancelledError
from copy import deepcopy

import pytest

from apps.knowledge_graph.projection.frontier_index_contracts_v2 import decode_record
from apps.knowledge_graph.projection.frontier_reads_v2 import QUERIES
from apps.knowledge_graph.retrieval.topology.failures import TopologyLoadError
from apps.knowledge_graph.retrieval.topology.frontier_gateway_contracts_v2 import FrontierCompletedBudgetFailureV2
from apps.knowledge_graph.tests.frontier_scope_batch_support import ScopeBatchFixture, batch_indices


def prepared():
    fixture = ScopeBatchFixture()
    reads, generation = fixture.reader()
    keys = tuple(sorted(fixture.driver.index)[:32])
    assert len(keys) == 32
    return fixture, reads, generation, keys


def test_batch_is_one_physical_read_with_exact_accounting_and_defensive_reuse():
    fixture, reads, generation, keys = prepared()
    before = fixture.budget.queries, fixture.budget.source_rows
    result = batch_indices(reads, generation, keys)
    assert result == {k: decode_record(fixture.driver.index[k], expected_key=k, generation=generation) for k in keys}
    assert fixture.budget.queries == before[0] + 1
    assert fixture.budget.source_rows == before[1] + 32
    assert fixture.driver.bounds[-1][1] == 64
    result[keys[0]][1].clear()
    again = batch_indices(reads, generation, tuple(reversed(keys)))
    assert again[keys[0]][1]
    assert fixture.budget.queries == before[0] + 1
    assert fixture.budget._reserved_rows == 0


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "foreign_key", "foreign_generation", "wrong_hash", "wrong_kind"])
def test_invalid_batch_never_admits_a_valid_prefix_to_immutable_cache(mutation):
    fixture, reads, generation, keys = prepared()

    def transform(query, params, rows):
        if query != QUERIES.get("index_batch"):
            return rows
        if mutation == "missing":
            return rows[:-1]
        if mutation == "duplicate":
            return rows + (deepcopy(rows[-1]),)
        rows = list(rows)
        if mutation == "foreign_key":
            rows[-1]["lookup_key"] = "0" * 64
        else:
            field = {"foreign_generation": "generation_key", "wrong_hash": "index_key", "wrong_kind": "kind"}[mutation]
            rows[-1]["record"][field] = "0" * 64 if field != "kind" else "unregistered"
        return tuple(rows)

    fixture.driver.transform = transform
    before_rows = fixture.budget.source_rows
    with pytest.raises(TopologyLoadError):
        batch_indices(reads, generation, keys)
    assert fixture.budget.source_rows > before_rows
    assert fixture.budget.source_rows == fixture.driver.scanned
    assert fixture.budget._reserved_rows == 0
    fixture.driver.transform = None
    before = fixture.budget.queries
    assert set(batch_indices(reads, generation, keys)) == set(keys)
    assert fixture.budget.queries == before + 1
    assert fixture.driver.calls[-1][1]["keys_csv"] == ",".join(keys)


def test_response_order_is_irrelevant_but_per_key_multiplicity_is_not():
    fixture, reads, generation, keys = prepared()
    fixture.driver.transform = lambda q, p, rows: tuple(reversed(rows))
    assert set(batch_indices(reads, generation, keys)) == set(keys)
    assert fixture.budget.source_rows == fixture.driver.scanned


@pytest.mark.parametrize("closed", ["deadline", "turn", "uncertain"])
def test_cached_batch_still_checks_original_lifetime(monkeypatch, closed):
    fixture, reads, generation, keys = prepared()
    batch_indices(reads, generation, keys)
    before = len(fixture.driver.calls)
    if closed == "deadline":
        monkeypatch.setattr(reads, "deadline", fixture.budget.clock() - 1)
    elif closed == "turn":
        fixture.budget.turn_budget.close("cancelled")
    else:
        fixture.budget.stop_uncertain()
    with pytest.raises(CancelledError if closed == "turn" else TimeoutError):
        batch_indices(reads, generation, keys)
    assert len(fixture.driver.calls) == before


def test_completed_wire_refusal_keeps_exact_physical_row_charge():
    fixture, reads, generation, keys = prepared()
    original = fixture.driver.execute_read

    def too_large(*args, **kwargs):
        rows = original(*args, **kwargs)
        raise FrontierCompletedBudgetFailureV2(len(rows))

    fixture.driver.execute_read = too_large
    before = fixture.budget.queries, fixture.budget.source_rows
    with pytest.raises(FrontierCompletedBudgetFailureV2):
        batch_indices(reads, generation, keys)
    assert fixture.budget.queries == before[0] + 1
    assert fixture.budget.source_rows == before[1] + 32
    assert fixture.budget._reserved_rows == 0 and not fixture.budget._uncertain
