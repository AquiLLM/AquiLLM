"""Private eager source/row boundaries stage positive observations until handoff."""

from dataclasses import replace

import pytest

from apps.documents.models import TextChunk
from apps.documents.services import chunk_search_scoped_rows as materialization
from apps.documents.services.chunk_search_scope import FrozenRetrievalScope
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.source_loading import SourceRuntime, source_runtime_scope
from apps.documents.tests.test_source_metadata_closed_operation import captured
from lib.embeddings.tests.test_document_provider import transport as transport
from lib.retrieval.turn_budget import TurnBudget, TurnLimits

from .source_body_test_hooks import install_body_hook
from .test_scoped_source_consolidation import assert_unpublished, revoke

pytestmark = pytest.mark.django_db(transaction=True)


def test_runtime_replacement_during_database_entry_cannot_use_an_unheld_lock(
    transport, monkeypatch, settings
):
    _, scope, query, _ = captured(1, "none", transport, monkeypatch, settings)
    original = scope.runtime
    replacement = SourceRuntime(TurnBudget(TurnLimits()), original.authorization)
    replacement._frozen_retrieval_scope = replace(
        original._frozen_retrieval_scope, runtime=replacement
    )
    real_clock, changes = original.budget._clock, []

    def clock():
        if not changes:
            changes.append(True)
            scope.runtime = replacement
        return real_clock()

    monkeypatch.setattr(original.budget, "_clock", clock)
    try:
        with source_runtime_scope(original):
            with pytest.raises(SourceScopeUnavailable):
                materialization.scoped_rows(query, scope, lexical=True)
            assert changes == [True]
            assert_unpublished(scope)
            assert original.budget.sources_used == replacement.budget.sources_used == 0
    finally:
        original.budget.close("scoped_consolidation_finished")
        replacement.budget.close("scoped_consolidation_finished")


def test_budget_replacement_cannot_move_charges_to_a_new_ledger(
    transport, monkeypatch, settings
):
    _, scope, query, expected = captured(2, "none", transport, monkeypatch, settings)
    original, replacement = scope.runtime.budget, TurnBudget(TurnLimits())
    fetched = []

    def first(queryset, row):
        if queryset.model is TextChunk and row is not None:
            fetched.append(row.pk)
            scope.runtime.budget = replacement
        return row

    install_body_hook(monkeypatch, first)
    try:
        with source_runtime_scope(scope.runtime):
            with pytest.raises(SourceScopeUnavailable):
                materialization.scoped_rows(query, scope, lexical=True)
            assert fetched == [item[0] for item in expected]
            assert_unpublished(scope)
            assert original.sources_used == 2
            assert original.text_used["materialized"] == sum(
                len(item[2]) for item in expected
            )
            assert replacement.sources_used == 0
            assert replacement.text_used["materialized"] == 0
    finally:
        original.close("scoped_consolidation_finished")
        replacement.close("scoped_consolidation_finished")


def test_reentrant_publication_cannot_be_overwritten_by_outer_operation(
    transport, monkeypatch, settings
):
    _, scope, query, expected = captured(1, "none", transport, monkeypatch, settings)
    inner = []

    def first(queryset, row):
        if queryset.model is TextChunk and row is not None and not inner:
            inner.append(None)
            inner[0] = materialization.scoped_rows(query, scope, lexical=True)
        return row

    install_body_hook(monkeypatch, first)
    try:
        with source_runtime_scope(scope.runtime):
            with pytest.raises(SourceScopeUnavailable):
                materialization.scoped_rows(query, scope)
            assert tuple((r.pk, r.doc_id, r.content) for r in inner[0]) == expected
            assert set(scope.bodies) == scope.lexical_ids == {expected[0][0]}
            assert len(scope.runtime.cache) == 1
    finally:
        scope.runtime.budget.close("scoped_consolidation_finished")


@pytest.mark.parametrize("replace_header", [False, True])
def test_withdrawn_full_text_vote_survives_restoration_and_aborted_final_check(
    replace_header, transport, monkeypatch, settings
):
    from apps.documents.services.candidate_rank_fusion import fuse_ranked_channels_v2

    case, scope, query, expected = captured(1, "none", transport, monkeypatch, settings)
    doc = case.docs[0]
    real_current, withdrawn = materialization._current_rows, []
    try:
        with source_runtime_scope(scope.runtime):
            rows = materialization.scoped_rows(query, scope, lexical="full_text")
            assert tuple((r.pk, r.doc_id, r.content) for r in rows) == expected
            scope.fusion_v2 = fuse_ranked_channels_v2(
                {"full_text": ((rows[0].pk, 1),)}, limit=1, scope_key="a" * 64
            )
            assert scope.fusion_v2.candidates
            captured_fusion = scope.fusion_v2
            before_bodies, before_cache = (
                scope.bodies.copy(),
                scope.runtime.cache.copy(),
            )
            type(doc).objects.filter(pk=doc.pk).update(title="Changed projection")

            def current(*args, **kwargs):
                result = real_current(*args, **kwargs)
                assert result == ((), set())
                withdrawn.append(rows[0].pk)
                type(doc).objects.filter(pk=doc.pk).update(title=doc.title)
                if replace_header:
                    scope.authorization = replace(scope.authorization)
                else:
                    revoke(case)
                return result

            monkeypatch.setattr(materialization, "_current_rows", current)
            with pytest.raises((SourceScopeUnavailable, ValueError)):
                materialization.scoped_rows(query, scope, lexical="full_text")
            assert withdrawn == [rows[0].pk]
            assert scope.fts_invalid_ids == {rows[0].pk}
            if replace_header:
                assert scope.fusion_v2 is captured_fusion
            else:
                assert scope.fusion_v2.candidates == ()
            assert scope.bodies == before_bodies and scope.runtime.cache == before_cache
    finally:
        scope.runtime.budget.close("scoped_consolidation_finished")


def test_in_place_reentrant_binding_change_is_detected(
    transport, monkeypatch, settings
):
    case, scope, query, expected = captured(1, "none", transport, monkeypatch, settings)
    real_current = materialization._current_rows
    try:
        with source_runtime_scope(scope.runtime):
            rows = materialization.scoped_rows(query, scope, lexical="full_text")
            assert tuple((r.pk, r.doc_id, r.content) for r in rows) == expected
            binding = scope.fts_rows[rows[0].pk]
            assert type(binding) is dict

            def current(*args, **kwargs):
                result = real_current(*args, **kwargs)
                assert len(result[0]) == 1
                binding["nested_callback"] = {"changed": True}
                return result

            monkeypatch.setattr(materialization, "_current_rows", current)
            with pytest.raises(SourceScopeUnavailable):
                materialization.scoped_rows(query, scope, lexical=True)
            assert scope.lexical_ids == set()
            assert scope.fts_rows[rows[0].pk] is binding
            assert binding["nested_callback"] == {"changed": True}
    finally:
        scope.runtime.budget.close("scoped_consolidation_finished")


@pytest.mark.parametrize("empty", [False, True])
def test_final_deadline_failure_discards_positive_observations(
    empty, transport, monkeypatch, settings
):
    _, scope, query, _ = captured(1, "none", transport, monkeypatch, settings)
    real_validate, checks = FrozenRetrievalScope.validate, []

    def validate(current):
        checks.append(current)
        if len(checks) == 2:
            scope.runtime.budget.close("ordered_final_deadline")
        return real_validate(current)

    monkeypatch.setattr(FrozenRetrievalScope, "validate", validate)
    with source_runtime_scope(scope.runtime):
        with pytest.raises(SourceScopeUnavailable):
            materialization.scoped_rows(
                query.none() if empty else query, scope, lexical=True
            )
        assert len(checks) == 2
        assert_unpublished(scope)


def test_equivalent_but_uncaptured_authority_uses_guarded_fallback(
    transport, monkeypatch, settings
):
    _, scope, query, expected = captured(1, "none", transport, monkeypatch, settings)
    scope = replace(scope, authorization=replace(scope.authorization))
    real_source, fallback = materialization.source_query_rows, []

    def source(*args, **kwargs):
        rows = real_source(*args, **kwargs)
        fallback.append(rows)
        return rows

    monkeypatch.setattr(materialization, "source_query_rows", source)
    try:
        with source_runtime_scope(scope.runtime):
            rows = materialization.scoped_rows(query, scope)
            assert tuple((r.pk, r.doc_id, r.content) for r in rows) == expected
            assert fallback == [rows]
    finally:
        scope.runtime.budget.close("scoped_consolidation_finished")
