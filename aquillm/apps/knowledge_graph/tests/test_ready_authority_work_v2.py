"""Fresh whole authority is bounded during work, not only after late return."""

import pytest
from django.db import connection
from django.test.utils import CaptureQueriesContext

from apps.documents.services.source_deadline import source_preparation_scope
from apps.documents.services.source_loading import SourcePreparationLimited
from apps.knowledge_graph.models import CollectionArtifactInput, CollectionGraphMembershipState, CollectionGraphProjection, GraphArtifact
from apps.knowledge_graph.projection import memberships
from apps.knowledge_graph.retrieval import ready_scope_repository_v2 as repository
from apps.knowledge_graph.tests.ready_subset_support import ScopeFixture

pytestmark = pytest.mark.django_db(transaction=True)


def configured(monkeypatch):
    f = ScopeFixture(monkeypatch)
    f.persist_ready("A")
    f.persist_ready("B")
    return f


def authorities(f):
    return repository.load_authorities_v2(
        authorization=f.authorization, settings=f.settings,
        representation_snapshot=f.snapshot, codec=f.codec,
    )


def test_two_collections_read_fresh_metadata_in_five_bounded_queries(monkeypatch):
    f = configured(monkeypatch)
    models = (CollectionArtifactInput, CollectionGraphMembershipState, CollectionGraphProjection, GraphArtifact)
    with CaptureQueriesContext(connection) as queries:
        result = authorities(f)
    assert len(result) == 2
    tables = {model._meta.db_table for model in models}
    projection_table = CollectionGraphProjection._meta.db_table
    statements = [q["sql"] for q in queries.captured_queries
                  if q["sql"].lstrip().upper().startswith(("SELECT", "DECLARE"))
                  and any('FROM "' + table + '"' in q["sql"] for table in tables)
                  and (('FROM "' + projection_table + '"') not in q["sql"]
                       or '"schema_version"' in q["sql"])]
    assert len(statements) == 5, statements
    binding_reads = [q["sql"] for q in queries.captured_queries
                     if ('FROM "' + projection_table + '"') in q["sql"]
                     and '"schema_version"' not in q["sql"]]
    assert len(binding_reads) == 2  # Existing independent binding reads stay fresh.


def test_authority_expiry_aborts_before_next_collection_and_returns_no_proof(monkeypatch):
    f = configured(monkeypatch)
    now, calls, returned = [0.0], [], []
    original = repository.load_automatic_membership_assignments

    def expire(**kwargs):
        calls.append(kwargs["collection_ids"])
        result = original(**kwargs)
        now[0] = 2.0
        return result

    monkeypatch.setattr(repository, "load_automatic_membership_assignments", expire)
    with pytest.raises((TimeoutError, SourcePreparationLimited)):
        with source_preparation_scope(1.0, lambda: now[0]):
            returned.append(authorities(f))
    assert returned == []
    assert calls == [(f.documents["A"].collection_id,)]


def test_membership_cpu_expiry_aborts_within_row_batch_without_returned_assignments(monkeypatch):
    from apps.knowledge_graph.projection.identifiers import HmacSha256ProjectionIdentifierCodec

    now, calls, returned = [0.0], [], []
    real = HmacSha256ProjectionIdentifierCodec(b"synthetic", key_version="key-v1")

    class SlowCodec:
        def encode(self, *args, **kwargs):
            calls.append(True)
            result = real.encode(*args, **kwargs)
            now[0] += 0.01
            return result

    monkeypatch.setattr(memberships, "_load_membership_source_rows", lambda **kwargs:
                        tuple((pk, "resolver-v1", "a" * 64, None, None) for pk in range(1, 301)))
    with pytest.raises((TimeoutError, SourcePreparationLimited)):
        with source_preparation_scope(1.0, lambda: now[0]):
            returned.append(memberships.load_automatic_membership_assignments(
                collection_ids=(7,), using="default", batch_size=1000, codec=SlowCodec(),
            ))
    assert returned == []
    assert len(calls) < 300


def test_replacement_future_source_deadline_cannot_mint_new_authority(monkeypatch):
    from apps.documents.services.source_deadline import _DEADLINE

    f = configured(monkeypatch)
    now, tokens, returned = [0.0], [], []
    original = repository.load_automatic_membership_assignments

    def replace_bound(**kwargs):
        result = original(**kwargs)
        tokens.append(_DEADLINE.set((100.0, lambda: now[0], None)))
        return result

    monkeypatch.setattr(repository, "load_automatic_membership_assignments", replace_bound)
    with source_preparation_scope(1.0, lambda: now[0]):
        try:
            with pytest.raises(TimeoutError):
                returned.append(authorities(f))
        finally:
            for token in reversed(tokens):
                _DEADLINE.reset(token)
    assert returned == []


def test_operation_only_original_budget_cannot_be_replaced(monkeypatch):
    from lib.retrieval.operation import operation_scope
    from lib.retrieval.turn_budget import TurnBudget, TurnLimits

    f = configured(monkeypatch)
    original = repository.load_automatic_membership_assignments
    returned = []
    with operation_scope(TurnBudget(TurnLimits())) as operation:
        old_budget = operation.budget

        def replace_budget(**kwargs):
            result = original(**kwargs)
            operation.budget = TurnBudget(TurnLimits())
            return result

        monkeypatch.setattr(repository, "load_automatic_membership_assignments", replace_budget)
        try:
            with pytest.raises(TimeoutError):
                returned.append(authorities(f))
        finally:
            operation.budget = old_budget
    assert returned == []


def test_malformed_null_artifact_is_shared_failure_alongside_valid_authority(monkeypatch):
    from apps.knowledge_graph.retrieval import ready_authority_reads_v2 as reads
    from apps.knowledge_graph.retrieval.ready_scope import ReadyScopeError

    f = configured(monkeypatch)
    original = reads.checked_rows

    def malformed(query, check):
        for row in original(query, check):
            if query.model is CollectionGraphProjection and row.collection_id == f.documents["A"].collection_id:
                row.artifact_id = None  # Corrupt present authority cannot become absence.
            yield row

    monkeypatch.setattr(reads, "checked_rows", malformed)
    with pytest.raises(ReadyScopeError):
        authorities(f)
