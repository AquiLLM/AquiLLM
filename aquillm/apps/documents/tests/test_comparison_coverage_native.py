"""Real native scoped metadata, original source charges and publication races."""

from types import SimpleNamespace
from contextlib import nullcontext

import pytest
from django.conf import settings
from django.contrib.auth.models import User

from apps.collections.models import Collection
from apps.collections.services.retrieval_authorization import (
    OpaquePrincipalReference,
    bind_retrieval_reauthorization_capability,
    freeze_retrieval_authorization_context,
)
from apps.documents.models import RawTextDocument, TextChunk
from apps.documents.services.chunk_search_candidates import collect_hybrid_candidate_snapshot
from apps.documents.services.chunk_search_scope import bind_search_scope, freeze_search_scope, reset_search_scope
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.source_loading import SourceRuntime, source_runtime_scope
from apps.documents.tests.representation_support import reference
from apps.documents.tests.test_embedding_generation import stage
from apps.documents.tests.test_graph_admission_channels import native_dependencies
from apps.documents.tests.test_comparison_coverage_policy import selection_binding as final_binding
from lib.embeddings.document_provider import observe_query_embedding
from lib.embeddings.tests.test_document_provider import transport as transport
from lib.retrieval.operation import operation_scope
from lib.retrieval.turn_budget import TurnBudget, TurnLimits

pytestmark = pytest.mark.django_db(transaction=True)
QUESTION = "What are the compatibility changes from the second revision to the third revision of Quartz?"


@pytest.fixture
def pair(transport, monkeypatch):
    from apps.documents.services.embedding_index_store import activate_embedding_generation

    user = User.objects.create(username="comparison-owner")
    collection = Collection.objects.create(name="Generic comparison")
    shared = ("These adjacent stored source passages share a uniquely worded transition that "
              "preserves the context of the documented compatibility measurements and their conditions.")
    documents, entries = [], []
    for version in ("second", "third"):
        bodies = [f"This paper presents the {version} revision of Quartz."]
        bodies += [f"Unrelated appendix paragraph {number}." for number in range(1, 17)]
        bodies[5] = (QUESTION + " ") * 3 + shared
        bodies[6] = shared + (" Additional measured behavior follows in this independent source passage." * 6)
        text, spans = "", []
        for number, body in enumerate(bodies):
            start = len(text) - len(shared) if number == 6 else len(text)
            text += body[len(shared):] if number == 6 else body
            spans.append((start, start + len(body), body))
            text += "\n" if number != 5 else ""
        document = RawTextDocument(title=f"Technical report {version}", full_text=text,
                                   collection=collection, ingested_by=user)
        document.save(dont_rechunk=True)
        documents.append(document)
        entries.append(spans)
    rows = []
    for number in range(17):
        for document, spans in zip(documents, entries, strict=True):
            start, end, body = spans[number]
            rows.append(TextChunk(doc_id=document.id, content=body, chunk_number=number,
                                  start_position=start, end_position=end))
    TextChunk.objects.bulk_create(rows)
    for document in documents:
        _, lease, _, _ = stage((document, reference(document)), transport)
        assert activate_embedding_generation(lease, using="default") == "activated"
    monkeypatch.setattr(settings, "DOCUMENT_EMBEDDING_SPACE_KEY", transport[0].space_key)
    monkeypatch.setattr(settings, "RAG_CANDIDATE_MULTIPLIER", 1.7, raising=False)
    monkeypatch.setattr(settings, "RAG_SHORT_QUERY_CANDIDATE_SCALE", 1.0, raising=False)
    monkeypatch.setattr(settings, "RAG_TRIGRAM_SIMILARITY_MIN", 0.0, raising=False)

    class Policy:
        policy_version = "comparison-test-v1"
        policy_checksum = "a" * 64
        allowed = True

        def opaque_principal_reference(self, **kwargs):
            return OpaquePrincipalReference("b" * 64)

        def current_authorized_document_scope(self, **kwargs):
            return tuple((collection.pk, document.id) for document in documents) if self.allowed else ()

    policy, principal = Policy(), object()
    auth = freeze_retrieval_authorization_context(
        principal=principal, database_alias="default", policy=policy,
        selected_collection_ids=(collection.pk,), selected_document_ids=tuple(d.id for d in documents),
        reauthorization_capability=bind_retrieval_reauthorization_capability(principal=principal, policy=policy))
    runtime = SourceRuntime(TurnBudget(TurnLimits()), auth)
    return SimpleNamespace(documents=tuple(documents), rows=rows, runtime=runtime, policy=policy)


def capture(pair, transport, *, owned=True):
    from apps.knowledge_graph.retrieval.expansion import get_graph_expansion_config

    runtime = pair.runtime
    with source_runtime_scope(runtime):
        scope = freeze_search_scope(pair.documents, using="default", authorization=runtime.authorization)
        token = bind_search_scope(scope, QUESTION, graph_enabled=True)
        try:
            assert runtime.budget.reserve_action("initial-comparison")
            observation = observe_query_embedding(QUESTION, space=transport[0], budget=runtime.budget)
            operation = operation_scope(runtime.budget, timeout_ms=26000, source_limit=30) if owned else nullcontext()
            with operation:
                return collect_hybrid_candidate_snapshot(
                    TextChunk, QUESTION, 10, pair.documents, query_embedding=observation,
                    graph_config=get_graph_expansion_config(), authorization_context=runtime.authorization,
                    app_config_getter=lambda _: SimpleNamespace(vector_top_k=17, trigram_top_k=17),
                    _graph_dependencies=native_dependencies(runtime.authorization))
        finally:
            reset_search_scope(token)


@pytest.mark.parametrize("enabled", [False, True])
def test_native_comparison_requires_opt_in_and_preserves_the_initial_ledger(pair, transport, monkeypatch, enabled):
    monkeypatch.setenv("RAG_COMPARISON_COVERAGE_ENABLED", "1" if enabled else "0")
    snapshot = capture(pair, transport)
    assert snapshot.source_admission is not None
    assert pair.runtime.budget.sources_used <= 30
    record = getattr(pair.runtime, "comparison_coverage", None)
    if enabled:
        assert record is not None
        assert len(record.groups) == 2
        assert all(1 <= len(group.sources) <= 2 for group in record.groups)
        assert pair.runtime.budget.actions_used == 3
        assert {source.chunk_id for group in record.groups for source in group.sources} <= {
            row.pk for row in snapshot.baseline_candidates}
    else:
        assert record is None
        assert pair.runtime.budget.actions_used == 1


def test_missing_original_operation_preserves_ordinary_reads(pair, transport, monkeypatch):
    from apps.knowledge_graph.retrieval.graph_candidate_stage_v2 import _GraphSourceStage

    monkeypatch.setenv("RAG_COMPARISON_COVERAGE_ENABLED", "1")
    original, reads = _GraphSourceStage.read_captured, []

    def read(stage, refs):
        reads.append(tuple(ref.pk for ref in refs))
        return original(stage, refs)

    monkeypatch.setattr(_GraphSourceStage, "read_captured", read)
    snapshot = capture(pair, transport, owned=False)
    assert reads == [snapshot.source_admission.initial_ids]
    assert pair.runtime.budget.actions_used == 1
    assert pair.runtime.comparison_coverage is None


@pytest.mark.parametrize("race", ["permission", "source_revision", "flag"])
def test_changed_owner_after_real_group_read_cannot_publish_coverage(pair, transport, monkeypatch, race):
    from apps.knowledge_graph.retrieval.graph_candidate_stage_v2 import _GraphSourceStage

    monkeypatch.setenv("RAG_COMPARISON_COVERAGE_ENABLED", "1")
    original = _GraphSourceStage.read_captured
    reads = []

    def changed(stage, refs):
        rows = original(stage, refs)
        reads.append(tuple(ref.pk for ref in refs))
        if len(reads) == 2:
            if race == "permission":
                pair.policy.allowed = False
            elif race == "source_revision":
                from concurrent.futures import ThreadPoolExecutor
                from django.db import connection

                def committed_revision():
                    try:
                        document = RawTextDocument.objects.get(pkid=pair.documents[0].pkid)
                        document.full_text += " Independently committed revision."
                        document.save(dont_rechunk=True)
                    finally:
                        connection.close()

                # The native metadata owner is deliberately read-only. A real
                # concurrent edit must commit on its own connection and survive
                # the retrieval owner's eventual rollback.
                with ThreadPoolExecutor(max_workers=1) as executor:
                    executor.submit(committed_revision).result(timeout=10)
            else:
                monkeypatch.setenv("RAG_COMPARISON_COVERAGE_ENABLED", "0")
        return rows

    monkeypatch.setattr(_GraphSourceStage, "read_captured", changed)
    with pytest.raises(SourceScopeUnavailable):
        capture(pair, transport)
    assert len(reads) >= 2 and pair.runtime.budget.sources_used > 0
    assert getattr(pair.runtime, "comparison_coverage", None) is None


@pytest.mark.parametrize("expire_during_read", [False, True])
def test_final_validation_sql_uses_remaining_ledger_without_ambient_deadline(final_binding, expire_during_read):
    from django.db import connection
    from apps.documents.services.comparison_coverage_policy import bind_coverage, selection_coverage_groups
    from apps.documents.services.source_deadline import _DEADLINE
    from apps.documents.services.source_loading import bounded_source_database
    from lib.retrieval.operation import _OPERATION

    runtime, prepared, pool, _ = final_binding
    now = [0.0]
    runtime.budget = TurnBudget(TurnLimits(retrieval_ms=30000), clock=lambda: now[0])
    frozen = runtime._frozen_retrieval_scope
    runtime.comparison_coverage = bind_coverage(
        SimpleNamespace(runtime=runtime, representation=frozen.representation, dense=frozen.dense),
        QUESTION, runtime.comparison_coverage.groups)
    frozen.database = lambda: bounded_source_database(runtime, "default")
    assert _DEADLINE.get() is None and _OPERATION.get() is None and not connection.in_atomic_block
    observed = []

    def validate():
        assert connection.in_atomic_block
        with connection.cursor() as cursor:
            cursor.execute("SELECT setting::bigint FROM pg_settings WHERE name='statement_timeout'")
            milliseconds = cursor.fetchone()[0]
        assert 0 < milliseconds <= runtime.budget.remaining_ms() <= 50
        observed.append(milliseconds)
        if expire_during_read:
            now[0] = 30.1

    frozen.validate = validate
    now[0] = 29.95
    with source_runtime_scope(runtime):
        groups = selection_coverage_groups(runtime, prepared, pool, QUESTION)
    assert len(groups) == (0 if expire_during_read else 2)
    assert observed and not connection.in_atomic_block


@pytest.mark.parametrize("failure", ["child", "query_cancel", "real_query_cancel", "permission", "parent"])
def test_optional_scope_expiry_preserves_only_live_authorized_global_metadata(pair, transport, monkeypatch, failure):
    from django.db import OperationalError
    from apps.documents.services import graph_candidate_metadata as metadata
    from apps.documents.services.source_deadline import _DEADLINE, source_preparation_scope
    from apps.documents.services.source_loading import SourcePreparationLimited

    monkeypatch.setenv("RAG_COMPARISON_COVERAGE_ENABLED", "1")
    now = [0.0]
    pair.runtime.budget = TurnBudget(TurnLimits(retrieval_ms=30000), clock=lambda: now[0])
    original, initial, probes = metadata._capture_metadata, [], []

    def expire(model, query, scope, *args, **kwargs):
        result = original(model, query, scope, *args, **kwargs)
        if len(scope.requested_document_ids) == 2:
            initial.append({name: tuple(ref.pk for ref in lane) for name, lane in result[0].items()})
        else:
            probes.append(tuple(scope.requested_document_ids))
            child = _DEADLINE.get()
            assert child is not None
            if failure == "real_query_cancel":
                from django.db import connection

                try:
                    with connection.cursor() as cursor:
                        cursor.execute("SET LOCAL statement_timeout = '1ms'")
                        cursor.execute("SELECT pg_sleep(0.05)")
                except OperationalError:
                    now[0] = child[0] + 0.001
                    raise
                raise AssertionError("PostgreSQL did not cancel the scoped query")
            now[0] = 30.1 if failure == "parent" else child[0] + 0.001
            if failure == "permission":
                pair.policy.allowed = False
            if failure == "query_cancel":
                class Cancelled(Exception):
                    pgcode = "57014"
                raise OperationalError("synthetic scoped statement timeout") from Cancelled()
        return result

    monkeypatch.setattr(metadata, "_capture_metadata", expire)
    if failure in ("permission", "parent"):
        with pytest.raises((SourceScopeUnavailable, SourcePreparationLimited)):
            with source_preparation_scope(26.0, lambda: now[0], pair.runtime.budget):
                capture(pair, transport)
    else:
        with source_preparation_scope(26.0, lambda: now[0], pair.runtime.budget):
            snapshot = capture(pair, transport)
            assert snapshot.source_admission.channel_ids == initial[0]
            assert snapshot.baseline_candidates
            assert pair.runtime.budget.can_publish()
            from django.db import connection
            assert not connection.needs_rollback
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                assert cursor.fetchone() == (1,)
    assert len(probes) == 1 and pair.runtime.budget.actions_used == 2
    assert 5 <= pair.runtime.budget.sources_used <= 30
    assert pair.runtime.comparison_coverage is None
