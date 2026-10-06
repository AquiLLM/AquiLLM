"""Run only against an explicitly configured isolated PostgreSQL test database."""

import os
from contextlib import nullcontext
from dataclasses import replace
from time import monotonic
from uuid import uuid4

import pytest
from django.db import DatabaseError, connection, connections, transaction

from apps.knowledge_graph.retrieval.direct_seed_sql import (
    DirectSeedReadTimeout,
    bounded_seed_read,
)

pytestmark = [
    pytest.mark.django_db,
    pytest.mark.skipif(
        os.environ.get("KG_REQUIRE_POSTGRES_TESTS") != "1",
        reason="explicit PostgreSQL integration opt-in required",
    ),
]


def _settings():
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT current_setting('statement_timeout'), "
            "current_setting('join_collapse_limit'), current_setting('enable_nestloop')"
        )
        return cursor.fetchone()


@pytest.mark.parametrize("alias", [False, True])
def test_settings_restore_inside_successful_outer_transaction(alias):
    assert connection.vendor == "postgresql"
    with transaction.atomic():
        with connection.cursor() as cursor:
            cursor.execute("SET LOCAL statement_timeout='20ms'")
            cursor.execute("SET LOCAL join_collapse_limit=7")
        before = _settings()
        with bounded_seed_read(using="default", deadline=monotonic() + 2, alias=alias):
            during = _settings()
            assert during == ("20ms", "1" if alias else "7", before[2])
        assert _settings() == before


@pytest.mark.parametrize("failure", ["python", "database", "timeout"])
def test_failure_rolls_back_settings_and_preserves_caller_connection(failure):
    assert connection.vendor == "postgresql"
    with transaction.atomic():
        with connection.cursor() as cursor:
            cursor.execute("SET LOCAL statement_timeout='2s'")
            cursor.execute("SET LOCAL join_collapse_limit=7")
        before = _settings()
        expected = {
            "python": ValueError,
            "database": DatabaseError,
            "timeout": DirectSeedReadTimeout,
        }[failure]
        started = monotonic()
        with pytest.raises(expected):
            with bounded_seed_read(
                using="default", deadline=monotonic() + 0.15, alias=True
            ):
                if failure == "python":
                    raise ValueError("fixture validation failure")
                with connection.cursor() as cursor:
                    cursor.execute(
                        "SELECT pg_sleep(3)" if failure == "timeout" else "SELECT 1 / 0"
                    )
        if failure == "timeout":
            assert monotonic() - started < 1.5
        assert _settings() == before
        with connection.cursor() as cursor:
            cursor.execute("SELECT 42")
            assert cursor.fetchone() == (42,)


def test_expired_deadline_executes_no_database_statements():
    observed = []

    def observe(execute, sql, params, many, context):
        observed.append(sql)
        return execute(sql, params, many, context)

    with connection.execute_wrapper(observe), pytest.raises(DirectSeedReadTimeout):
        with bounded_seed_read(using="default", deadline=monotonic() - 1, alias=True):
            pytest.fail("expired read started")
    assert observed == []


def test_submillisecond_budget_starts_no_read_or_zero_timeout():
    observed = []

    def observe(execute, sql, params, many, context):
        observed.append(sql)
        return execute(sql, params, many, context)

    before = _settings()
    with connection.execute_wrapper(observe), pytest.raises(DirectSeedReadTimeout):
        with bounded_seed_read(using="default", deadline=0.0005, clock=lambda: 0.0):
            with connection.cursor() as cursor:
                cursor.execute("SELECT 42")
    assert "SELECT 42" not in observed
    assert not any("set_config" in sql for sql in observed)
    assert _settings() == before


@pytest.mark.django_db(transaction=True)
def test_sql_cancellation_releases_all_worker_slots_for_following_request():
    from threading import Event

    from apps.knowledge_graph.retrieval.scheduler_workers import BoundedWorkerPool

    pool = BoundedWorkerPool()

    def slow():
        try:
            with bounded_seed_read(
                using="default", deadline=monotonic() + 0.15, alias=True
            ):
                with connections["default"].cursor() as cursor:
                    cursor.execute("SELECT pg_sleep(3)")
        finally:
            connections["default"].close()

    try:
        futures = pool.submit_batch(tuple((slow, ()) for _ in range(4)))
        assert futures is not None
        done = [Event() for _ in futures]
        for future, event in zip(futures, done, strict=True):
            future.add_done_callback(lambda _, event=event: event.set())
        for future in futures:
            with pytest.raises(DirectSeedReadTimeout):
                future.result(timeout=2)
        assert all(event.wait(timeout=2) for event in done)
        following = pool.submit_batch(tuple((lambda: 42, ()) for _ in range(4)))
        assert following is not None
        assert [future.result(timeout=2) for future in following] == [42] * 4
    finally:
        pool._executor.shutdown(wait=True)


@pytest.fixture
def scoped_aliases(monkeypatch):
    from django.contrib.auth.models import User

    from apps.collections.models import Collection
    from apps.documents.models import RawTextDocument, TextChunk
    from apps.knowledge_graph.models import (
        CollectionArtifactInput,
        CollectionEntity,
        CollectionEntityDocumentLink,
        CollectionGraphMembershipState,
        DocumentEntity,
        DocumentEntityMention,
        EntityMention,
        GraphArtifact,
    )
    from apps.knowledge_graph.retrieval.direct_seed_types import DirectSeedScopeV1
    from apps.knowledge_graph.tests.test_canonical_resolution import (
        _embedding_signature,
    )

    monkeypatch.setattr(
        TextChunk,
        "get_chunk_embedding",
        lambda *_args, **_kwargs: pytest.fail("SQL fixture called embedding provider"),
    )
    collection = Collection.objects.create(name="bounded aliases")
    user = User.objects.create_user(username="bounded-alias-fixture")
    document = RawTextDocument(
        title="fixture",
        full_text="Atlas Atlas Atlas",
        collection=collection,
        ingested_by=user,
        full_text_hash=RawTextDocument.hash_fn("Atlas Atlas Atlas"),
    )
    document.save(dont_rechunk=True)
    common = dict(
        source_hash="a" * 64,
        ontology_version="ontology-v1",
        ontology_checksum="b" * 64,
        extractor_version="extractor-v1",
        filter_policy_version="filter-v1",
    )
    source = GraphArtifact.objects.create(
        scope_type="document",
        scope_id=document.id,
        status="active",
        resolver_version="document-v1",
        **common,
    )
    artifact = GraphArtifact.objects.create(
        scope_type="collection",
        scope_id=collection.pk,
        status="building",
        resolver_version="collection-v1",
        embedding_model_signature=_embedding_signature(),
        **common,
    )
    manifest = CollectionArtifactInput.objects.create(
        artifact=artifact,
        collection=collection,
        document_id=document.id,
        document_artifact=source,
        source_signature="c" * 64,
        membership_signature="d" * 64,
    )
    chunk = TextChunk.objects.create(
        content=document.full_text,
        doc_id=document.id,
        chunk_number=0,
        start_position=0,
        end_position=17,
        embedding=[0.0] * 1024,
    )
    entities = []
    assignments = []
    for index in range(3):
        entity = CollectionEntity.objects.create(
            artifact=artifact,
            collection=collection,
            cluster_key=f"{index + 1:064x}",
            label="Atlas",
            normalized_label="atlas",
            entity_type="model",
            extraction_confidence=1.0,
            resolution_confidence=1.0,
            retrieval_utility=1.0,
            promotion_confidence=1.0,
        )
        local = DocumentEntity.objects.create(
            artifact=source,
            document_id=document.id,
            cluster_key=f"{index + 1:064x}",
            label="Atlas",
            normalized_label="atlas",
            entity_type="model",
            resolution_confidence=1.0,
        )
        mention = EntityMention.objects.create(
            artifact=source,
            document_id=document.id,
            chunk=chunk,
            start=index * 6,
            end=index * 6 + 5,
            position_basis="document_global",
            raw_text="Atlas",
            normalized_text="atlas",
            entity_type="model",
            extraction_confidence=1.0,
        )
        assignments.append(
            DocumentEntityMention.objects.create(
                document_entity=local,
                mention=mention,
                method="root",
                resolver_version=source.resolver_version,
            )
        )
        CollectionEntityDocumentLink.objects.create(
            artifact=artifact,
            manifest_input=manifest,
            document_entity=local,
            collection_entity=entity,
            score=1.0,
            method="singleton",
            resolver_version=artifact.resolver_version,
            outcome="candidate" if index == 2 else "automatic",
            candidate_rank=1 if index == 2 else None,
            decision_checksum=f"{index + 1:064x}",
        )
        entities.append(entity)
    GraphArtifact.objects.filter(pk__in=(source.pk, artifact.pk)).update(
        status="active"
    )
    CollectionGraphMembershipState.objects.create(
        collection=collection,
        active_artifact=artifact,
        registry_epoch=1,
        membership_checksum="e" * 64,
        resolver_version="canonical-v1",
        resolution_config_checksum="f" * 64,
    )
    scope = DirectSeedScopeV1(
        "1" * 64,
        (collection.pk,),
        (artifact.pk,),
        (document.id,),
        (source.pk,),
        ((artifact.pk, "2" * 64),),
        ((artifact.pk, uuid4()),),
        "b" * 64,
        artifact.resolver_version,
    )
    states = (
        {
            "collection_id": collection.pk,
            "active_artifact_id": artifact.pk,
            "registry_epoch": 1,
            "membership_checksum": "e" * 64,
            "resolver_version": "canonical-v1",
            "resolution_config_checksum": "f" * 64,
        },
    )
    return scope, states, entities, assignments


def test_alias_plan_preserves_scope_provenance_ambiguity_and_cap(
    scoped_aliases, monkeypatch
):
    from apps.knowledge_graph.models import DocumentEntityMention
    from apps.knowledge_graph.retrieval import direct_seed_queries as queries
    from apps.knowledge_graph.retrieval.direct_seed_contracts import (
        DirectEntityMatchV1,
        DirectResolutionTier,
    )
    from apps.knowledge_graph.retrieval.direct_seed_resolution import _best_exact

    scope, states, entities, assignments = scoped_aliases
    options = dict(
        tier=DirectResolutionTier.ALIAS,
        lookup="atlas",
        lookup_field="document_links__document_entity__mention_links__mention__normalized_text",
        ontology_type="model",
        membership_states=states,
        automatic_only=True,
        scope=scope,
        using="default",
        limit=4,
        deadline=monotonic() + 5,
    )
    observed = []

    def observe(execute, sql, params, many, context):
        if 'FROM "apps_knowledge_graph_collectionentity"' in sql:
            observed.append(_settings())
        return execute(sql, params, many, context)

    before = _settings()
    with connection.execute_wrapper(observe):
        planned = queries._load_candidate_rows(**options)
    assert observed and all(settings[1:] == ("1", before[2]) for settings in observed)
    assert _settings() == before
    with monkeypatch.context() as patch:
        patch.setattr(queries, "bounded_seed_read", lambda **_: nullcontext())
        unplanned = queries._load_candidate_rows(**options)
    assert planned == unplanned
    assert {row.entity_id for row in planned} == {row.pk for row in entities[:2]}
    matches = tuple(
        DirectEntityMatchV1(
            0,
            f"{row.entity_id:064x}",
            f"{row.entity_id:064x}",
            "model",
            DirectResolutionTier.ALIAS,
            1.0,
            1.0,
            0.9,
        )
        for row in planned
    )
    selected, ambiguity = _best_exact(matches, 0)
    assert selected is None
    assert ambiguity.component_count == 2
    with pytest.raises(ValueError, match="hard cap"):
        queries._load_candidate_rows(**{**options, "limit": 1})
    assert (
        queries._load_candidate_rows(
            **{**options, "scope": replace(scope, selected_document_ids=(uuid4(),))}
        )
        == ()
    )
    DocumentEntityMention.objects.filter(pk=assignments[0].pk).update(
        status="superseded"
    )
    remaining = queries._load_candidate_rows(**options)
    assert [row.entity_id for row in remaining] == [entities[1].pk]
