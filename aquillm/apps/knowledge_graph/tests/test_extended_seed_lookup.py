from types import SimpleNamespace

import pytest

from apps.documents.tests.hybrid_graph_test_support import Policy, authorization
from apps.knowledge_graph.projection.identifiers import (
    HmacSha256ProjectionIdentifierCodec,
)
from apps.knowledge_graph.retrieval.production_extended import prepare_extended_branch
from apps.knowledge_graph.tests.test_production_hybrid_runtime import _ready_scope
from apps.knowledge_graph.tests.test_ready_scope import _DOC_A


def test_extended_preparation_loads_only_selected_seed_chunks():
    scope = _ready_scope()
    calls = []

    class Repository:
        def load_seed_identities(self, *, authority, chunks, **kwargs):
            calls.append((authority.projection_id, chunks))
            return {1: ("a" * 64,)}

        def load_projection_bundle(self, **kwargs):
            raise AssertionError("whole collection loading is forbidden")

    settings = SimpleNamespace(
        graph_extended_enabled=True,
        graph_extended_max_seeds=3,
        projection_batch_size=50,
    )
    runtime = SimpleNamespace(
        authorization=authorization(Policy()),
        settings=settings,
        codec=HmacSha256ProjectionIdentifierCodec(b"secret", key_version="key-v1"),
        clock=lambda: 0.0,
        projection_repository_factory=Repository,
        _exact_request=lambda *args: None,
        _shared_scope=lambda _: scope,
    )
    result = prepare_extended_branch(
        runtime,
        baseline=SimpleNamespace(
            graph_seeds=(SimpleNamespace(chunk_id=1, restart_weight=1.0),),
            baseline_candidates=(SimpleNamespace(pk=1, doc_id=_DOC_A),),
        ),
        shared=object(),
        authorization=runtime.authorization,
        settings=settings,
        deadline=1.0,
    )
    assert [(row.identity_key, row.mass) for row in result] == [("a" * 64, 1.0)]
    assert calls == [(scope.projections[0].projection_id, ((1, _DOC_A),))]


@pytest.mark.parametrize("invalid", ("revoked", "stale", "wrong_document", "over_cap"))
def test_seed_lookup_rejects_stale_unauthorized_or_unbounded_rows(monkeypatch, invalid):
    from apps.knowledge_graph.retrieval import extended_seed_repository as source

    scope = _ready_scope()
    policy = Policy()
    auth = authorization(policy)
    if invalid == "revoked":
        policy.rows = ()
    monkeypatch.setattr(source, "_authority_is_current", lambda **_: invalid != "stale")
    monkeypatch.setattr(
        source,
        "_chunk_documents",
        lambda **_: {} if invalid == "wrong_document" else {1: _DOC_A},
    )
    monkeypatch.setattr(
        source,
        "_seed_rows",
        lambda **_: ((7, None), (9, 42)) if invalid == "over_cap" else ((7, None),),
    )
    repository = source.ExtendedSeedRepository()
    with pytest.raises(ValueError):
        repository.load_seed_identities(
            authority=scope.projections[0],
            chunks=((1, _DOC_A),),
            authorization=auth,
            codec=HmacSha256ProjectionIdentifierCodec(b"secret", key_version="key-v1"),
            max_rows=1,
        )


def test_seed_lookup_matches_projection_identifiers(monkeypatch):
    from apps.knowledge_graph.projection.identifiers import (
        ProjectionIdentifierDomain as Domain,
    )
    from apps.knowledge_graph.retrieval import extended_seed_repository as source

    scope = _ready_scope()
    authority = scope.projections[0]
    codec = HmacSha256ProjectionIdentifierCodec(b"secret", key_version="key-v1")
    monkeypatch.setattr(source, "_authority_is_current", lambda **_: True)
    monkeypatch.setattr(source, "_chunk_documents", lambda **_: {1: _DOC_A})
    monkeypatch.setattr(source, "_seed_rows", lambda **_: ((7, None), (9, 42)))
    result = source.ExtendedSeedRepository().load_seed_identities(
        authority=authority,
        chunks=((1, _DOC_A),),
        authorization=authorization(Policy()),
        codec=codec,
        max_rows=2,
    )
    assert set(result[1]) == {
        codec.encode(Domain.ENTITY, generation=authority.generation_id, source=7).value,
        codec.encode(Domain.AUTOMATIC_CANONICAL_IDENTITY, source=42).value,
    }


def test_seed_query_binds_chunk_observations_and_manifest_in_one_join():
    from apps.knowledge_graph.retrieval.extended_seed_repository import _seed_query

    authority = _ready_scope().projections[0]
    query = _seed_query(
        authority=authority,
        chunk_id=123,
        document_id=_DOC_A,
        document_artifact_id=authority.documents[0][1],
        using="projection_source",
    )
    sql, parameters = query[:4].query.sql_with_params()
    assert query.db == "projection_source"
    assert '"chunk_id" = %s' in sql
    assert "@>" in sql  # JSON observations for non-representative chunk mentions.
    assert "123" in str(parameters)
    assert '"document_artifact_id" = %s' in sql
    assert '"registry_epoch" = %s' in sql
    assert '"membership_checksum" = %s' in sql
    assert '"canonical_entity_id"' in sql
    assert "LIMIT 4" in sql


def test_authority_query_compiles_with_current_projection_and_membership(monkeypatch):
    from django.db.models.query import QuerySet

    from apps.knowledge_graph.retrieval.extended_seed_repository import (
        _authority_is_current,
    )

    queries = []

    def exists(query):
        queries.append(query.query.sql_with_params())
        return False

    monkeypatch.setattr(QuerySet, "exists", exists)
    assert not _authority_is_current(
        authority=_ready_scope().projections[0], using="projection_source"
    )
    sql, _ = queries[0]
    assert '"graph_checksum" = %s' in sql
    assert '"generation_key" = %s' in sql
    assert '"active_artifact_id" = %s' in sql
