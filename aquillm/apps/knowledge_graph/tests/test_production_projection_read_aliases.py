"""Production retrieval reads only through the projection-source authority."""

from __future__ import annotations

from types import SimpleNamespace

from apps.documents.tests.hybrid_graph_test_support import Policy, authorization
from apps.knowledge_graph.projection import (
    django_projection_source,
    postgres_repository,
)
from apps.knowledge_graph.projection.identifiers import (
    HmacSha256ProjectionIdentifierCodec,
    ProjectionIdentifierDomain,
)
from apps.knowledge_graph.projection.serialization import projection_checksum
from apps.knowledge_graph.retrieval.production_extended import prepare_extended_branch
from apps.knowledge_graph.retrieval.ready_materialization import (
    materialize_selected_ready_chunks,
)
from apps.knowledge_graph.retrieval.ready_scope_repository import (
    load_selected_ready_scope,
)
from apps.knowledge_graph.tests.test_projection_records import _bundle
from apps.knowledge_graph.tests.test_ready_scope import (
    _DOC_A,
    _DOC_B,
    _authority,
)


class _Query:
    def __init__(self, rows):
        self.rows = tuple(rows)

    def filter(self, **_kwargs):
        return self

    def order_by(self, *_fields):
        return self

    def values(self, *_fields):
        return self.rows

    def __iter__(self):
        return iter(self.rows)


class _SourceManager:
    def __init__(self, rows, aliases):
        self.rows, self.aliases = rows, aliases

    def using(self, alias):
        self.aliases.append(alias)
        if alias == "projection_state":
            raise PermissionError("function-only projection state denied SELECT")
        if alias != "projection_source":
            raise AssertionError(f"unexpected projection read alias: {alias}")
        return _Query(self.rows)


def _ready_rows(authorities):
    projections, states, artifacts, inputs = [], [], [], []
    for row in authorities:
        projections.append(
            {
                "id": row.projection_id,
                "generation_key": row.generation_id,
                "collection_id": row.collection_id,
                "collection_pk_snapshot": row.collection_id,
                "artifact_id": row.artifact_id,
                "artifact_pk_snapshot": row.artifact_id,
                "schema_version": row.schema_version,
                "projection_version": row.projection_version,
                "identifier_key_version": row.identifier_key_version,
                "membership_epoch": row.membership_epoch,
                "membership_checksum": row.membership_checksum,
                "graph_checksum": row.graph_checksum,
                "private_mapping_checksum": row.private_mapping_checksum,
            }
        )
        states.append(
            {
                "collection_id": row.collection_id,
                "active_artifact_id": row.artifact_id,
                "registry_epoch": row.membership_epoch,
                "membership_checksum": row.membership_checksum,
                "resolver_version": row.resolver_version,
                "resolution_config_checksum": row.resolution_config_checksum,
            }
        )
        artifacts.append(
            {
                "id": row.artifact_id,
                "collection_scope_id": row.collection_id,
                "ontology_version": row.ontology_version,
                "ontology_checksum": row.ontology_checksum,
                "embedding_model_signature": row.embedding_model_signature,
            }
        )
        inputs.extend(
            {
                "artifact_id": row.artifact_id,
                "document_id": document_id,
                "document_artifact_id": artifact_id,
            }
            for document_id, artifact_id in row.documents
        )
    return projections, states, artifacts, inputs


def test_ready_scope_succeeds_when_state_select_is_denied(monkeypatch):
    from apps.knowledge_graph import models
    from apps.knowledge_graph.projection import runtime as projection_runtime

    authorities = (_authority(7, _DOC_A, "1"), _authority(9, _DOC_B, "2"))
    aliases = []
    for model, rows in zip(
        (
            models.CollectionGraphProjection,
            models.CollectionGraphMembershipState,
            models.GraphArtifact,
            models.CollectionArtifactInput,
        ),
        _ready_rows(authorities),
        strict=True,
    ):
        monkeypatch.setattr(model, "objects", _SourceManager(rows, aliases))
    codec = HmacSha256ProjectionIdentifierCodec(b"secret", key_version="key-v1")
    monkeypatch.setattr(
        projection_runtime, "projection_identifier_codec", lambda _settings: codec
    )
    settings = SimpleNamespace(
        projection_schema_version="collection-graph-v1",
        projection_format_version="projection-v1",
        projection_identifier_key_version="key-v1",
    )

    scope = load_selected_ready_scope(
        authorization=authorization(Policy()), settings=settings
    )

    assert scope.projections == authorities
    assert aliases == ["projection_source"] * 4


class _ReferenceQuery(_Query):
    def values_list(self, *_fields):
        return tuple(
            (row["projection_id"], row["projection_chunk_key"])
            for row in self.rows
        )


class _ReferenceManager(_SourceManager):
    def using(self, alias):
        super().using(alias)
        return _ReferenceQuery(self.rows)


def test_private_materialization_uses_source_for_authority_and_default_for_chunks(
    monkeypatch,
):
    from apps.documents.models import TextChunk
    from apps.knowledge_graph import models
    from apps.knowledge_graph.tests.test_production_hybrid_runtime import _ready_scope

    scope = _ready_scope()
    authority = scope.projections[0]
    codec = HmacSha256ProjectionIdentifierCodec(b"secret", key_version="key-v1")
    key = codec.encode(
        ProjectionIdentifierDomain.CHUNK,
        generation=authority.generation_id,
        source=10,
    )
    chunk = SimpleNamespace(pk=10, doc_id=_DOC_A, chunk_number=2)
    reference = {
        "projection_id": authority.projection_id,
        "projection_chunk_key": key.value,
        "integer_chunk_pk": 10,
        "document_uuid": _DOC_A,
        "chunk_number": 2,
    }
    aliases = []
    monkeypatch.setattr(
        models.ProjectionChunkReference,
        "objects",
        _ReferenceManager((reference,), aliases),
    )

    class ProjectionManager(_SourceManager):
        def using(self, alias):
            super().using(alias)
            return SimpleNamespace(
                get=lambda **_kwargs: SimpleNamespace(
                    private_mapping_checksum=authority.private_mapping_checksum
                )
            )

    monkeypatch.setattr(
        models.CollectionGraphProjection, "objects", ProjectionManager((), aliases)
    )

    class ChunkManager:
        def using(self, alias):
            aliases.append(alias)
            if alias != "default":
                raise AssertionError("private chunk objects must use request authority")
            return _Query((chunk,))

    monkeypatch.setattr(TextChunk, "objects", ChunkManager())

    result = materialize_selected_ready_chunks(
        scope=scope,
        chunk_keys=(key,),
        authorization=authorization(Policy()),
    )

    assert tuple(row.candidate_object for row in result) == (chunk,)
    assert aliases == ["projection_source"] * 3 + ["default"]


def test_extended_seed_conversion_uses_only_projection_source(monkeypatch):
    bundle = _bundle()
    projection_id = _authority(7, _DOC_A, "1").projection_id
    authority = SimpleNamespace(
        projection_id=projection_id,
        generation_id=_authority(7, _DOC_A, "1").generation_id,
        graph_checksum=projection_checksum(bundle),
        documents=((_DOC_A, 107),),
    )
    scope = SimpleNamespace(
        projections=(authority,),
        generation_keys_by_projection=(
            (projection_id, bundle.generation.generation_key),
        ),
    )
    aliases = []

    class Source:
        def __init__(self, using, *, state_using, **_kwargs):
            aliases.extend((using, state_using))
            if "projection_state" in (using, state_using):
                raise PermissionError("function-only state denied SELECT")

    class Repository:
        def __init__(self, *, using, source):
            aliases.append(using)
            self.source = source

        def load_projection_bundle(self, **_kwargs):
            return bundle

    monkeypatch.setattr(django_projection_source, "DjangoProjectionRowSource", Source)
    monkeypatch.setattr(postgres_repository, "PostgresProjectionRepository", Repository)
    auth = authorization(Policy(), collection_ids=(7,), document_ids=(_DOC_A,))
    runtime = SimpleNamespace(
        authorization=auth,
        settings=SimpleNamespace(
            projection_identifier_hmac_key=SimpleNamespace(
                get_secret_value=lambda: "secret"
            ),
            projection_identifier_key_version="key-v1",
            projection_schema_version="collection-graph-v1",
            projection_format_version="projection-v1",
            projection_batch_size=50,
        ),
        projection_repository_factory=None,
        codec=SimpleNamespace(
            encode=lambda *_args, **_kwargs: SimpleNamespace(
                value=bundle.entity_mentions[0].chunk_key
            )
        ),
        clock=lambda: 0.0,
        _exact_request=lambda *_args: None,
        _shared_scope=lambda _shared: scope,
    )
    settings = SimpleNamespace(
        graph_extended_enabled=True,
        graph_extended_max_seeds=3,
    )
    baseline = SimpleNamespace(
        graph_seeds=(SimpleNamespace(chunk_id=10, restart_weight=1.0),),
        baseline_candidates=(SimpleNamespace(pk=10, doc_id=_DOC_A),),
    )

    seeds = prepare_extended_branch(
        runtime,
        baseline=baseline,
        shared=object(),
        authorization=auth,
        settings=settings,
        deadline=1.0,
    )

    assert tuple(row.identity_key for row in seeds) == (
        bundle.entity_mentions[0].entity_key,
    )
    assert aliases == ["projection_source"] * 3
