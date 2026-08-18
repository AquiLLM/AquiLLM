from __future__ import annotations

import inspect
from dataclasses import replace
from types import SimpleNamespace

import pytest

from apps.knowledge_graph.graph import assembly, filtering
from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun, RelationMention
from apps.knowledge_graph.resolution import collection as collection_resolution


def test_manifest_cap_is_shared_and_checksum_addressed_by_both_build_stages():
    cap = getattr(collection_resolution, "MAX_COLLECTION_DOCUMENT_INPUTS", None)

    assert type(cap) is int and 1 <= cap <= 10_000
    assembly_config = assembly.AssemblyConfig()
    resolution_config = collection_resolution.CollectionResolutionConfig()
    assert assembly_config.max_document_inputs == cap
    assert resolution_config.max_document_inputs == cap
    assert assembly.assembly_config_checksum(assembly_config) != (
        assembly.assembly_config_checksum(
            replace(assembly_config, max_document_inputs=cap - 1)
        )
    )
    assert collection_resolution.resolution_config_checksum(resolution_config) != (
        collection_resolution.resolution_config_checksum(
            replace(resolution_config, max_document_inputs=cap - 1)
        )
    )
    with pytest.raises(ValueError, match="max_document_inputs"):
        replace(assembly_config, max_document_inputs=cap + 1)
    with pytest.raises(ValueError, match="max_document_inputs"):
        replace(resolution_config, max_document_inputs=cap + 1)


def test_task9_bounds_document_inputs_before_materializing_the_source_iterable():
    bounded = getattr(collection_resolution, "_bounded_document_artifacts", None)
    assert callable(bounded)
    consumed = []

    def documents():
        for value in range(20):
            consumed.append(value)
            yield value

    with pytest.raises(
        collection_resolution.CollectionResolutionPersistenceError,
        match="manifest.*cap",
    ):
        bounded(documents(), 3)
    assert consumed == [0, 1, 2, 3]

    source = inspect.getsource(collection_resolution.build_collection_snapshot)
    assert "_bounded_document_artifacts(" in source
    assert source.index("_bounded_document_artifacts(") < source.index("source_ids =")


def test_task9_counts_queryset_like_inputs_before_bounded_iteration():
    events = []

    class QueryLike:
        def count(self):
            events.append("count")
            return 4

        def iterator(self, *, chunk_size):
            events.append(("iterator", chunk_size))
            raise AssertionError("over-cap query must not be iterated")

        def __iter__(self):
            raise AssertionError("queryset cache must not be materialized")

    with pytest.raises(
        collection_resolution.CollectionResolutionPersistenceError,
        match="manifest.*cap",
    ):
        collection_resolution._bounded_document_artifacts(QueryLike(), 3)

    assert events == ["count"]


def test_task10_counts_manifests_before_materializing_them():
    source = inspect.getsource(assembly._load_locked_manifest)

    assert "config.max_document_inputs" in source
    assert "_bounded_query_rows(" in source
    assert "tuple(manifest_query)" not in source


def test_task9_markers_audit_the_exact_manifest_cap():
    persistence_source = inspect.getsource(
        collection_resolution.persist_collection_resolution
    )
    marker_validation_source = inspect.getsource(
        collection_resolution._collection_resolution_marker_is_valid
    )
    filtering_source = inspect.getsource(filtering.create_filter_rerun_artifact)
    lineage_source = inspect.getsource(assembly._validate_task9_lineage_node)

    assert '"max_document_inputs": result.config.max_document_inputs' in (
        persistence_source
    )
    assert 'marker.get("max_document_inputs")' in marker_validation_source
    assert '"max_document_inputs": manifest_cap' in filtering_source
    assert '"max_document_inputs"' in lineage_source


def test_contributor_scope_and_relation_source_queries_use_bounded_batches():
    batcher = getattr(assembly, "_query_value_batches", None)
    batch_size = getattr(assembly, "_QUERY_PREDICATE_BATCH_SIZE", None)
    assert callable(batcher)
    assert type(batch_size) is int and 1 <= batch_size <= 10_000

    document_ids = tuple(
        f"00000000-0000-0000-0000-{value:012d}" for value in range(20_001)
    )
    artifact_ids = tuple(range(1, 20_002))
    document_batches = tuple(batcher(document_ids))
    artifact_batches = tuple(batcher(artifact_ids))
    assert max(map(len, document_batches)) <= batch_size
    assert max(map(len, artifact_batches)) <= batch_size

    for batch in document_batches:
        query = GraphArtifact.objects.filter(
            scope_type=GraphArtifact.ScopeType.DOCUMENT,
            scope_id__in=batch,
            status=GraphArtifact.Status.ACTIVE,
        )
        _sql, params = query.query.sql_with_params()
        assert len(params) <= batch_size + 2
    for batch in artifact_batches:
        query = RelationMention.objects.filter(artifact_id__in=batch)
        _sql, params = query.query.sql_with_params()
        assert len(params) <= batch_size

    contributor_source = inspect.getsource(assembly._lock_current_contributors)
    relation_source = inspect.getsource(assembly._load_assembly_evidence)
    assert "_query_value_batches(document_ids)" in contributor_source
    assert "scope_id__in=document_id_batch" in contributor_source
    assert "_query_value_batches(current_document_ids)" in contributor_source
    assert "scope_id__in=current_document_id_batch" in contributor_source
    assert "_query_value_batches(source_artifact_ids)" in relation_source
    assert "artifact_id__in=source_artifact_batch" in relation_source


def test_competing_build_runs_use_collection_scope_join_without_id_lists():
    query = GraphBuildRun.objects.filter(
        artifact__scope_type=GraphArtifact.ScopeType.COLLECTION,
        artifact__scope_id="42",
    ).order_by("pk")
    sql, params = query.query.sql_with_params()

    assert "JOIN" in sql.upper()
    assert set(params) == {GraphArtifact.ScopeType.COLLECTION, "42"}

    sources = (
        inspect.getsource(assembly._locked_candidate),
        inspect.getsource(collection_resolution.build_collection_snapshot),
        inspect.getsource(collection_resolution.persist_collection_resolution),
        inspect.getsource(filtering.create_filter_rerun_artifact),
    )
    assert all("artifact__scope_type" in source for source in sources)
    assert all("artifact__scope_id" in source for source in sources)
    assert all(
        "filter(artifact_id__in=(row.pk for row in scope_artifacts))" not in source
        for source in sources
    )


def test_collection_resolution_caps_are_checksum_addressed_for_every_row_family():
    config = collection_resolution.CollectionResolutionConfig()
    cap_fields = (
        "max_entities",
        "max_memberships",
        "max_relations",
        "max_links",
    )

    for field_name in cap_fields:
        assert type(getattr(config, field_name)) is int
        assert getattr(config, field_name) > 1
        assert collection_resolution.resolution_config_checksum(config) != (
            collection_resolution.resolution_config_checksum(
                replace(config, **{field_name: getattr(config, field_name) - 1})
            )
        )


def test_pure_collection_resolution_and_filter_bound_iterables_before_tuple():
    resolution_source = inspect.getsource(
        collection_resolution.resolve_collection_entities
    )
    filter_source = inspect.getsource(filtering.filter_collection_resolution)

    assert "islice(iter(entities), config.max_entities + 1)" in resolution_source
    assert "islice(iter(relations), config.max_relations + 1)" in resolution_source
    assert "islice(iter(entities), resolution.config.max_entities + 1)" in filter_source


def test_collection_query_cap_counts_before_iterator_or_cache_materialization():
    bounded = getattr(collection_resolution, "_bounded_query_rows", None)
    assert callable(bounded)
    events = []

    class OverCapQuery:
        def count(self):
            events.append("count")
            return 2

        def iterator(self, *, chunk_size):
            events.append(("iterator", chunk_size))
            raise AssertionError("over-cap query must not be iterated")

        def __iter__(self):
            raise AssertionError("over-cap query must not populate its cache")

    with pytest.raises(
        collection_resolution.CollectionResolutionPersistenceError,
        match="membership.*cap",
    ):
        bounded(OverCapQuery(), 1, "collection membership")
    assert events == ["count"]


def test_collection_query_cap_bounds_actual_iteration_after_count_drift():
    consumed = []

    class Query:
        def count(self):
            return 1

        def iterator(self, *, chunk_size):
            assert chunk_size == 2
            for value in range(10):
                consumed.append(value)
                yield value

    with pytest.raises(
        collection_resolution.CollectionResolutionPersistenceError,
        match="relation.*cap",
    ):
        collection_resolution._bounded_query_rows(Query(), 2, "collection relation")
    assert consumed == [0, 1, 2]


def test_collection_endpoint_queries_batch_dedupe_and_abort_at_cap_plus_one():
    collect = getattr(collection_resolution, "_bounded_batched_query_rows", None)
    batch_size = getattr(collection_resolution, "_QUERY_PREDICATE_BATCH_SIZE", None)
    assert callable(collect)
    assert batch_size == 5_000
    batches = []

    class Query:
        def __init__(self, rows):
            self.rows = rows

        def iterator(self, *, chunk_size):
            assert 1 <= chunk_size <= 1_000
            yield from self.rows

    def query_factory(batch):
        batches.append(batch)
        return Query(
            [SimpleNamespace(pk=value) for value in reversed(batch)]
            + [SimpleNamespace(pk=1)]
        )

    rows = collect(
        range(1, 12_002),
        query_factory,
        maximum=20_000,
        label="endpoint rows",
        row_key=lambda row: row.pk,
        sort_key=lambda row: row.pk,
    )
    assert [row.pk for row in rows] == list(range(1, 12_002))
    assert [len(batch) for batch in batches] == [5_000, 5_000, 2_001]

    consumed = []
    batches.clear()

    class CapQuery(Query):
        def iterator(self, *, chunk_size):
            for row in self.rows:
                consumed.append(row.pk)
                yield row

    def cap_factory(batch):
        batches.append(batch)
        return CapQuery([SimpleNamespace(pk=value) for value in batch])

    with pytest.raises(
        collection_resolution.CollectionResolutionPersistenceError,
        match="endpoint rows.*cap",
    ):
        collect(
            range(1, 12_002),
            cap_factory,
            maximum=5_000,
            label="endpoint rows",
            row_key=lambda row: row.pk,
            sort_key=lambda row: row.pk,
        )
    assert len(consumed) == 5_001
    assert len(batches) == 2
    assert all(len(batch) <= batch_size for batch in batches)


def test_raw_relation_snapshot_accepts_200001_under_exact_task9_cap():
    fingerprint_rows = getattr(
        collection_resolution,
        "_fingerprint_raw_relation_rows",
        None,
    )
    assert callable(fingerprint_rows)

    def rows(count, consumed=None):
        for index in range(1, count + 1):
            if consumed is not None:
                consumed.append(index)
            yield {
                "pk": index,
                "artifact_id": 1,
                "document_id": "11111111-1111-4111-8111-111111111111",
                "chunk_id": index,
                "head_id": index * 2,
                "tail_id": index * 2 + 1,
                "relation_type": "uses",
                "extraction_confidence": 0.9,
                "metadata": {},
            }

    count, fingerprint = fingerprint_rows(rows(200_001), max_relations=250_000)
    assert count == 200_001
    assert len(fingerprint) == 64

    consumed = []
    with pytest.raises(
        collection_resolution.CollectionResolutionPersistenceError,
        match="raw relation.*cap",
    ):
        fingerprint_rows(rows(200_001, consumed), max_relations=200_000)
    assert consumed[-1] == 200_001
    assert len(consumed) == 200_001


def test_task9_and_filter_loaders_guard_every_large_queryset_before_conversion():
    resolution_source = inspect.getsource(
        collection_resolution._load_resolution_source_rows
    )
    filter_projection_source = inspect.getsource(
        collection_resolution._filter_inputs_for_resolution
    )
    load_resolution_source = inspect.getsource(
        collection_resolution.load_collection_resolution_inputs
    )
    load_filter_source = inspect.getsource(
        collection_resolution.load_collection_filter_inputs
    )
    persist_resolution_source = inspect.getsource(
        collection_resolution.persist_collection_resolution
    )
    artifact_filter_source = inspect.getsource(filtering._filter_inputs_from_artifact)
    rerun_source = inspect.getsource(filtering.create_filter_rerun_artifact)
    existing_rerun_source = inspect.getsource(filtering._validate_existing_filter_rerun)
    lineage_source = inspect.getsource(assembly._load_filter_source_lineage)

    for source, query_names in (
        (load_resolution_source, ("manifest_query",)),
        (load_filter_source, ("manifest_query",)),
        (persist_resolution_source, ("manifest_query",)),
        (
            artifact_filter_source,
            (
                "entity_query",
                "automatic_link_query",
            ),
        ),
        (rerun_source, ("source_manifest_query", "source_link_query")),
        (existing_rerun_source, ("manifest_query", "entity_query", "link_query")),
        (
            lineage_source,
            ("source_manifest_query", "source_entity_query", "source_link_query"),
        ),
    ):
        compact = " ".join(source.split())
        for query_name in query_names:
            assert f"_bounded_query_rows( {query_name}" in compact

    for source, query_names in (
        (
            resolution_source,
            ("source_entity_query", "membership_query", "relation_query"),
        ),
        (filter_projection_source, ("membership_query", "relation_query")),
        (artifact_filter_source, ("membership_query", "relation_query")),
    ):
        compact = " ".join(source.split())
        assert "_bounded_batched_query_rows(" in compact
        for query_name in query_names:
            assert f"def {query_name}(" in compact
            assert f"{query_name}," in compact

    for source in (load_filter_source, persist_resolution_source):
        assert "_bounded_batched_query_rows(" in source
        assert "artifact_id__in=artifact_id_batch" in source


def test_task9_and_filter_endpoint_in_predicates_are_always_fixed_batches():
    resolution_source = inspect.getsource(
        collection_resolution._load_resolution_source_rows
    )
    filter_projection_source = inspect.getsource(
        collection_resolution._filter_inputs_for_resolution
    )
    artifact_filter_source = inspect.getsource(filtering._filter_inputs_from_artifact)
    raw_snapshot_source = inspect.getsource(
        collection_resolution._raw_relation_snapshot
    )

    assert "artifact_id__in=artifact_id_batch" in resolution_source
    assert "document_entity_id__in=document_entity_id_batch" in resolution_source
    assert "head_id__in=mention_id_batch" in resolution_source
    assert "tail_id__in=mention_id_batch" in resolution_source
    assert "document_entity_id__in=document_entity_id_batch" in (
        filter_projection_source
    )
    assert "head_id__in=mention_id_batch" in filter_projection_source
    assert "tail_id__in=mention_id_batch" in filter_projection_source
    assert "document_entity_id__in=document_entity_id_batch" in (artifact_filter_source)
    assert "head_id__in=mention_id_batch" in artifact_filter_source
    assert "tail_id__in=mention_id_batch" in artifact_filter_source
    assert "artifact_id__in=artifact_id_batch" in raw_snapshot_source

    forbidden = (
        "artifact_id__in=manifest_by_artifact",
        "document_entity__in=source_entities",
        "document_entity_id__in=source_ids",
        "head_id__in=mention_owner",
        "tail_id__in=mention_owner",
        "head_id__in=mention_ids",
        "tail_id__in=mention_ids",
        "head_id__in=all_mention_ids",
        "tail_id__in=all_mention_ids",
    )
    joined = "\n".join(
        (resolution_source, filter_projection_source, artifact_filter_source)
    )
    assert all(pattern not in joined for pattern in forbidden)


def test_task9_and_filter_markers_persist_every_checksum_addressed_row_cap():
    persist_source = inspect.getsource(
        collection_resolution.persist_collection_resolution
    )
    marker_validation_source = inspect.getsource(
        collection_resolution._collection_resolution_marker_is_valid
    )
    filter_source = inspect.getsource(filtering.create_filter_rerun_artifact)
    lineage_source = inspect.getsource(assembly._validate_task9_lineage_node)

    for field_name in (
        "max_entities",
        "max_memberships",
        "max_relations",
        "max_links",
    ):
        marker_field = f'"{field_name}"'
        assert marker_field in persist_source
        assert marker_field in marker_validation_source
        assert marker_field in filter_source
        assert marker_field in lineage_source


def test_filter_rerun_duplicate_audits_the_exact_persisted_assembly_config():
    validator_source = inspect.getsource(filtering._validate_existing_filter_rerun)
    creator_source = inspect.getsource(filtering.create_filter_rerun_artifact)

    assert '"assembly_config": _config_payload(source_assembly_config)' in (
        validator_source
    )
    assert "source_assembly_config=source_assembly_config" in creator_source


def test_filter_rerun_authenticates_exact_marker_caps_before_source_rows():
    creator_source = inspect.getsource(filtering.create_filter_rerun_artifact)
    existing_source = inspect.getsource(filtering._validate_existing_filter_rerun)
    compact_creator = " ".join(creator_source.split())
    compact_existing = " ".join(existing_source.split())

    assert creator_source.index("_lock_filter_source_envelope(") < (
        creator_source.index("_filter_inputs_from_artifact(")
    )
    assert "source_config = CollectionResolutionConfig(" in creator_source
    for field_name in (
        "max_entities",
        "max_memberships",
        "max_relations",
        "max_links",
    ):
        assert f'{field_name}=source_caps["{field_name}"]' in compact_creator
    assert "_bounded_query_rows( entity_query, max_entities," in compact_existing
    assert "_bounded_query_rows( link_query, max_links," in compact_existing
