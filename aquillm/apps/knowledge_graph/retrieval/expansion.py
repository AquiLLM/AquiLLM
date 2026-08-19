"""Pure, deterministic ranking over permission-safe graph snapshots."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from hashlib import sha256
from itertools import chain
from math import fsum, isfinite
from typing import cast
from uuid import UUID

from .ppr import (
    MENTION_FACTOR,
    PPRAlgorithmConfig,
    RetrievalDirection,
    StableNodeKey,
    _MonotonicDeadline,
    edge_evidence_flow,
    graph_algorithm_signature,
    personalized_pagerank,
    raw_edge_weight,
)
from .types import (
    GraphExpansionDiagnostics,
    GraphExpansionRequest,
    GraphExpansionResult,
)

_DATABASE_ID_MAX = 2**63 - 1
_CHUNK_NUMBER_MAX = 2**31 - 1
_MAX_EVIDENCE_ROWS = 3_000
_MAX_RETRIEVAL_DIRECTIONS_PER_RELATION = 2
_QUERY_PREDICATE_BATCH_SIZE = 5_000
_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
_RELATION_TYPE_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,127}")


class _SnapshotMiss(Exception):
    """Internal control flow for a valid but unusable authorized snapshot."""


@dataclass(frozen=True, slots=True)
class _RetrievalSnapshotState:
    config: PPRAlgorithmConfig
    deadline: _MonotonicDeadline
    using: str


_ACTIVE_RETRIEVAL_SNAPSHOT: ContextVar[_RetrievalSnapshotState | None] = ContextVar(
    "kg_active_retrieval_snapshot",
    default=None,
)


@dataclass(frozen=True, slots=True)
class _AuthorizedStorageScope:
    """Exact artifact/manifests selected inside one repeatable-read snapshot."""

    document_membership: tuple[tuple[UUID, int], ...]
    collection_artifacts: tuple[dict[str, object], ...]
    manifest_rows: tuple[dict[str, object], ...]
    seed_chunks: tuple[dict[str, object], ...]
    ontology_directions: dict[tuple[int, str], str]

    @property
    def collection_artifact_ids(self) -> tuple[int, ...]:
        return tuple(int(row["id"]) for row in self.collection_artifacts)

    @property
    def document_artifact_ids(self) -> tuple[int, ...]:
        return tuple(int(row["document_artifact_id"]) for row in self.manifest_rows)

    @property
    def manifest_ids(self) -> tuple[int, ...]:
        return tuple(int(row["id"]) for row in self.manifest_rows)


@dataclass(frozen=True, slots=True)
class _AuthorizedEntityRow:
    pk: int
    artifact_id: int
    collection_id: int
    cluster_key: str
    retrieval_utility: float


@dataclass(frozen=True, slots=True)
class _AuthorizedPhysicalRelation:
    pk: int
    artifact_id: int
    source_id: int
    relation_type: str
    target_id: int


@dataclass(frozen=True, slots=True)
class _AuthorizedEvidenceProjection:
    relation_id: int
    evidence_id: int
    relation_mention_id: int
    evidence: AuthorizedChunkEvidence
    semantic_signature: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class _DirectionalPhysicalProjection:
    source_key: StableNodeKey
    relation_type: str
    target_key: StableNodeKey
    direction: RetrievalDirection
    admission_hop: int
    destination_retrieval_utility: float
    evidence: tuple[_AuthorizedEvidenceProjection, ...]


def _load_algorithm_config() -> PPRAlgorithmConfig:
    """Load the effective bounded v1 settings without touching persistence."""

    from django.conf import settings

    from apps.knowledge_graph.resolution.canonical import (
        CANONICAL_RESOLVER_VERSION,
    )

    if getattr(settings, "KG_OVERLAY_ALGORITHM", "ppr_v1") != "ppr_v1":
        raise ValueError("KG_OVERLAY_ALGORITHM must be exactly 'ppr_v1'")
    values = {
        "canonical_resolver_version": getattr(
            settings,
            "KG_CANONICAL_RESOLVER_VERSION",
            CANONICAL_RESOLVER_VERSION,
        ),
        "rrf_k": getattr(settings, "KG_OVERLAY_RRF_K", 60),
        "max_seeds": getattr(settings, "KG_OVERLAY_MAX_SEEDS", 64),
        "max_scope_documents": getattr(
            settings,
            "KG_OVERLAY_MAX_SCOPE_DOCUMENTS",
            10_000,
        ),
        "max_scope_collections": getattr(
            settings,
            "KG_OVERLAY_MAX_SCOPE_COLLECTIONS",
            128,
        ),
        "max_hops": getattr(settings, "KG_OVERLAY_MAX_HOPS", 2),
        "max_fanout": getattr(settings, "KG_OVERLAY_MAX_FANOUT", 10),
        "max_nodes": getattr(settings, "KG_OVERLAY_MAX_NODES", 200),
        "max_edges": getattr(settings, "KG_OVERLAY_MAX_EDGES", 1_000),
        "max_evidence_rows": getattr(
            settings,
            "KG_OVERLAY_MAX_EVIDENCE_ROWS",
            3_000,
        ),
        "max_evidence_per_edge": getattr(
            settings,
            "KG_OVERLAY_MAX_EVIDENCE_PER_EDGE",
            3,
        ),
        "max_mentions_per_entity": getattr(
            settings,
            "KG_OVERLAY_MAX_MENTIONS_PER_ENTITY",
            2,
        ),
        "ppr_restart": getattr(settings, "KG_OVERLAY_PPR_RESTART", 0.20),
        "ppr_iterations": getattr(settings, "KG_OVERLAY_PPR_ITERATIONS", 8),
        "max_candidates": getattr(settings, "KG_OVERLAY_MAX_CANDIDATES", 20),
        "max_per_document": getattr(
            settings,
            "KG_OVERLAY_MAX_PER_DOCUMENT",
            3,
        ),
        "timeout_ms": getattr(settings, "KG_OVERLAY_TIMEOUT_MS", 150),
    }
    return PPRAlgorithmConfig(**values)


def _query_batches(values: Sequence[object]) -> Iterator[tuple[object, ...]]:
    for start in range(0, len(values), _QUERY_PREDICATE_BATCH_SIZE):
        yield tuple(values[start : start + _QUERY_PREDICATE_BATCH_SIZE])


def _batched_in_q(field_name: str, values: Sequence[object]):
    """Build an OR of bounded ``IN`` predicates without an unbounded clause."""

    from django.db.models import Q

    predicate = Q(pk__in=())
    for batch in _query_batches(values):
        predicate |= Q(**{f"{field_name}__in": batch})
    return predicate


def _validate_scope_membership(
    request: GraphExpansionRequest,
    observed_rows: Iterable[tuple[UUID, int]],
) -> tuple[tuple[UUID, int], ...]:
    """Require one concrete document row and the exact declared collections."""

    rows: list[tuple[UUID, int]] = []
    try:
        for raw_row in observed_rows:
            if type(raw_row) is not tuple or len(raw_row) != 2:
                raise _SnapshotMiss
            document_id, collection_id = raw_row
            if type(document_id) is not UUID:
                raise _SnapshotMiss
            rows.append(
                (
                    document_id,
                    _positive_database_int(collection_id, "scope collection_id"),
                )
            )
    except (TypeError, ValueError) as error:
        raise _SnapshotMiss from error
    document_ids = tuple(row[0] for row in rows)
    if len(document_ids) != len(set(document_ids)):
        raise _SnapshotMiss
    if set(document_ids) != set(request.allowed_doc_ids):
        raise _SnapshotMiss
    if {row[1] for row in rows} != set(request.allowed_collection_ids):
        raise _SnapshotMiss
    return tuple(sorted(rows, key=lambda row: row[0].int))


def _matching_observation_chunks(
    *,
    representative_chunk_id: int,
    metadata: object,
    candidate_chunk_ids: tuple[int, ...],
) -> tuple[int, ...]:
    """Decode bounded seed membership from representative and observation rows."""

    representative = _positive_database_int(
        representative_chunk_id,
        "representative_chunk_id",
    )
    candidates = frozenset(
        _positive_database_int(value, "candidate_chunk_ids")
        for value in candidate_chunk_ids
    )
    matches = {representative}.intersection(candidates)
    if type(metadata) is not dict:
        raise _SnapshotMiss
    observations = metadata.get("observations", ())
    if type(observations) is not list:
        raise _SnapshotMiss
    for observation in observations:
        if type(observation) is not dict:
            raise _SnapshotMiss
        chunk_id = observation.get("chunk_id")
        if type(chunk_id) is not int or not 1 <= chunk_id <= _DATABASE_ID_MAX:
            raise _SnapshotMiss
        if chunk_id in candidates:
            matches.add(chunk_id)
    return tuple(sorted(matches))


def _directions_from_frontier(
    source_key: StableNodeKey,
    target_key: StableNodeKey,
    *,
    ontology_direction: str,
    frontier: frozenset[StableNodeKey],
) -> tuple[tuple[StableNodeKey, StableNodeKey, RetrievalDirection], ...]:
    """Admit only retrieval directions whose source is in this frontier."""

    source = _identity_key(source_key, "source_key")
    target = _identity_key(target_key, "target_key")
    if type(frontier) is not frozenset or any(
        _identity_key(value, "frontier") != value for value in frontier
    ):
        raise ValueError("frontier must be an exact identity-key frozenset")
    if ontology_direction not in {"directed", "undirected"}:
        raise _SnapshotMiss
    directions: list[
        tuple[StableNodeKey, StableNodeKey, RetrievalDirection]
    ] = []
    if source in frontier:
        directions.append(
            (
                source,
                target,
                (
                    RetrievalDirection.FORWARD
                    if ontology_direction == "directed"
                    else RetrievalDirection.UNDIRECTED
                ),
            )
        )
    if target in frontier:
        directions.append(
            (
                target,
                source,
                (
                    RetrievalDirection.REVERSE_DIRECTED
                    if ontology_direction == "directed"
                    else RetrievalDirection.UNDIRECTED
                ),
            )
        )
    return tuple(directions)


def _select_fallback_rows(
    rows: Iterable[tuple[object, ...]],
    *,
    maximum_per_identity: int,
) -> dict[StableNodeKey, tuple[tuple[object, ...], ...]]:
    """Pure global confidence-first selection after SQL chunk projection."""

    by_identity: dict[StableNodeKey, list[tuple[object, ...]]] = defaultdict(list)
    for raw_row in rows:
        if type(raw_row) is not tuple or len(raw_row) != 7:
            raise _SnapshotMiss
        kind = cast(str, raw_row[0])
        value: object = (
            int(raw_row[1]) if kind == "canonical" else cast(str, raw_row[1])
        )
        identity = _identity_key((kind, value))
        by_identity[identity].append(raw_row)
    return {
        identity: tuple(
            sorted(
                identity_rows,
                key=lambda row: (
                    -float(row[5]),
                    cast(UUID, row[3]).int,
                    int(row[4]),
                    int(row[2]),
                    int(row[6]),
                ),
            )[:maximum_per_identity]
        )
        for identity, identity_rows in sorted(by_identity.items())
    }


def _require_live_snapshot_state() -> _RetrievalSnapshotState:
    state = _ACTIVE_RETRIEVAL_SNAPSHOT.get()
    if state is None:
        raise RuntimeError("loader requires a live authorized snapshot context")
    from django.db import connections

    connection = connections[state.using]
    if connection.vendor != "postgresql" or not connection.in_atomic_block:
        raise RuntimeError("authorized retrieval snapshot context is no longer live")
    state.deadline.check()
    return state


def _bounded_values(
    queryset: object,
    fields: tuple[str, ...],
    *,
    maximum: int,
    state: _RetrievalSnapshotState,
) -> tuple[dict[str, object], ...]:
    """Materialize at most ``maximum`` ORM rows, always using cap+1."""

    if type(maximum) is not int or maximum < 0:
        raise ValueError("query maximum must be a nonnegative exact integer")
    state.deadline.check()
    values_query = queryset.values(*fields).order_by("pk")[: maximum + 1]
    rows = list(values_query)
    state.deadline.check()
    if len(rows) > maximum:
        raise _SnapshotMiss
    return tuple(rows)


def _load_document_membership(
    request: GraphExpansionRequest,
    state: _RetrievalSnapshotState,
) -> tuple[tuple[UUID, int], ...]:
    """Resolve every logical UUID through exactly one concrete document table."""

    from apps.documents.models import DESCENDED_FROM_DOCUMENT

    observed: list[tuple[UUID, int]] = []
    for document_model in sorted(
        DESCENDED_FROM_DOCUMENT,
        key=lambda model: model._meta.label_lower,
    ):
        for raw_batch in _query_batches(request.allowed_doc_ids):
            state.deadline.check()
            batch = cast(tuple[UUID, ...], raw_batch)
            remaining = len(request.allowed_doc_ids) - len(observed)
            if remaining < 0:
                raise _SnapshotMiss
            rows = list(
                document_model._base_manager.using(state.using)
                .filter(id__in=batch)
                .order_by("id")
                .values_list("id", "collection_id")[: remaining + 1]
            )
            state.deadline.check()
            if len(rows) > remaining:
                raise _SnapshotMiss
            observed.extend(rows)
    return _validate_scope_membership(request, observed)


def _load_storage_scope(
    request: GraphExpansionRequest,
    state: _RetrievalSnapshotState,
) -> _AuthorizedStorageScope:
    """Load the exact active C/D artifact snapshot and revalidate its ontology."""

    from django.db.models import F, Q

    from apps.documents.models import TextChunk
    from apps.knowledge_graph.models import (
        CollectionArtifactInput,
        GraphArtifact,
        OntologyVersion,
    )
    from apps.knowledge_graph.services.ontology import load_ontology_yaml

    membership = _load_document_membership(request, state)
    collection_artifacts = _bounded_values(
        GraphArtifact.objects.using(state.using).filter(
            scope_type=GraphArtifact.ScopeType.COLLECTION,
            scope_id__in=tuple(str(value) for value in request.allowed_collection_ids),
            collection_scope_id__in=request.allowed_collection_ids,
            status=GraphArtifact.Status.ACTIVE,
            evaluation_only=False,
        ),
        (
            "id",
            "scope_id",
            "collection_scope_id",
            "build_key",
            "source_hash",
            "ontology_version",
            "ontology_checksum",
            "resolver_version",
            "assembly_version",
            "assembly_config_checksum",
        ),
        maximum=state.config.max_scope_collections,
        state=state,
    )
    selected_collection_ids = tuple(
        int(row["collection_scope_id"]) for row in collection_artifacts
    )
    if len(set(selected_collection_ids)) != len(selected_collection_ids):
        raise _SnapshotMiss
    if any(
        str(row["collection_scope_id"]) != row["scope_id"]
        for row in collection_artifacts
    ):
        raise _SnapshotMiss

    artifact_ids = tuple(int(row["id"]) for row in collection_artifacts)
    manifest_fields = (
        "id",
        "artifact_id",
        "collection_id",
        "document_id",
        "document_artifact_id",
        "source_signature",
        "membership_signature",
        "build_signature",
        "document_artifact__build_key",
        "document_artifact__scope_id",
        "document_artifact__source_hash",
        "document_artifact__resolver_version",
        "document_artifact__ontology_version",
        "document_artifact__ontology_checksum",
    )
    manifest_accumulator: list[dict[str, object]] = []
    for raw_batch in _query_batches(request.allowed_doc_ids):
        remaining = state.config.max_scope_documents - len(manifest_accumulator)
        rows = _bounded_values(
            CollectionArtifactInput.objects.using(state.using).filter(
                artifact_id__in=artifact_ids,
                artifact__status=GraphArtifact.Status.ACTIVE,
                collection_id__in=request.allowed_collection_ids,
                document_id__in=cast(tuple[UUID, ...], raw_batch),
                artifact__collection_scope_id=F("collection_id"),
                document_artifact__status=GraphArtifact.Status.ACTIVE,
                document_artifact__scope_type=GraphArtifact.ScopeType.DOCUMENT,
                document_artifact__evaluation_only=False,
            ),
            manifest_fields,
            maximum=remaining,
            state=state,
        )
        manifest_accumulator.extend(rows)
    manifest_rows = tuple(sorted(manifest_accumulator, key=lambda row: int(row["id"])))
    membership_by_document = dict(membership)
    manifest_scope = tuple(
        (cast(UUID, row["document_id"]), int(row["collection_id"]))
        for row in manifest_rows
    )
    if len({row[0] for row in manifest_scope}) != len(manifest_scope):
        raise _SnapshotMiss
    if any(
        membership_by_document.get(document_id) != collection_id
        for document_id, collection_id in manifest_scope
    ):
        raise _SnapshotMiss
    artifact_by_collection = {
        int(row["collection_scope_id"]): int(row["id"])
        for row in collection_artifacts
    }
    if any(
        artifact_by_collection.get(int(row["collection_id"]))
        != int(row["artifact_id"])
        for row in manifest_rows
    ):
        raise _SnapshotMiss
    if any(
        cast(str, row["document_artifact__scope_id"])
        != str(row["document_id"])
        for row in manifest_rows
    ):
        raise _SnapshotMiss

    seed_ids = tuple(seed.chunk_id for seed in request.seeds)
    seed_chunks = _bounded_values(
        TextChunk.objects.using(state.using).filter(
            _batched_in_q("doc_id", request.allowed_doc_ids),
            pk__in=seed_ids,
        ),
        ("id", "doc_id", "chunk_number"),
        maximum=state.config.max_seeds,
        state=state,
    )
    if {int(row["id"]) for row in seed_chunks} != set(seed_ids):
        raise _SnapshotMiss
    if any(row["doc_id"] not in set(request.allowed_doc_ids) for row in seed_chunks):
        raise _SnapshotMiss

    artifact_identity = {
        int(row["id"]): (
            row["ontology_version"],
            row["ontology_checksum"],
        )
        for row in collection_artifacts
    }
    if any(
        artifact_identity[int(row["artifact_id"])]
        != (
            row["document_artifact__ontology_version"],
            row["document_artifact__ontology_checksum"],
        )
        for row in manifest_rows
    ):
        raise _SnapshotMiss

    ontology_directions: dict[tuple[int, str], str] = {}
    ontology_cache: dict[tuple[str, str], object] = {}
    ontology_keys = tuple(
        sorted(
            {
                (
                    cast(str, artifact["ontology_version"]),
                    cast(str, artifact["ontology_checksum"]),
                )
                for artifact in collection_artifacts
            }
        )
    )
    ontology_predicate = Q(pk__in=())
    for version, checksum in ontology_keys:
        ontology_predicate |= Q(version=version, checksum=checksum)
    ontology_rows = _bounded_values(
        OntologyVersion.objects.using(state.using).filter(
            ontology_predicate,
            kind=OntologyVersion.Kind.GRAPH,
            status__in=(
                OntologyVersion.Status.ACTIVE,
                OntologyVersion.Status.SUPERSEDED,
            ),
        ),
        ("id", "version", "checksum", "status", "metadata"),
        maximum=len(ontology_keys),
        state=state,
    )
    if {
        (cast(str, row["version"]), cast(str, row["checksum"]))
        for row in ontology_rows
    } != set(ontology_keys):
        raise _SnapshotMiss
    for row in ontology_rows:
        raw_yaml = row["metadata"]
        if type(raw_yaml) is not dict or type(raw_yaml.get("yaml")) is not str:
            raise _SnapshotMiss
        definition = load_ontology_yaml(raw_yaml["yaml"])
        key = (cast(str, row["version"]), cast(str, row["checksum"]))
        if definition.version != key[0] or definition.checksum != key[1]:
            raise _SnapshotMiss
        ontology_cache[key] = definition
    for artifact in collection_artifacts:
        version = cast(str, artifact["ontology_version"])
        checksum = cast(str, artifact["ontology_checksum"])
        cache_key = (version, checksum)
        definition = ontology_cache.get(cache_key)
        if definition is None:
            raise _SnapshotMiss
        artifact_id = int(artifact["id"])
        for relation_type, relation_definition in definition.relations.items():
            ontology_directions[(artifact_id, relation_type)] = (
                relation_definition.direction
            )

    return _AuthorizedStorageScope(
        document_membership=membership,
        collection_artifacts=collection_artifacts,
        manifest_rows=manifest_rows,
        seed_chunks=seed_chunks,
        ontology_directions=ontology_directions,
    )


def _execute_bounded_sql(
    sql: str,
    parameters: Sequence[object],
    *,
    maximum: int,
    state: _RetrievalSnapshotState,
) -> tuple[tuple[object, ...], ...]:
    """Execute one private PostgreSQL projection with a hard fetchmany cap."""

    from django.db import connections

    state.deadline.check()
    connection = connections[state.using]
    with connection.cursor() as cursor:
        cursor.execute(sql, list(parameters))
        rows = cursor.fetchmany(maximum + 2)
    state.deadline.check()
    if len(rows) > maximum:
        raise _SnapshotMiss
    return tuple(tuple(row) for row in rows)


def _load_seed_collection_entity_ids(
    request: GraphExpansionRequest,
    scope: _AuthorizedStorageScope,
    state: _RetrievalSnapshotState,
) -> tuple[tuple[int, int], ...]:
    """Project representative/observation seed matches without decoding JSON."""

    from django.db import connections

    from apps.knowledge_graph.models import (
        CollectionArtifactInput,
        CollectionEntity,
        CollectionEntityDocumentLink,
        DocumentEntity,
        DocumentEntityMention,
        EntityMention,
        GraphArtifact,
    )

    connection = connections[state.using]
    quote = connection.ops.quote_name
    seed_by_id = {
        int(row["id"]): cast(UUID, row["doc_id"]) for row in scope.seed_chunks
    }
    seed_values = ", ".join("(%s::bigint, %s::uuid)" for _ in seed_by_id)
    parameters: list[object] = []
    for seed_id in sorted(seed_by_id):
        parameters.extend((seed_id, seed_by_id[seed_id]))
    seed_documents = frozenset(seed_by_id.values())
    seed_manifest_ids = tuple(
        int(row["id"])
        for row in scope.manifest_rows
        if row["document_id"] in seed_documents
    )
    parameters.extend(
        (
            list(scope.collection_artifact_ids),
            list(seed_manifest_ids),
            list(request.allowed_collection_ids),
            list(sorted(seed_documents, key=lambda value: value.int)),
            state.config.max_nodes + 1,
        )
    )
    tables = {
        "link": quote(CollectionEntityDocumentLink._meta.db_table),
        "manifest": quote(CollectionArtifactInput._meta.db_table),
        "artifact": quote(GraphArtifact._meta.db_table),
        "entity": quote(CollectionEntity._meta.db_table),
        "document_entity": quote(DocumentEntity._meta.db_table),
        "document_link": quote(DocumentEntityMention._meta.db_table),
        "mention": quote(EntityMention._meta.db_table),
    }
    sql = f"""
        WITH seed_scope(seed_chunk_id, document_id) AS (VALUES {seed_values})
        SELECT seed_scope.seed_chunk_id, link.collection_entity_id,
               link.id, document_entity_mention.id, entity_mention.id
        FROM {tables['link']} AS link
        JOIN {tables['artifact']} AS collection_artifact
          ON collection_artifact.id = link.artifact_id
        JOIN {tables['manifest']} AS manifest_input
          ON manifest_input.id = link.manifest_input_id
        JOIN {tables['artifact']} AS document_artifact
          ON document_artifact.id = manifest_input.document_artifact_id
        JOIN {tables['document_entity']} AS document_entity
          ON document_entity.id = link.document_entity_id
        JOIN {tables['document_link']} AS document_entity_mention
          ON document_entity_mention.document_entity_id = document_entity.id
        JOIN {tables['mention']} AS entity_mention
          ON entity_mention.id = document_entity_mention.mention_id
        JOIN {tables['entity']} AS collection_entity
          ON collection_entity.id = link.collection_entity_id
        JOIN seed_scope
          ON seed_scope.document_id = entity_mention.document_id
         AND (
              entity_mention.chunk_id = seed_scope.seed_chunk_id
              OR entity_mention.metadata @> jsonb_build_object(
                    'observations', jsonb_build_array(
                        jsonb_build_object('chunk_id', seed_scope.seed_chunk_id)
                    )
                 )
         )
        WHERE link.artifact_id = ANY(%s::bigint[])
          AND link.manifest_input_id = ANY(%s::bigint[])
          AND manifest_input.collection_id = ANY(%s::bigint[])
          AND manifest_input.document_id = ANY(%s::uuid[])
          AND collection_artifact.status = 'active'
          AND collection_artifact.scope_type = 'collection'
          AND collection_artifact.collection_scope_id = manifest_input.collection_id
          AND collection_artifact.scope_id = manifest_input.collection_id::text
          AND link.status = 'active'
          AND link.outcome = 'automatic'
          AND link.resolver_version = collection_artifact.resolver_version
          AND link.artifact_id = manifest_input.artifact_id
          AND link.artifact_id = collection_entity.artifact_id
          AND collection_entity.status = 'active'
          AND collection_entity.collection_id = manifest_input.collection_id
          AND document_artifact.status = 'active'
          AND document_artifact.scope_type = 'document'
          AND document_artifact.scope_id = manifest_input.document_id::text
          AND document_entity.status = 'active'
          AND document_entity.artifact_id = manifest_input.document_artifact_id
          AND document_entity.document_id = manifest_input.document_id
          AND document_entity_mention.status = 'active'
          AND document_entity_mention.resolver_version =
              document_artifact.resolver_version
          AND entity_mention.artifact_id = document_artifact.id
          AND entity_mention.document_id = manifest_input.document_id
        ORDER BY seed_scope.seed_chunk_id, link.collection_entity_id,
                 link.id, document_entity_mention.id, entity_mention.id
        LIMIT %s
    """
    rows = _execute_bounded_sql(
        sql,
        parameters,
        maximum=state.config.max_nodes,
        state=state,
    )
    projected = tuple(sorted({(int(row[0]), int(row[1])) for row in rows}))
    if not projected:
        raise _SnapshotMiss
    return projected


def _permission_membership_subquery(
    *,
    scope: _AuthorizedStorageScope,
    request: GraphExpansionRequest,
    state: _RetrievalSnapshotState,
    collection_entity_outer_ref: str,
):
    """Return a current exact-scope permission proof for one collection entity."""

    from django.db.models import OuterRef

    from apps.knowledge_graph.models import CollectionEntityDocumentLink

    query = CollectionEntityDocumentLink.objects.using(state.using).current().filter(
        collection_entity_id=OuterRef(collection_entity_outer_ref),
        artifact_id__in=scope.collection_artifact_ids,
        manifest_input__collection_id__in=request.allowed_collection_ids,
    )
    return query.filter(
        _batched_in_q("manifest_input_id", scope.manifest_ids),
        _batched_in_q(
            "manifest_input__document_artifact_id",
            scope.document_artifact_ids,
        ),
        _batched_in_q(
            "manifest_input__document_id",
            request.allowed_doc_ids,
        ),
    )


def _load_authorized_entity_rows(
    entity_ids: tuple[int, ...],
    *,
    request: GraphExpansionRequest,
    scope: _AuthorizedStorageScope,
    state: _RetrievalSnapshotState,
    maximum: int,
) -> tuple[_AuthorizedEntityRow, ...]:
    """Load candidate-scoped permission-bearing entity endpoints once."""

    from django.db.models import Exists

    from apps.knowledge_graph.models import CollectionEntity

    if not entity_ids:
        return ()
    query = (
        CollectionEntity.objects.using(state.using)
        .current()
        .filter(
            pk__in=entity_ids,
            artifact_id__in=scope.collection_artifact_ids,
            collection_id__in=request.allowed_collection_ids,
        )
        .annotate(
            permission_membership=Exists(
                _permission_membership_subquery(
                    scope=scope,
                    request=request,
                    state=state,
                    collection_entity_outer_ref="pk",
                )
            )
        )
        .filter(permission_membership=True)
    )
    rows = _bounded_values(
        query,
        (
            "id",
            "artifact_id",
            "collection_id",
            "cluster_key",
            "retrieval_utility",
        ),
        maximum=maximum,
        state=state,
    )
    loaded = tuple(
        _AuthorizedEntityRow(
            pk=int(row["id"]),
            artifact_id=int(row["artifact_id"]),
            collection_id=int(row["collection_id"]),
            cluster_key=cast(str, row["cluster_key"]),
            retrieval_utility=float(row["retrieval_utility"]),
        )
        for row in rows
    )
    if {row.pk for row in loaded} != set(entity_ids):
        raise _SnapshotMiss
    return loaded


def _project_identity_keys(
    entity_rows: tuple[_AuthorizedEntityRow, ...],
    canonical_memberships: tuple[tuple[int, int], ...],
    *,
    existing: tuple[tuple[int, StableNodeKey], ...] = (),
) -> dict[int, StableNodeKey]:
    """Pure deterministic canonical-peer collapse over validated storage rows."""

    result = dict(existing)
    canonical_by_entity: dict[int, int] = {}
    for entity_id, canonical_id in canonical_memberships:
        _positive_database_int(entity_id, "canonical membership entity")
        _positive_database_int(canonical_id, "canonical membership target")
        previous = canonical_by_entity.setdefault(entity_id, canonical_id)
        if previous != canonical_id:
            raise _SnapshotMiss
    for entity in sorted(entity_rows, key=lambda row: row.pk):
        canonical_id = canonical_by_entity.get(entity.pk)
        identity: StableNodeKey = (
            ("canonical", canonical_id)
            if canonical_id is not None
            else ("local", entity.cluster_key)
        )
        previous = result.setdefault(entity.pk, identity)
        if previous != identity:
            raise _SnapshotMiss
    return result


def _extend_identity_registry(
    requested_entity_ids: tuple[int, ...],
    *,
    request: GraphExpansionRequest,
    scope: _AuthorizedStorageScope,
    state: _RetrievalSnapshotState,
    entities: dict[int, _AuthorizedEntityRow],
    identity_by_entity: dict[int, StableNodeKey],
    canonical_audit: dict[int, tuple[object, ...]],
    discovery_hop: int,
) -> None:
    """Add exact endpoints plus authorized canonical peers under raw-node caps."""

    from django.db.models import Exists

    from apps.knowledge_graph.models import CanonicalEntityLink

    missing = tuple(sorted(set(requested_entity_ids).difference(entities)))
    remaining = state.config.max_nodes - len(entities)
    if len(missing) > remaining:
        raise _SnapshotMiss
    for row in _load_authorized_entity_rows(
        missing,
        request=request,
        scope=scope,
        state=state,
        maximum=remaining,
    ):
        entities[row.pk] = row

    if not missing:
        return
    direct_links = _bounded_values(
        CanonicalEntityLink.objects.using(state.using)
        .current(resolver_version=state.config.canonical_resolver_version)
        .filter(
            collection_entity_id__in=missing,
            collection_entity__artifact_id__in=scope.collection_artifact_ids,
        ),
        (
            "id",
            "collection_entity_id",
            "canonical_entity_id",
            "decision_checksum",
            "resolver_version",
        ),
        maximum=len(missing),
        state=state,
    )
    canonical_by_entity: dict[int, int] = {
        entity_id: int(identity[1])
        for entity_id, identity in identity_by_entity.items()
        if identity[0] == "canonical"
    }
    for link in direct_links:
        entity_id = int(link["collection_entity_id"])
        canonical_id = int(link["canonical_entity_id"])
        previous = canonical_by_entity.setdefault(entity_id, canonical_id)
        if previous != canonical_id:
            raise _SnapshotMiss
        audit_value = (
            discovery_hop,
            int(link["id"]),
            entity_id,
            canonical_id,
            link["decision_checksum"],
            link["resolver_version"],
        )
        previous_audit = canonical_audit.setdefault(int(link["id"]), audit_value)
        if previous_audit[1:] != audit_value[1:]:
            raise _SnapshotMiss

    canonical_ids = tuple(sorted(set(canonical_by_entity.values())))
    peer_rows: tuple[dict[str, object], ...] = ()
    if canonical_ids:
        peer_query = (
            CanonicalEntityLink.objects.using(state.using)
            .current(resolver_version=state.config.canonical_resolver_version)
            .filter(canonical_entity_id__in=canonical_ids)
            .annotate(
                permission_membership=Exists(
                    _permission_membership_subquery(
                        scope=scope,
                        request=request,
                        state=state,
                        collection_entity_outer_ref="collection_entity_id",
                    )
                )
            )
            .filter(permission_membership=True)
        )
        peer_rows = _bounded_values(
            peer_query,
            (
                "id",
                "collection_entity_id",
                "canonical_entity_id",
                "decision_checksum",
                "resolver_version",
            ),
            maximum=state.config.max_nodes,
            state=state,
        )
        peer_entity_ids = tuple(
            sorted({int(row["collection_entity_id"]) for row in peer_rows})
        )
        peer_missing = tuple(
            value for value in peer_entity_ids if value not in entities
        )
        remaining = state.config.max_nodes - len(entities)
        if len(peer_missing) > remaining:
            raise _SnapshotMiss
        for row in _load_authorized_entity_rows(
            peer_missing,
            request=request,
            scope=scope,
            state=state,
            maximum=remaining,
        ):
            entities[row.pk] = row
        for peer in peer_rows:
            entity_id = int(peer["collection_entity_id"])
            canonical_id = int(peer["canonical_entity_id"])
            previous = canonical_by_entity.setdefault(entity_id, canonical_id)
            if previous != canonical_id:
                raise _SnapshotMiss
            audit_value = (
                discovery_hop,
                int(peer["id"]),
                entity_id,
                canonical_id,
                peer["decision_checksum"],
                peer["resolver_version"],
            )
            previous_audit = canonical_audit.setdefault(
                int(peer["id"]), audit_value
            )
            if previous_audit[1:] != audit_value[1:]:
                raise _SnapshotMiss

    assignable_entity_ids = set(missing)
    assignable_entity_ids.update(
        int(peer["collection_entity_id"]) for peer in peer_rows
    )
    projected = _project_identity_keys(
        tuple(entities[entity_id] for entity_id in sorted(assignable_entity_ids)),
        tuple(sorted(canonical_by_entity.items())),
        existing=tuple(sorted(identity_by_entity.items())),
    )
    identity_by_entity.clear()
    identity_by_entity.update(projected)


def _authorized_evidence_queryset(
    *,
    request: GraphExpansionRequest,
    scope: _AuthorizedStorageScope,
    state: _RetrievalSnapshotState,
    relation_id: object | None = None,
):
    """Build the exact current, authorized evidence boundary for physical edges."""

    from django.db.models import F, Q

    from apps.knowledge_graph.models import (
        CollectionRelationEvidence,
        GraphArtifact,
    )
    from apps.knowledge_graph.models.associations import _current_link_filters
    from apps.knowledge_graph.models.entities import ResolutionStatus

    query = CollectionRelationEvidence.objects.using(state.using).filter(
        artifact_id__in=scope.collection_artifact_ids,
        artifact__status=GraphArtifact.Status.ACTIVE,
        status=CollectionRelationEvidence.Status.ACTIVE,
        relation__isnull=False,
        relation__artifact_id=F("artifact_id"),
        relation__status=ResolutionStatus.ACTIVE,
        relation__source__status=ResolutionStatus.ACTIVE,
        relation__target__status=ResolutionStatus.ACTIVE,
        relation__source__artifact_id=F("artifact_id"),
        relation__target__artifact_id=F("artifact_id"),
        ontology_checksum=F("artifact__ontology_checksum"),
        assembly_config_checksum=F("artifact__assembly_config_checksum"),
        relation_mention__artifact__status=GraphArtifact.Status.ACTIVE,
        relation_mention__artifact__scope_type=GraphArtifact.ScopeType.DOCUMENT,
        relation_mention__relation_type=F("relation__relation_type"),
        relation_mention__chunk__doc_id=F("relation_mention__document_id"),
        relation_mention__head__artifact_id=F("relation_mention__artifact_id"),
        relation_mention__tail__artifact_id=F("relation_mention__artifact_id"),
        relation_mention__head__document_id=F("relation_mention__document_id"),
        relation_mention__tail__document_id=F("relation_mention__document_id"),
        head_mapping__document_entity__mention_links__status=ResolutionStatus.ACTIVE,
        tail_mapping__document_entity__mention_links__status=ResolutionStatus.ACTIVE,
        head_mapping__document_entity__mention_links__mention_id=F(
            "relation_mention__head_id"
        ),
        tail_mapping__document_entity__mention_links__mention_id=F(
            "relation_mention__tail_id"
        ),
        head_mapping__document_entity__mention_links__resolver_version=F(
            "head_mapping__document_entity__artifact__resolver_version"
        ),
        tail_mapping__document_entity__mention_links__resolver_version=F(
            "tail_mapping__document_entity__artifact__resolver_version"
        ),
        **_current_link_filters("head_mapping"),
        **_current_link_filters("tail_mapping"),
    ).filter(
        Q(
            relation__source__collection_id=F(
                "relation__target__collection_id"
            )
        ),
        Q(
            relation__source__collection_id=F(
                "artifact__collection_scope_id"
            )
        ),
        Q(
            head_mapping__manifest_input__document_artifact_id=F(
                "relation_mention__artifact_id"
            )
        ),
        Q(
            tail_mapping__manifest_input__document_artifact_id=F(
                "relation_mention__artifact_id"
            )
        ),
        Q(
            head_mapping__manifest_input__document_id=F(
                "relation_mention__document_id"
            )
        ),
        Q(
            tail_mapping__manifest_input__document_id=F(
                "relation_mention__document_id"
            )
        ),
        _batched_in_q(
            "relation_mention__artifact_id",
            scope.document_artifact_ids,
        ),
        _batched_in_q("head_mapping__manifest_input_id", scope.manifest_ids),
        _batched_in_q("tail_mapping__manifest_input_id", scope.manifest_ids),
        _batched_in_q(
            "head_mapping__manifest_input__document_id",
            request.allowed_doc_ids,
        ),
        _batched_in_q(
            "tail_mapping__manifest_input__document_id",
            request.allowed_doc_ids,
        ),
        Q(
            head_mapping__manifest_input__collection_id__in=(
                request.allowed_collection_ids
            ),
            tail_mapping__manifest_input__collection_id__in=(
                request.allowed_collection_ids
            ),
        ),
    )
    if relation_id is not None:
        query = query.filter(relation_id=relation_id)
    orientation = (
        Q(
            orientation=CollectionRelationEvidence.Orientation.HEAD_TO_TAIL,
            head_mapping__collection_entity_id=F("relation__source_id"),
            tail_mapping__collection_entity_id=F("relation__target_id"),
        )
        | Q(
            orientation=CollectionRelationEvidence.Orientation.TAIL_TO_HEAD,
            head_mapping__collection_entity_id=F("relation__target_id"),
            tail_mapping__collection_entity_id=F("relation__source_id"),
        )
    )
    return query.filter(orientation).distinct()


def _load_physical_relations(
    frontier_entity_ids: tuple[int, ...],
    loaded_relation_ids: tuple[int, ...],
    *,
    request: GraphExpansionRequest,
    scope: _AuthorizedStorageScope,
    state: _RetrievalSnapshotState,
    maximum: int,
) -> tuple[_AuthorizedPhysicalRelation, ...]:
    """Load only physical relations with current authorized evidence."""

    from django.db.models import Exists, F, OuterRef, Q

    from apps.knowledge_graph.models import (
        CollectionRelation,
        GraphArtifact,
    )
    from apps.knowledge_graph.models.entities import ResolutionStatus

    if not frontier_entity_ids or maximum < 0:
        return ()
    evidence_exists = _authorized_evidence_queryset(
        request=request,
        scope=scope,
        state=state,
        relation_id=OuterRef("pk"),
    )
    query = CollectionRelation.objects.using(state.using).filter(
        Exists(evidence_exists),
        artifact_id__in=scope.collection_artifact_ids,
        artifact__status=GraphArtifact.Status.ACTIVE,
        status=ResolutionStatus.ACTIVE,
        source__status=ResolutionStatus.ACTIVE,
        target__status=ResolutionStatus.ACTIVE,
        source__artifact_id=F("artifact_id"),
        target__artifact_id=F("artifact_id"),
    ).filter(
        Q(source__collection_id=F("target__collection_id")),
        Q(source__collection_id=F("artifact__collection_scope_id")),
        Q(source_id__in=frontier_entity_ids)
        | Q(target_id__in=frontier_entity_ids)
    )
    if loaded_relation_ids:
        query = query.exclude(pk__in=loaded_relation_ids)
    rows = _bounded_values(
        query,
        ("id", "artifact_id", "source_id", "relation_type", "target_id"),
        maximum=maximum,
        state=state,
    )
    return tuple(
        _AuthorizedPhysicalRelation(
            pk=int(row["id"]),
            artifact_id=int(row["artifact_id"]),
            source_id=int(row["source_id"]),
            relation_type=cast(str, row["relation_type"]),
            target_id=int(row["target_id"]),
        )
        for row in rows
    )


def _load_authorized_relation_evidence(
    relation_ids: tuple[int, ...],
    *,
    request: GraphExpansionRequest,
    scope: _AuthorizedStorageScope,
    state: _RetrievalSnapshotState,
    maximum: int,
) -> tuple[_AuthorizedEvidenceProjection, ...]:
    """Load bounded evidence and recompute only authorized semantic support."""

    if not relation_ids:
        return ()
    query = _authorized_evidence_queryset(
        request=request,
        scope=scope,
        state=state,
    ).filter(relation_id__in=relation_ids)
    fields = (
        "id",
        "relation_id",
        "relation_mention_id",
        "relation_mention__chunk_id",
        "relation_mention__chunk__doc_id",
        "relation_mention__chunk__chunk_number",
        "relation_mention__extraction_confidence",
        "relation_mention__artifact_id",
        "relation_mention__document_id",
        "relation_mention__head_id",
        "relation_mention__tail_id",
        "relation_mention__relation_type",
        "head_mapping_id",
        "tail_mapping_id",
        "orientation",
        "ontology_checksum",
        "assembly_config_checksum",
    )
    rows = _bounded_values(
        query,
        fields,
        maximum=maximum,
        state=state,
    )
    projected: list[_AuthorizedEvidenceProjection] = []
    seen: set[tuple[int, int]] = set()
    for row in rows:
        relation_id = int(row["relation_id"])
        mention_id = int(row["relation_mention_id"])
        if (relation_id, mention_id) in seen:
            continue
        seen.add((relation_id, mention_id))
        semantic_signature = tuple(row[field] for field in fields)
        encoded = json.dumps(
            semantic_signature,
            default=str,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        provenance_key = sha256(encoded).hexdigest()
        evidence = AuthorizedChunkEvidence(
            chunk_id=int(row["relation_mention__chunk_id"]),
            document_id=cast(UUID, row["relation_mention__chunk__doc_id"]),
            chunk_number=int(row["relation_mention__chunk__chunk_number"]),
            confidence=float(row["relation_mention__extraction_confidence"]),
            provenance_key=provenance_key,
        )
        projected.append(
            _AuthorizedEvidenceProjection(
                relation_id=relation_id,
                evidence_id=int(row["id"]),
                relation_mention_id=mention_id,
                evidence=evidence,
                semantic_signature=semantic_signature,
            )
        )
    if {row.relation_id for row in projected} != set(relation_ids):
        raise _SnapshotMiss
    return tuple(projected)


def _load_authorized_identity_mentions(
    *,
    request: GraphExpansionRequest,
    scope: _AuthorizedStorageScope,
    state: _RetrievalSnapshotState,
    entities: dict[int, _AuthorizedEntityRow],
    identity_by_entity: dict[int, StableNodeKey],
) -> tuple[AuthorizedIdentityMention, ...]:
    """Project representative and observation chunks with DB-side identity caps."""

    from django.db import connections

    from apps.documents.models import TextChunk
    from apps.knowledge_graph.models import (
        CollectionArtifactInput,
        CollectionEntity,
        CollectionEntityDocumentLink,
        DocumentEntity,
        DocumentEntityMention,
        EntityMention,
        GraphArtifact,
    )

    if not identity_by_entity:
        return ()
    connection = connections[state.using]
    quote = connection.ops.quote_name
    identity_rows = tuple(
        (
            entity_id,
            identity_by_entity[entity_id][0],
            str(identity_by_entity[entity_id][1]),
        )
        for entity_id in sorted(identity_by_entity)
        if entity_id in entities
    )
    identity_values = ", ".join(
        "(%s::bigint, %s::text, %s::text)" for _ in identity_rows
    )
    tables = {
        "chunk": quote(TextChunk._meta.db_table),
        "link": quote(CollectionEntityDocumentLink._meta.db_table),
        "manifest": quote(CollectionArtifactInput._meta.db_table),
        "artifact": quote(GraphArtifact._meta.db_table),
        "entity": quote(CollectionEntity._meta.db_table),
        "document_entity": quote(DocumentEntity._meta.db_table),
        "document_link": quote(DocumentEntityMention._meta.db_table),
        "mention": quote(EntityMention._meta.db_table),
    }
    seed_chunk_ids = tuple(seed.chunk_id for seed in request.seeds)
    total_cap = state.config.max_nodes * state.config.max_mentions_per_entity
    document_batches = tuple(_query_batches(request.allowed_doc_ids))
    manifest_batches = tuple(_query_batches(scope.manifest_ids))
    document_cte = " UNION ALL ".join(
        "SELECT unnest(%s::uuid[]) AS document_id" for _ in document_batches
    )
    manifest_cte = " UNION ALL ".join(
        "SELECT unnest(%s::bigint[]) AS manifest_id" for _ in manifest_batches
    )
    if not manifest_cte:
        return ()
    parameters: list[object] = []
    for row in identity_rows:
        parameters.extend(row)
    parameters.extend(list(batch) for batch in document_batches)
    parameters.extend(list(batch) for batch in manifest_batches)
    parameters.extend(
        (
            list(scope.collection_artifact_ids),
            list(seed_chunk_ids),
            state.config.max_mentions_per_entity,
            total_cap + 1,
        )
    )
    sql = f"""
            WITH identity_scope(collection_entity_id, identity_kind, identity_value)
                 AS (VALUES {identity_values}),
            authorized_documents AS ({document_cte}),
            selected_manifests AS ({manifest_cte}),
            authorized_mentions AS (
                SELECT identity_scope.identity_kind,
                       identity_scope.identity_value,
                       observed.chunk_id,
                       chunk.doc_id AS document_id,
                       chunk.chunk_number,
                       MAX(entity_mention.extraction_confidence) AS confidence,
                       MIN(entity_mention.id) AS mention_id
                FROM identity_scope
                JOIN {tables['entity']} AS collection_entity
                  ON collection_entity.id = identity_scope.collection_entity_id
                JOIN {tables['link']} AS link
                  ON link.collection_entity_id = collection_entity.id
                JOIN {tables['artifact']} AS collection_artifact
                  ON collection_artifact.id = link.artifact_id
                JOIN {tables['manifest']} AS manifest_input
                  ON manifest_input.id = link.manifest_input_id
                JOIN selected_manifests
                  ON selected_manifests.manifest_id = manifest_input.id
                JOIN authorized_documents
                  ON authorized_documents.document_id = manifest_input.document_id
                JOIN {tables['artifact']} AS document_artifact
                  ON document_artifact.id = manifest_input.document_artifact_id
                JOIN {tables['document_entity']} AS document_entity
                  ON document_entity.id = link.document_entity_id
                JOIN {tables['document_link']} AS document_entity_mention
                  ON document_entity_mention.document_entity_id = document_entity.id
                JOIN {tables['mention']} AS entity_mention
                  ON entity_mention.id = document_entity_mention.mention_id
                CROSS JOIN LATERAL (
                    SELECT entity_mention.chunk_id::bigint AS chunk_id
                    UNION
                    SELECT CASE
                        WHEN observation.value ? 'chunk_id'
                         AND (observation.value ->> 'chunk_id')
                             ~ '^[1-9][0-9]{{0,18}}$'
                        THEN CASE
                            WHEN (observation.value ->> 'chunk_id')::numeric
                                 <= 9223372036854775807
                            THEN (observation.value ->> 'chunk_id')::bigint
                            ELSE NULL
                        END
                        ELSE NULL
                    END AS chunk_id
                    FROM jsonb_array_elements(
                        CASE
                            WHEN jsonb_typeof(
                                entity_mention.metadata -> 'observations'
                            ) = 'array'
                            THEN entity_mention.metadata -> 'observations'
                            ELSE '[]'::jsonb
                        END
                    ) AS observation(value)
                ) AS observed
                JOIN {tables['chunk']} AS chunk
                  ON chunk.id = observed.chunk_id
                 AND chunk.doc_id = entity_mention.document_id
                WHERE link.artifact_id = ANY(%s::bigint[])
                  AND observed.chunk_id <> ALL(%s::bigint[])
                  AND collection_artifact.status = 'active'
                  AND collection_artifact.scope_type = 'collection'
                  AND collection_artifact.collection_scope_id =
                      manifest_input.collection_id
                  AND collection_artifact.scope_id = manifest_input.collection_id::text
                  AND collection_entity.status = 'active'
                  AND collection_entity.artifact_id = collection_artifact.id
                  AND collection_entity.collection_id = manifest_input.collection_id
                  AND link.status = 'active'
                  AND link.outcome = 'automatic'
                  AND link.resolver_version = collection_artifact.resolver_version
                  AND manifest_input.artifact_id = collection_artifact.id
                  AND document_artifact.status = 'active'
                  AND document_artifact.scope_type = 'document'
                  AND document_artifact.scope_id = manifest_input.document_id::text
                  AND document_entity.status = 'active'
                  AND document_entity.artifact_id = document_artifact.id
                  AND document_entity.document_id = manifest_input.document_id
                  AND document_entity_mention.status = 'active'
                  AND document_entity_mention.resolver_version =
                      document_artifact.resolver_version
                  AND entity_mention.artifact_id = document_artifact.id
                  AND entity_mention.document_id = manifest_input.document_id
                GROUP BY identity_scope.identity_kind,
                         identity_scope.identity_value,
                         observed.chunk_id, chunk.doc_id, chunk.chunk_number
            ), ranked AS (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY identity_kind, identity_value
                    ORDER BY confidence DESC, document_id,
                             chunk_number, chunk_id, mention_id
                ) AS mention_rank
                FROM authorized_mentions
            )
            SELECT identity_kind, identity_value, chunk_id, document_id,
                   chunk_number, confidence, mention_id
            FROM ranked
            WHERE mention_rank <= %s
            ORDER BY identity_kind, identity_value,
                     confidence DESC, document_id,
                     chunk_number, chunk_id, mention_id
            LIMIT %s
        """
    raw_rows = list(
        _execute_bounded_sql(
            sql,
            parameters,
            maximum=total_cap,
            state=state,
        )
    )

    by_identity = _select_fallback_rows(
        raw_rows,
        maximum_per_identity=state.config.max_mentions_per_entity,
    )

    result: list[AuthorizedIdentityMention] = []
    for identity in sorted(by_identity):
        selected = by_identity[identity]
        for row in selected:
            semantic = (
                identity,
                int(row[2]),
                str(row[3]),
                int(row[4]),
                float(row[5]),
                int(row[6]),
            )
            provenance_key = sha256(
                json.dumps(
                    semantic,
                    default=str,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
            result.append(
                AuthorizedIdentityMention(
                    identity_key=identity,
                    evidence=AuthorizedChunkEvidence(
                        chunk_id=int(row[2]),
                        document_id=cast(UUID, row[3]),
                        chunk_number=int(row[4]),
                        confidence=float(row[5]),
                        provenance_key=provenance_key,
                    ),
                )
            )
    if len(result) > total_cap:
        raise _SnapshotMiss
    return tuple(
        sorted(
            result,
            key=lambda mention: (
                mention.identity_key,
                _evidence_order(mention.evidence),
            ),
        )
    )


@contextmanager
def authorized_retrieval_snapshot(
    *,
    timeout_ms: int,
) -> Iterator[_MonotonicDeadline]:
    """Open one outer PostgreSQL read-only repeatable-read graph snapshot."""

    if type(timeout_ms) is not int or not 1 <= timeout_ms <= 150:
        raise ValueError("timeout_ms must be an exact integer in [1, 150]")
    if _ACTIVE_RETRIEVAL_SNAPSHOT.get() is not None:
        raise RuntimeError("an authorized retrieval snapshot is already active")

    from django.db import connection, transaction

    if connection.vendor != "postgresql":
        raise RuntimeError("authorized retrieval snapshots require PostgreSQL")
    if connection.in_atomic_block:
        raise RuntimeError(
            "authorized retrieval snapshot requires an outer transaction"
        )
    base_config = _load_algorithm_config()
    if timeout_ms > base_config.timeout_ms:
        raise ValueError("timeout_ms exceeds the effective configured timeout")
    config = replace(base_config, timeout_ms=timeout_ms)
    deadline = _MonotonicDeadline.after_ms(timeout_ms)
    with transaction.atomic():
        with connection.cursor() as cursor:
            cursor.execute(
                "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"
            )
            cursor.execute("SET LOCAL statement_timeout = %s", [timeout_ms])
        state = _RetrievalSnapshotState(
            config=config,
            deadline=deadline,
            using=connection.alias,
        )
        token = _ACTIVE_RETRIEVAL_SNAPSHOT.set(state)
        try:
            deadline.check()
            yield deadline
            deadline.check()
        finally:
            _ACTIVE_RETRIEVAL_SNAPSHOT.reset(token)


@dataclass(frozen=True, slots=True, repr=False)
class _EvaluationTraceCapability:
    """Package-private capability for Task 20 evaluation trace collection."""

    sink: Callable[[bytes], object]

    def __post_init__(self) -> None:
        if not callable(self.sink):
            raise ValueError("evaluation trace sink must be callable")


def _positive_database_int(value: object, field_name: str) -> int:
    if type(value) is not int or not 1 <= value <= _DATABASE_ID_MAX:
        raise ValueError(f"{field_name} must be a positive database integer")
    return value


def _unit_float(value: object, field_name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{field_name} must be a finite number in [0, 1]")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(
            f"{field_name} must be a finite number in [0, 1]"
        ) from error
    if not isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{field_name} must be a finite number in [0, 1]")
    return number


def _positive_float(value: object, field_name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{field_name} must be finite and positive")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"{field_name} must be finite and positive") from error
    if not isfinite(number) or number <= 0.0:
        raise ValueError(f"{field_name} must be finite and positive")
    return number


def _exact_tuple(
    value: object,
    field_name: str,
    *,
    maximum: int,
    nonempty: bool = False,
) -> tuple[object, ...]:
    if type(value) is not tuple:
        raise ValueError(f"{field_name} must be an exact tuple")
    if len(value) > maximum:
        raise ValueError(f"{field_name} exceeds the hard cap of {maximum}")
    if nonempty and not value:
        raise ValueError(f"{field_name} must not be empty")
    return value


def _identity_key(value: object, field_name: str = "identity_key") -> StableNodeKey:
    if type(value) is not tuple or len(value) != 2:
        raise ValueError(
            f"{field_name} must be an exact canonical or local identity key"
        )
    kind, identifier = value
    if type(kind) is not str:
        raise ValueError(
            f"{field_name} must be an exact canonical or local identity key"
        )
    if kind == "canonical":
        _positive_database_int(identifier, field_name)
    elif kind == "local":
        if type(identifier) is not str or _HASH_PATTERN.fullmatch(identifier) is None:
            raise ValueError(
                f"{field_name} local key must be an exact lowercase cluster hash"
            )
    else:
        raise ValueError(
            f"{field_name} must be an exact canonical or local identity key"
        )
    return cast(StableNodeKey, value)


def _safe_fsum(values: Iterable[float], field_name: str) -> float:
    try:
        total = fsum(values)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{field_name} produced non-finite math") from error
    if not isfinite(total) or total < 0.0:
        raise ValueError(f"{field_name} produced non-finite math")
    return total


def _require_sha256(value: object, field_name: str) -> str:
    if type(value) is not str or _HASH_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be an exact lowercase SHA-256 digest")
    return value


def _canonical_audit_value(value: object) -> object:
    """Convert private storage audit values to deterministic JSON scalars."""

    if type(value) in (type(None), bool, int, str):
        return value
    if type(value) is UUID:
        return str(value)
    if type(value) is float:
        if not isfinite(value):
            raise ValueError("raw audit floats must be finite")
        return value.hex()
    if type(value) is tuple:
        return tuple(_canonical_audit_value(item) for item in value)
    raise ValueError("raw audit values must be exact JSON-safe scalars")


@dataclass(frozen=True, slots=True)
class AuthorizedChunkEvidence:
    """One authorized real-chunk evidence record with stable coordinates."""

    chunk_id: int
    document_id: UUID
    chunk_number: int
    confidence: float
    provenance_key: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "chunk_id",
            _positive_database_int(self.chunk_id, "chunk_id"),
        )
        if type(self.document_id) is not UUID:
            raise ValueError("document_id must be an exact UUID value")
        if (
            type(self.chunk_number) is not int
            or not 0 <= self.chunk_number <= _CHUNK_NUMBER_MAX
        ):
            raise ValueError("chunk_number must be an exact nonnegative integer")
        object.__setattr__(
            self,
            "confidence",
            _unit_float(self.confidence, "confidence"),
        )
        provenance = self.provenance_key
        if (
            type(provenance) is not str
            or not provenance
            or provenance != provenance.strip()
            or len(provenance) > 256
            or "\x00" in provenance
        ):
            raise ValueError("provenance_key must be a bounded exact string")


@dataclass(frozen=True, slots=True)
class AuthorizedSeedIdentity:
    """One authorized seed-to-canonical-identity association."""

    seed_chunk_id: int
    identity_key: StableNodeKey

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "seed_chunk_id",
            _positive_database_int(self.seed_chunk_id, "seed_chunk_id"),
        )
        object.__setattr__(
            self,
            "identity_key",
            _identity_key(self.identity_key),
        )


@dataclass(frozen=True, slots=True)
class AuthorizedRelationGroup:
    """One canonicalized pre-normalized relation transition group."""

    source_key: StableNodeKey
    relation_type: str
    target_key: StableNodeKey
    direction: RetrievalDirection
    raw_weight: float
    admission_hop: int
    evidence: tuple[AuthorizedChunkEvidence, ...]

    def __post_init__(self) -> None:
        source = _identity_key(self.source_key, "source_key")
        target = _identity_key(self.target_key, "target_key")
        if source == target:
            raise ValueError("relation group must not be a collapsed self-loop")
        object.__setattr__(self, "source_key", source)
        object.__setattr__(self, "target_key", target)
        if (
            type(self.relation_type) is not str
            or _RELATION_TYPE_PATTERN.fullmatch(self.relation_type) is None
        ):
            raise ValueError("relation_type must be an exact canonical token")
        if type(self.direction) is not RetrievalDirection:
            raise ValueError("direction must be an exact RetrievalDirection value")
        object.__setattr__(
            self,
            "raw_weight",
            _positive_float(self.raw_weight, "raw_weight"),
        )
        if (
            type(self.admission_hop) is not int
            or not 1 <= self.admission_hop <= 2
        ):
            raise ValueError("admission_hop must be an exact integer in [1, 2]")
        evidence = _exact_tuple(
            self.evidence,
            "evidence",
            maximum=_MAX_EVIDENCE_ROWS,
            nonempty=True,
        )
        if any(type(item) is not AuthorizedChunkEvidence for item in evidence):
            raise ValueError(
                "evidence must contain exact AuthorizedChunkEvidence values"
            )
        provenance_keys = tuple(item.provenance_key for item in evidence)
        if len(set(provenance_keys)) != len(provenance_keys):
            raise ValueError("evidence provenance keys must be unique per group")


@dataclass(frozen=True, slots=True)
class AuthorizedIdentityMention:
    """One authorized fallback entity mention mapped to an identity."""

    identity_key: StableNodeKey
    evidence: AuthorizedChunkEvidence

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "identity_key",
            _identity_key(self.identity_key),
        )
        if type(self.evidence) is not AuthorizedChunkEvidence:
            raise ValueError(
                "evidence must be an exact AuthorizedChunkEvidence value"
            )


@dataclass(frozen=True, slots=True)
class AuthorizedGraphSnapshot:
    """Immutable, provider-neutral permission-safe graph ranking input."""

    config: PPRAlgorithmConfig
    load_max_hops: int
    allowed_doc_ids: tuple[UUID, ...]
    allowed_collection_ids: tuple[int, ...]
    scope_version_signature: str
    identity_keys: tuple[StableNodeKey, ...]
    seed_identities: tuple[AuthorizedSeedIdentity, ...]
    relation_groups: tuple[AuthorizedRelationGroup, ...]
    mentions: tuple[AuthorizedIdentityMention, ...]
    raw_audit_rows: tuple[tuple[int, str, tuple[object, ...]], ...] = ()

    def __post_init__(self) -> None:
        if type(self.config) is not PPRAlgorithmConfig:
            raise ValueError("config must be an exact PPRAlgorithmConfig value")
        if (
            type(self.load_max_hops) is not int
            or not 1 <= self.load_max_hops <= self.config.max_hops
        ):
            raise ValueError("load_max_hops must be bounded by config.max_hops")

        document_ids = _exact_tuple(
            self.allowed_doc_ids,
            "allowed_doc_ids",
            maximum=self.config.max_scope_documents,
            nonempty=True,
        )
        if any(type(identifier) is not UUID for identifier in document_ids):
            raise ValueError("allowed_doc_ids must contain exact UUID values")
        if len(set(document_ids)) != len(document_ids):
            raise ValueError("allowed_doc_ids must be unique")
        if document_ids != tuple(
            sorted(document_ids, key=lambda identifier: identifier.int)
        ):
            raise ValueError("allowed_doc_ids must be canonically sorted")

        collection_ids = _exact_tuple(
            self.allowed_collection_ids,
            "allowed_collection_ids",
            maximum=self.config.max_scope_collections,
            nonempty=True,
        )
        validated_collections = tuple(
            _positive_database_int(identifier, "allowed_collection_ids")
            for identifier in collection_ids
        )
        if len(set(validated_collections)) != len(validated_collections):
            raise ValueError("allowed_collection_ids must be unique")
        if validated_collections != tuple(sorted(validated_collections)):
            raise ValueError("allowed_collection_ids must be canonically sorted")
        if len(validated_collections) > len(document_ids):
            raise ValueError(
                "allowed_collection_ids cannot outnumber authorized documents"
            )

        identities = _exact_tuple(
            self.identity_keys,
            "identity_keys",
            maximum=self.config.max_nodes,
            nonempty=True,
        )
        validated_identities = tuple(
            _identity_key(identity, "identity_keys") for identity in identities
        )
        identity_set = set(validated_identities)
        if len(identity_set) != len(validated_identities):
            raise ValueError("identity_keys must be unique")

        seed_identities = _exact_tuple(
            self.seed_identities,
            "seed_identities",
            maximum=self.config.max_nodes,
        )
        if any(type(item) is not AuthorizedSeedIdentity for item in seed_identities):
            raise ValueError(
                "seed_identities must contain exact AuthorizedSeedIdentity values"
            )
        if any(item.identity_key not in identity_set for item in seed_identities):
            raise ValueError("seed_identities must reference identity_keys")

        groups = _exact_tuple(
            self.relation_groups,
            "relation_groups",
            maximum=(
                self.config.max_edges * _MAX_RETRIEVAL_DIRECTIONS_PER_RELATION
            ),
        )
        if any(type(item) is not AuthorizedRelationGroup for item in groups):
            raise ValueError(
                "relation_groups must contain exact AuthorizedRelationGroup values"
            )
        group_keys = tuple(_group_key(group) for group in groups)
        if len(set(group_keys)) != len(group_keys):
            raise ValueError("relation_groups must have unique semantic keys")
        if any(
            group.source_key not in identity_set
            or group.target_key not in identity_set
            for group in groups
        ):
            raise ValueError("relation_groups must reference identity_keys")
        if any(group.admission_hop > self.load_max_hops for group in groups):
            raise ValueError("relation_groups exceed load_max_hops")
        evidence_reference_count = sum(len(group.evidence) for group in groups)
        if evidence_reference_count > (
            self.config.max_evidence_rows
            * _MAX_RETRIEVAL_DIRECTIONS_PER_RELATION
        ):
            raise ValueError("relation evidence exceeds its directional hard cap")
        evidence_by_provenance: dict[
            str, tuple[int, UUID, int, float]
        ] = {}
        provenance_reference_counts: dict[str, int] = defaultdict(int)
        for group in groups:
            for evidence in group.evidence:
                provenance_key = evidence.provenance_key
                provenance_reference_counts[provenance_key] += 1
                if provenance_reference_counts[provenance_key] > (
                    _MAX_RETRIEVAL_DIRECTIONS_PER_RELATION
                ):
                    raise ValueError(
                        "relation evidence provenance exceeds direction count"
                    )
                evidence_identity = (
                    evidence.chunk_id,
                    evidence.document_id,
                    evidence.chunk_number,
                    evidence.confidence,
                )
                previous = evidence_by_provenance.setdefault(
                    provenance_key,
                    evidence_identity,
                )
                if previous != evidence_identity:
                    raise ValueError(
                        "relation evidence provenance has conflicting values"
                    )
                if len(evidence_by_provenance) > self.config.max_evidence_rows:
                    raise ValueError("relation evidence exceeds max_evidence_rows")

        mentions = _exact_tuple(
            self.mentions,
            "mentions",
            maximum=(
                self.config.max_nodes * self.config.max_mentions_per_entity
            ),
        )
        if any(type(item) is not AuthorizedIdentityMention for item in mentions):
            raise ValueError(
                "mentions must contain exact AuthorizedIdentityMention values"
            )
        if any(item.identity_key not in identity_set for item in mentions):
            raise ValueError("mentions must reference identity_keys")
        mention_counts: dict[StableNodeKey, int] = defaultdict(int)
        for mention in mentions:
            mention_counts[mention.identity_key] += 1
            if (
                mention_counts[mention.identity_key]
                > self.config.max_mentions_per_entity
            ):
                raise ValueError("mentions per identity exceed their configured cap")

        allowed_documents = set(document_ids)
        evidence_rows = chain(
            chain.from_iterable(group.evidence for group in groups),
            (mention.evidence for mention in mentions),
        )
        coordinates: dict[int, tuple[UUID, int]] = {}
        for evidence in evidence_rows:
            if evidence.document_id not in allowed_documents:
                raise ValueError("evidence must reference allowed_doc_ids")
            coordinate = (evidence.document_id, evidence.chunk_number)
            previous = coordinates.setdefault(evidence.chunk_id, coordinate)
            if previous != coordinate:
                raise ValueError(
                    "one chunk_id cannot have conflicting stable coordinates"
                )

        audit_rows = _exact_tuple(
            self.raw_audit_rows,
            "raw_audit_rows",
            maximum=(
                self.config.max_nodes
                + self.config.max_edges
                + self.config.max_evidence_rows
                + self.config.max_nodes * self.config.max_mentions_per_entity
            ),
        )
        previous_audit_key: tuple[object, ...] | None = None
        canonical_audit_rows: list[tuple[int, str, tuple[object, ...]]] = []
        for audit_row in audit_rows:
            if (
                type(audit_row) is not tuple
                or len(audit_row) != 3
                or type(audit_row[0]) is not int
                or not 0 <= audit_row[0] <= self.load_max_hops
                or type(audit_row[1]) is not str
                or not audit_row[1]
                or type(audit_row[2]) is not tuple
            ):
                raise ValueError("raw_audit_rows contain malformed private rows")
            canonical_values = cast(
                tuple[object, ...],
                _canonical_audit_value(audit_row[2]),
            )
            canonical_row = (audit_row[0], audit_row[1], canonical_values)
            audit_key = (canonical_row[0], canonical_row[1], repr(canonical_values))
            if previous_audit_key is not None and audit_key <= previous_audit_key:
                raise ValueError("raw_audit_rows must be unique and sorted")
            previous_audit_key = audit_key
            canonical_audit_rows.append(canonical_row)

        object.__setattr__(
            self,
            "scope_version_signature",
            _require_sha256(
                self.scope_version_signature,
                "scope_version_signature",
            ),
        )
        object.__setattr__(self, "identity_keys", validated_identities)
        object.__setattr__(self, "raw_audit_rows", tuple(canonical_audit_rows))
        object.__setattr__(
            self,
            "allowed_collection_ids",
            validated_collections,
        )


def _compose_authorized_relation_groups(
    projections: tuple[_DirectionalPhysicalProjection, ...],
) -> tuple[AuthorizedRelationGroup, ...]:
    """Collapse physical copies and recompute support/confidence from evidence."""

    grouped: dict[
        tuple[StableNodeKey, str, StableNodeKey, RetrievalDirection],
        list[_DirectionalPhysicalProjection],
    ] = defaultdict(list)
    for projection in projections:
        if type(projection) is not _DirectionalPhysicalProjection:
            raise ValueError("directional projections must be exact private values")
        if projection.source_key == projection.target_key:
            continue
        grouped[
            (
                projection.source_key,
                projection.relation_type,
                projection.target_key,
                projection.direction,
            )
        ].append(projection)

    result: list[AuthorizedRelationGroup] = []
    for key in sorted(grouped, key=lambda value: (*value[:3], value[3].value)):
        rows = grouped[key]
        evidence_by_provenance: dict[str, AuthorizedChunkEvidence] = {}
        admission_hop = 2
        physical_weights: list[float] = []
        for row in rows:
            admission_hop = min(admission_hop, row.admission_hop)
            if not row.evidence:
                raise _SnapshotMiss
            physical_evidence = {
                item.evidence.provenance_key: item.evidence
                for item in row.evidence
            }
            physical_weights.append(
                raw_edge_weight(
                    direction=row.direction,
                    confidence=max(
                        item.confidence for item in physical_evidence.values()
                    ),
                    support_count=len(physical_evidence),
                    destination_retrieval_utility=(
                        row.destination_retrieval_utility
                    ),
                )
            )
            for projection in row.evidence:
                evidence = projection.evidence
                previous = evidence_by_provenance.setdefault(
                    evidence.provenance_key,
                    evidence,
                )
                if previous != evidence:
                    raise _SnapshotMiss
        evidence = tuple(
            sorted(evidence_by_provenance.values(), key=_evidence_order)
        )
        if not evidence:
            raise _SnapshotMiss
        weight = max(physical_weights)
        if weight <= 0.0:
            continue
        result.append(
            AuthorizedRelationGroup(
                source_key=key[0],
                relation_type=key[1],
                target_key=key[2],
                direction=key[3],
                raw_weight=weight,
                admission_hop=admission_hop,
                evidence=evidence,
            )
        )
    return tuple(sorted(result, key=_group_key))


@dataclass(frozen=True, slots=True)
class _Candidate:
    chunk_id: int
    document_id: UUID
    chunk_number: int
    contribution: float
    semantic_hop: int
    seed_rank: int


def _group_key(group: AuthorizedRelationGroup) -> tuple[object, ...]:
    return (
        group.source_key,
        group.relation_type,
        group.target_key,
        group.direction.value,
    )


def _group_selection_key(group: AuthorizedRelationGroup) -> tuple[object, ...]:
    return (-group.raw_weight, *_group_key(group))


def _evidence_order(evidence: AuthorizedChunkEvidence) -> tuple[object, ...]:
    return (
        -evidence.confidence,
        evidence.document_id.int,
        evidence.chunk_number,
        evidence.chunk_id,
        evidence.provenance_key,
    )


def _collapse_evidence(
    evidence: Iterable[AuthorizedChunkEvidence],
    seed_chunk_ids: set[int],
) -> tuple[AuthorizedChunkEvidence, ...]:
    by_chunk: dict[int, AuthorizedChunkEvidence] = {}
    for row in evidence:
        if row.chunk_id in seed_chunk_ids:
            continue
        previous = by_chunk.get(row.chunk_id)
        if previous is None or row.confidence > previous.confidence:
            by_chunk[row.chunk_id] = row
        elif (
            row.confidence == previous.confidence
            and row.provenance_key < previous.provenance_key
        ):
            by_chunk[row.chunk_id] = row
    return tuple(sorted(by_chunk.values(), key=_evidence_order))


def _build_restart_vector(
    snapshot: AuthorizedGraphSnapshot,
    request: GraphExpansionRequest,
) -> tuple[
    dict[StableNodeKey, float],
    dict[StableNodeKey, tuple[int, int]],
    tuple[tuple[int, StableNodeKey], ...],
]:
    request_by_chunk = {seed.chunk_id: seed for seed in request.seeds}
    mapped_seed_ids = {row.seed_chunk_id for row in snapshot.seed_identities}
    unknown = mapped_seed_ids.difference(request_by_chunk)
    if unknown:
        raise ValueError("seed_identities contain a non-request seed")

    canonical_mappings = tuple(
        sorted(
            {
                (row.seed_chunk_id, row.identity_key)
                for row in snapshot.seed_identities
            }
        )
    )
    identities_by_seed: dict[int, list[StableNodeKey]] = defaultdict(list)
    for seed_chunk_id, identity in canonical_mappings:
        identities_by_seed[seed_chunk_id].append(identity)

    contributions: dict[StableNodeKey, list[float]] = defaultdict(list)
    labels: dict[StableNodeKey, tuple[int, int]] = {}
    for seed in request.seeds:
        identities = tuple(sorted(identities_by_seed[seed.chunk_id]))
        if not identities:
            continue
        divided = seed.restart_weight / len(identities)
        if not isfinite(divided) or divided <= 0.0:
            raise ValueError("seed restart split produced non-finite math")
        for identity in identities:
            contributions[identity].append(divided)
            label = (0, seed.rank)
            labels[identity] = min(labels.get(identity, label), label)

    restart = {
        identity: _safe_fsum(
            sorted(contributions[identity]),
            "seed restart accumulation",
        )
        for identity in sorted(contributions)
    }
    if not restart:
        raise _SnapshotMiss
    return restart, labels, canonical_mappings


def _replay_groups(
    snapshot: AuthorizedGraphSnapshot,
    restart: dict[StableNodeKey, float],
    initial_labels: dict[StableNodeKey, tuple[int, int]],
    effective_max_hops: int,
) -> tuple[
    tuple[AuthorizedRelationGroup, ...],
    dict[AuthorizedRelationGroup, float],
    dict[StableNodeKey, tuple[int, int]],
    tuple[StableNodeKey, ...],
    tuple[AuthorizedRelationGroup, ...],
]:
    eligible = tuple(
        sorted(
            (
                group
                for group in snapshot.relation_groups
                if group.admission_hop <= effective_max_hops
            ),
            key=_group_selection_key,
        )
    )
    by_source: dict[StableNodeKey, list[AuthorizedRelationGroup]] = defaultdict(list)
    for group in eligible:
        by_source[group.source_key].append(group)
    fanout_selected = tuple(
        group
        for source in sorted(by_source)
        for group in sorted(by_source[source], key=_group_selection_key)[
            : snapshot.config.max_fanout
        ]
    )

    admitted = set(restart)
    labels = dict(initial_labels)
    retained: list[AuthorizedRelationGroup] = []
    cap_reached = False
    for hop in range(1, effective_max_hops + 1):
        frontier = sorted(
            (
                group
                for group in fanout_selected
                if group.admission_hop == hop
            ),
            key=_group_selection_key,
        )
        for group in frontier:
            if group.source_key not in admitted:
                continue
            target_is_new = group.target_key not in admitted
            if len(retained) >= snapshot.config.max_edges or (
                target_is_new and len(admitted) >= snapshot.config.max_nodes
            ):
                cap_reached = True
                break
            retained.append(group)
            source_hop, source_rank = labels[group.source_key]
            target_label = (source_hop + 1, source_rank)
            if target_is_new:
                admitted.add(group.target_key)
                labels[group.target_key] = target_label
            else:
                labels[group.target_key] = min(
                    labels[group.target_key],
                    target_label,
                )
        if cap_reached:
            break

    retained_tuple = tuple(retained)
    retained_by_source: dict[
        StableNodeKey, list[AuthorizedRelationGroup]
    ] = defaultdict(list)
    for group in retained_tuple:
        retained_by_source[group.source_key].append(group)
    shares: dict[AuthorizedRelationGroup, float] = {}
    for source in sorted(retained_by_source):
        row = sorted(retained_by_source[source], key=_group_key)
        total = _safe_fsum(
            (group.raw_weight for group in row),
            "transition row normalization",
        )
        for group in row:
            share = group.raw_weight / total
            if not isfinite(share) or share <= 0.0:
                raise ValueError("transition share produced non-finite math")
            shares[group] = share
    return (
        retained_tuple,
        shares,
        labels,
        tuple(sorted(admitted)),
        eligible,
    )


def _project_candidates(
    snapshot: AuthorizedGraphSnapshot,
    request: GraphExpansionRequest,
    retained: tuple[AuthorizedRelationGroup, ...],
    shares: dict[AuthorizedRelationGroup, float],
    labels: dict[StableNodeKey, tuple[int, int]],
    admitted: tuple[StableNodeKey, ...],
    scores: dict[StableNodeKey, float],
) -> tuple[_Candidate, ...]:
    seed_chunk_ids = {seed.chunk_id for seed in request.seeds}
    identity_chunk: dict[tuple[StableNodeKey, int], float] = {}
    coordinates: dict[int, tuple[UUID, int]] = {}
    identities_with_selected_relation_evidence: set[StableNodeKey] = set()

    for group in sorted(retained, key=_group_key):
        selected = _collapse_evidence(group.evidence, seed_chunk_ids)[
            : snapshot.config.max_evidence_per_edge
        ]
        if not selected:
            continue
        identities_with_selected_relation_evidence.add(group.target_key)
        flow = edge_evidence_flow(
            restart_probability=snapshot.config.ppr_restart,
            source_score=scores[group.source_key],
            normalized_share=shares[group],
        )
        contribution = flow / len(selected)
        if not isfinite(contribution) or contribution < 0.0:
            raise ValueError("edge evidence projection produced non-finite math")
        for evidence in selected:
            coordinates[evidence.chunk_id] = (
                evidence.document_id,
                evidence.chunk_number,
            )
            key = (group.target_key, evidence.chunk_id)
            identity_chunk[key] = max(identity_chunk.get(key, 0.0), contribution)

    mentions_by_identity: dict[
        StableNodeKey, list[AuthorizedChunkEvidence]
    ] = defaultdict(list)
    admitted_set = set(admitted)
    for mention in snapshot.mentions:
        if (
            mention.identity_key in admitted_set
            and mention.identity_key
            not in identities_with_selected_relation_evidence
        ):
            mentions_by_identity[mention.identity_key].append(mention.evidence)
    for identity in sorted(mentions_by_identity):
        selected = _collapse_evidence(
            mentions_by_identity[identity],
            seed_chunk_ids,
        )[: snapshot.config.max_mentions_per_entity]
        if not selected:
            continue
        contribution = MENTION_FACTOR * scores.get(identity, 0.0) / len(selected)
        if not isfinite(contribution) or contribution < 0.0:
            raise ValueError("mention projection produced non-finite math")
        for evidence in selected:
            coordinates[evidence.chunk_id] = (
                evidence.document_id,
                evidence.chunk_number,
            )
            key = (identity, evidence.chunk_id)
            identity_chunk[key] = max(identity_chunk.get(key, 0.0), contribution)

    contributions_by_chunk: dict[
        int, list[tuple[StableNodeKey, float]]
    ] = defaultdict(list)
    for (identity, chunk_id), contribution in identity_chunk.items():
        if contribution > 0.0:
            contributions_by_chunk[chunk_id].append((identity, contribution))

    candidates: list[_Candidate] = []
    for chunk_id in sorted(contributions_by_chunk):
        identity_values = sorted(contributions_by_chunk[chunk_id])
        total = _safe_fsum(
            (contribution for _, contribution in identity_values),
            "candidate contribution accumulation",
        )
        contributing_labels = tuple(labels[identity] for identity, _ in identity_values)
        semantic_hop, seed_rank = min(contributing_labels)
        document_id, chunk_number = coordinates[chunk_id]
        candidates.append(
            _Candidate(
                chunk_id=chunk_id,
                document_id=document_id,
                chunk_number=chunk_number,
                contribution=total,
                semantic_hop=semantic_hop,
                seed_rank=seed_rank,
            )
        )

    ordered = sorted(
        candidates,
        key=lambda candidate: (
            -candidate.contribution,
            candidate.semantic_hop,
            candidate.seed_rank,
            candidate.document_id.int,
            candidate.chunk_number,
            candidate.chunk_id,
        ),
    )
    document_counts: dict[UUID, int] = defaultdict(int)
    selected_candidates: list[_Candidate] = []
    for candidate in ordered:
        if document_counts[candidate.document_id] >= snapshot.config.max_per_document:
            continue
        selected_candidates.append(candidate)
        document_counts[candidate.document_id] += 1
        if len(selected_candidates) >= snapshot.config.max_candidates:
            break
    return tuple(selected_candidates)


def _evidence_payload(evidence: AuthorizedChunkEvidence) -> list[object]:
    return [
        evidence.chunk_id,
        str(evidence.document_id),
        evidence.chunk_number,
        evidence.confidence.hex(),
        evidence.provenance_key,
    ]


def _effective_graph_signature(
    snapshot: AuthorizedGraphSnapshot,
    effective_max_hops: int,
    canonical_mappings: tuple[tuple[int, StableNodeKey], ...],
    eligible: tuple[AuthorizedRelationGroup, ...],
    retained: tuple[AuthorizedRelationGroup, ...],
    shares: dict[AuthorizedRelationGroup, float],
    admitted: tuple[StableNodeKey, ...],
) -> str:
    eligible_set = set(eligible)
    effective_mentions = tuple(
        sorted(
            (
                mention
                for mention in snapshot.mentions
                if mention.identity_key in set(admitted)
            ),
            key=lambda mention: (
                mention.identity_key,
                _evidence_order(mention.evidence),
            ),
        )
    )
    payload = {
        "admitted_identities": [list(identity) for identity in admitted],
        "allowed_collection_ids": list(snapshot.allowed_collection_ids),
        "allowed_doc_ids": [str(item) for item in snapshot.allowed_doc_ids],
        "effective_max_hops": effective_max_hops,
        "eligible_groups": [
            [
                list(group.source_key),
                group.relation_type,
                list(group.target_key),
                group.direction.value,
                group.raw_weight.hex(),
                group.admission_hop,
                [
                    _evidence_payload(evidence)
                    for evidence in sorted(group.evidence, key=_evidence_order)
                ],
            ]
            for group in sorted(eligible_set, key=_group_key)
        ],
        "mentions": [
            [list(mention.identity_key), _evidence_payload(mention.evidence)]
            for mention in effective_mentions
        ],
        "raw_storage_audit": [
            [hop, kind, list(values)]
            for hop, kind, values in snapshot.raw_audit_rows
            if hop <= effective_max_hops
        ],
        "retained_groups": [
            [*_group_key(group), shares[group].hex()]
            for group in sorted(retained, key=_group_key)
        ],
        "scope_version_signature": snapshot.scope_version_signature,
        "seed_identities": [
            [seed_chunk_id, list(identity)]
            for seed_chunk_id, identity in canonical_mappings
        ],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _trace_bytes(
    *,
    algorithm_signature: str,
    graph_version_signature: str,
    effective_max_hops: int,
    restart: dict[StableNodeKey, float],
    retained: tuple[AuthorizedRelationGroup, ...],
    shares: dict[AuthorizedRelationGroup, float],
    scores: dict[StableNodeKey, float],
    candidates: tuple[_Candidate, ...],
) -> bytes:
    restart_total = _safe_fsum(restart.values(), "restart trace normalization")
    payload = {
        "algorithm_signature": algorithm_signature,
        "candidate_contributions": [
            [
                candidate.chunk_id,
                candidate.contribution.hex(),
                candidate.semantic_hop,
                candidate.seed_rank,
                str(candidate.document_id),
                candidate.chunk_number,
            ]
            for candidate in candidates
        ],
        "effective_max_hops": effective_max_hops,
        "graph_version_signature": graph_version_signature,
        "ppr_scores": [
            [list(identity), scores[identity].hex()]
            for identity in sorted(scores)
        ],
        "restart_vector": [
            [list(identity), (restart[identity] / restart_total).hex()]
            for identity in sorted(restart)
        ],
        "retained_groups": [
            [*_group_key(group), shares[group].hex()]
            for group in sorted(retained, key=_group_key)
        ],
    }
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _empty_result(
    request: GraphExpansionRequest,
    *,
    status: str,
    algorithm_signature: str | None = None,
    graph_version_signature: str | None = None,
) -> GraphExpansionResult:
    seed_ids = tuple(seed.chunk_id for seed in request.seeds)
    return GraphExpansionResult(
        chunk_ids=(),
        diagnostics=GraphExpansionDiagnostics(
            status=status,
            seed_count=len(seed_ids),
            candidate_count=0,
            algorithm_signature=algorithm_signature,
            graph_version_signature=graph_version_signature,
        ),
        seed_chunk_ids=seed_ids,
    )


def rank_authorized_graph_snapshot(
    snapshot: AuthorizedGraphSnapshot,
    request: GraphExpansionRequest,
    *,
    effective_max_hops: int,
    _eval_trace: _EvaluationTraceCapability | None = None,
    _deadline: _MonotonicDeadline | None = None,
) -> GraphExpansionResult:
    """Rank an immutable authorized snapshot without touching persistence."""

    if type(request) is not GraphExpansionRequest:
        raise ValueError("request must be an exact GraphExpansionRequest value")
    if type(snapshot) is not AuthorizedGraphSnapshot:
        return _empty_result(request, status="error")
    if (
        type(effective_max_hops) is not int
        or not 1 <= effective_max_hops <= snapshot.load_max_hops
    ):
        return _empty_result(request, status="error")
    effective_config = replace(snapshot.config, max_hops=effective_max_hops)
    algorithm_signature = graph_algorithm_signature(effective_config)

    try:
        if _deadline is None:
            deadline = _MonotonicDeadline.after_ms(effective_config.timeout_ms)
        elif type(_deadline) is _MonotonicDeadline:
            deadline = _deadline
        else:
            raise ValueError("_deadline requires the private monotonic deadline")
        deadline.check()
        if (
            _eval_trace is not None
            and type(_eval_trace) is not _EvaluationTraceCapability
        ):
            raise ValueError("_eval_trace requires the private evaluation capability")
        if (
            snapshot.allowed_doc_ids != request.allowed_doc_ids
            or snapshot.allowed_collection_ids != request.allowed_collection_ids
        ):
            raise _SnapshotMiss
        if len(request.seeds) > snapshot.config.max_seeds:
            raise _SnapshotMiss
        deadline.check()

        restart, labels, canonical_mappings = _build_restart_vector(
            snapshot,
            request,
        )
        deadline.check()
        retained, shares, labels, admitted, eligible = _replay_groups(
            snapshot,
            restart,
            labels,
            effective_max_hops,
        )
        deadline.check()
        transition_rows: dict[
            StableNodeKey, tuple[tuple[StableNodeKey, float], ...]
        ] = {}
        groups_by_source: dict[
            StableNodeKey, list[AuthorizedRelationGroup]
        ] = defaultdict(list)
        for group in retained:
            groups_by_source[group.source_key].append(group)
        for source in sorted(groups_by_source):
            transition_rows[source] = tuple(
                (group.target_key, group.raw_weight)
                for group in sorted(groups_by_source[source], key=_group_key)
            )
        deadline.check()
        scores = personalized_pagerank(
            restart,
            transition_rows,
            restart_probability=effective_config.ppr_restart,
            iterations=effective_config.ppr_iterations,
            _deadline=deadline,
        )
        deadline.check()
        candidates = _project_candidates(
            snapshot,
            request,
            retained,
            shares,
            labels,
            admitted,
            scores,
        )
        deadline.check()
        graph_version_signature = _effective_graph_signature(
            snapshot,
            effective_max_hops,
            canonical_mappings,
            eligible,
            retained,
            shares,
            admitted,
        )
        deadline.check()
        if _eval_trace is not None:
            _eval_trace.sink(
                _trace_bytes(
                    algorithm_signature=algorithm_signature,
                    graph_version_signature=graph_version_signature,
                    effective_max_hops=effective_max_hops,
                    restart=restart,
                    retained=retained,
                    shares=shares,
                    scores=scores,
                    candidates=candidates,
                )
            )
            deadline.check()
        chunk_ids = tuple(candidate.chunk_id for candidate in candidates)
        status = "hit" if chunk_ids else "miss"
        return GraphExpansionResult(
            chunk_ids=chunk_ids,
            diagnostics=GraphExpansionDiagnostics(
                status=status,
                seed_count=len(request.seeds),
                candidate_count=len(chunk_ids),
                algorithm_signature=algorithm_signature,
                graph_version_signature=graph_version_signature,
            ),
            seed_chunk_ids=tuple(seed.chunk_id for seed in request.seeds),
        )
    except _SnapshotMiss:
        return _empty_result(
            request,
            status="miss",
            algorithm_signature=algorithm_signature,
        )
    except TimeoutError:
        return _empty_result(
            request,
            status="timeout",
            algorithm_signature=algorithm_signature,
        )
    except Exception:
        return _empty_result(
            request,
            status="error",
            algorithm_signature=algorithm_signature,
        )


def _storage_scope_signature(
    *,
    scope: _AuthorizedStorageScope,
) -> str:
    """Hash only the radius-invariant authorized artifact snapshot."""

    payload = {
        "collection_artifacts": [
            [
                row["id"],
                row["collection_scope_id"],
                row["build_key"],
                row["source_hash"],
                row["ontology_version"],
                row["ontology_checksum"],
                row["resolver_version"],
                row["assembly_version"],
                row["assembly_config_checksum"],
            ]
            for row in scope.collection_artifacts
        ],
        "document_membership": [
            [str(document_id), collection_id]
            for document_id, collection_id in scope.document_membership
        ],
        "manifests": [
            [
                row["id"],
                row["artifact_id"],
                row["collection_id"],
                str(row["document_id"]),
                row["document_artifact_id"],
                row["source_signature"],
                row["membership_signature"],
                row["build_signature"],
                row["document_artifact__build_key"],
                row["document_artifact__source_hash"],
                row["document_artifact__resolver_version"],
                row["document_artifact__ontology_version"],
                row["document_artifact__ontology_checksum"],
            ]
            for row in scope.manifest_rows
        ],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def load_authorized_graph_snapshot(
    request: GraphExpansionRequest,
    *,
    load_max_hops: int,
) -> AuthorizedGraphSnapshot:
    """Load an immutable authorized graph inside the live snapshot context."""

    state = _require_live_snapshot_state()
    if type(request) is not GraphExpansionRequest:
        raise ValueError("request must be an exact GraphExpansionRequest value")
    if (
        type(load_max_hops) is not int
        or not 1 <= load_max_hops <= state.config.max_hops
    ):
        raise ValueError("load_max_hops exceeds the effective graph hop limit")
    if not _request_fits_config(request, state.config):
        raise _SnapshotMiss
    _deadline = state.deadline
    _deadline.check()

    scope = _load_storage_scope(request, state)
    _deadline.check()
    seed_entity_rows = _load_seed_collection_entity_ids(request, scope, state)
    entities: dict[int, _AuthorizedEntityRow] = {}
    identity_by_entity: dict[int, StableNodeKey] = {}
    canonical_audit: dict[int, tuple[object, ...]] = {}
    _extend_identity_registry(
        tuple(sorted({entity_id for _, entity_id in seed_entity_rows})),
        request=request,
        scope=scope,
        state=state,
        entities=entities,
        identity_by_entity=identity_by_entity,
        canonical_audit=canonical_audit,
        discovery_hop=0,
    )
    identity_discovery_hop = {
        identity: 0 for identity in identity_by_entity.values()
    }
    seed_identities = tuple(
        sorted(
            {
                AuthorizedSeedIdentity(
                    seed_chunk_id=seed_id,
                    identity_key=identity_by_entity[entity_id],
                )
                for seed_id, entity_id in seed_entity_rows
            },
            key=lambda row: (row.seed_chunk_id, row.identity_key),
        )
    )
    if not seed_identities:
        raise _SnapshotMiss

    physical_relations: dict[int, _AuthorizedPhysicalRelation] = {}
    relation_discovery_hop: dict[int, int] = {}
    evidence_by_relation: dict[int, tuple[_AuthorizedEvidenceProjection, ...]] = {}
    evidence_row_count = 0
    directional_rows: list[_DirectionalPhysicalProjection] = []
    frontier = frozenset(row.identity_key for row in seed_identities)
    discovered_identities = set(frontier)

    for hop in range(1, load_max_hops + 1):
        _deadline.check()
        frontier_entity_ids = tuple(
            sorted(
                entity_id
                for entity_id, identity in identity_by_entity.items()
                if identity in frontier
            )
        )
        new_relations = _load_physical_relations(
            frontier_entity_ids,
            tuple(sorted(physical_relations)),
            request=request,
            scope=scope,
            state=state,
            maximum=state.config.max_edges - len(physical_relations),
        )
        endpoint_ids = tuple(
            sorted(
                {
                    endpoint
                    for relation in new_relations
                    for endpoint in (relation.source_id, relation.target_id)
                }
            )
        )
        _extend_identity_registry(
            endpoint_ids,
            request=request,
            scope=scope,
            state=state,
            entities=entities,
            identity_by_entity=identity_by_entity,
            canonical_audit=canonical_audit,
            discovery_hop=hop,
        )
        for identity in identity_by_entity.values():
            identity_discovery_hop.setdefault(identity, hop)
        if len(entities) > state.config.max_nodes:
            raise _SnapshotMiss
        new_relation_ids = tuple(relation.pk for relation in new_relations)
        new_evidence = _load_authorized_relation_evidence(
            new_relation_ids,
            request=request,
            scope=scope,
            state=state,
            maximum=state.config.max_evidence_rows - evidence_row_count,
        )
        evidence_row_count += len(new_evidence)
        for relation in new_relations:
            physical_relations[relation.pk] = relation
            relation_discovery_hop.setdefault(relation.pk, hop)
            evidence_by_relation[relation.pk] = tuple(
                row for row in new_evidence if row.relation_id == relation.pk
            )

        next_frontier: set[StableNodeKey] = set()
        for relation in sorted(
            physical_relations.values(), key=lambda value: value.pk
        ):
            source_key = identity_by_entity[relation.source_id]
            target_key = identity_by_entity[relation.target_id]
            ontology_direction = scope.ontology_directions.get(
                (relation.artifact_id, relation.relation_type)
            )
            if ontology_direction is None:
                raise _SnapshotMiss
            directions = _directions_from_frontier(
                source_key,
                target_key,
                ontology_direction=ontology_direction,
                frontier=frontier,
            )
            for source, target, direction in directions:
                if source == target:
                    continue
                destination_entity_id = (
                    relation.target_id
                    if source == source_key
                    else relation.source_id
                )
                directional_rows.append(
                    _DirectionalPhysicalProjection(
                        source_key=source,
                        relation_type=relation.relation_type,
                        target_key=target,
                        direction=direction,
                        admission_hop=hop,
                        destination_retrieval_utility=(
                            entities[destination_entity_id].retrieval_utility
                        ),
                        evidence=evidence_by_relation[relation.pk],
                    )
                )
                if target not in discovered_identities:
                    next_frontier.add(target)
        discovered_identities.update(next_frontier)
        frontier = frozenset(next_frontier)
        if not frontier:
            break

    relation_groups = _compose_authorized_relation_groups(tuple(directional_rows))
    mentions = _load_authorized_identity_mentions(
        request=request,
        scope=scope,
        state=state,
        entities=entities,
        identity_by_entity=identity_by_entity,
    )
    _deadline.check()
    scope_version_signature = _storage_scope_signature(
        scope=scope,
    )
    identity_keys = tuple(sorted(set(identity_by_entity.values())))
    if not identity_keys:
        raise _SnapshotMiss
    raw_audit_rows: list[tuple[int, str, tuple[object, ...]]] = []
    for audit_value in canonical_audit.values():
        raw_audit_rows.append(
            (int(audit_value[0]), "canonical_link", tuple(audit_value[1:]))
        )
    for relation_id, relation in physical_relations.items():
        discovery_hop = relation_discovery_hop[relation_id]
        raw_audit_rows.append(
            (
                discovery_hop,
                "physical_relation",
                (
                    relation.pk,
                    relation.artifact_id,
                    relation.source_id,
                    relation.relation_type,
                    relation.target_id,
                ),
            )
        )
        for evidence_row in evidence_by_relation[relation_id]:
            raw_audit_rows.append(
                (
                    discovery_hop,
                    "relation_evidence",
                    cast(
                        tuple[object, ...],
                        _canonical_audit_value(
                            evidence_row.semantic_signature
                        ),
                    ),
                )
            )
    for mention in mentions:
        raw_audit_rows.append(
            (
                identity_discovery_hop[mention.identity_key],
                "fallback_mention",
                (
                    mention.identity_key,
                    *_evidence_payload(mention.evidence),
                ),
            )
        )
    sorted_audit_rows = tuple(
        sorted(raw_audit_rows, key=lambda row: (row[0], row[1], repr(row[2])))
    )
    snapshot = AuthorizedGraphSnapshot(
        config=state.config,
        load_max_hops=load_max_hops,
        allowed_doc_ids=request.allowed_doc_ids,
        allowed_collection_ids=request.allowed_collection_ids,
        scope_version_signature=scope_version_signature,
        identity_keys=identity_keys,
        seed_identities=seed_identities,
        relation_groups=relation_groups,
        mentions=mentions,
        raw_audit_rows=sorted_audit_rows,
    )
    _deadline.check()
    return snapshot


def _request_fits_config(
    request: GraphExpansionRequest,
    config: PPRAlgorithmConfig,
) -> bool:
    return (
        len(request.seeds) <= config.max_seeds
        and len(request.allowed_doc_ids) <= config.max_scope_documents
        and len(request.allowed_collection_ids) <= config.max_scope_collections
    )


def expand_chunk_candidates(request: GraphExpansionRequest) -> GraphExpansionResult:
    """Fail-open production composition for bounded graph candidate expansion."""

    if type(request) is not GraphExpansionRequest:
        raise ValueError("request must be an exact GraphExpansionRequest value")
    try:
        config = _load_algorithm_config()
        algorithm_signature = graph_algorithm_signature(config)
    except Exception:
        return _empty_result(request, status="error")
    if not _request_fits_config(request, config):
        return _empty_result(
            request,
            status="miss",
            algorithm_signature=algorithm_signature,
        )
    try:
        with authorized_retrieval_snapshot(timeout_ms=config.timeout_ms) as deadline:
            snapshot = load_authorized_graph_snapshot(
                request,
                load_max_hops=config.max_hops,
            )
            return rank_authorized_graph_snapshot(
                snapshot,
                request,
                effective_max_hops=config.max_hops,
                _deadline=deadline,
            )
    except _SnapshotMiss:
        return _empty_result(
            request,
            status="miss",
            algorithm_signature=algorithm_signature,
        )
    except TimeoutError:
        return _empty_result(
            request,
            status="timeout",
            algorithm_signature=algorithm_signature,
        )
    except Exception:
        return _empty_result(
            request,
            status="error",
            algorithm_signature=algorithm_signature,
        )


__all__ = ["expand_chunk_candidates"]
