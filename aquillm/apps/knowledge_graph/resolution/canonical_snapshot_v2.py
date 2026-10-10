"""Bounded immutable source capture, separate from canonical computation."""

import json
from dataclasses import asdict, dataclass
from hashlib import sha256

from django.db.models import F

from . import canonical as c


@dataclass(frozen=True, slots=True)
class CanonicalSourceEntityV2:
    pk: int
    artifact_id: int
    collection_id: int
    cluster_key: str
    label: str
    normalized_label: str
    entity_type: str
    identifier: str
    version_signature: str
    embedding: tuple[float, ...] | None
    embedding_model_signature: str
    embedding_input_hash: str


@dataclass(frozen=True, slots=True)
class CanonicalSourceSnapshotV2:
    artifacts: tuple[tuple[int, int], ...]
    artifact_signatures: tuple[tuple[int, str], ...]
    entities: tuple[CanonicalSourceEntityV2, ...]
    inputs: tuple[c.CanonicalEntityInput, ...]
    identity_policy: str
    revision_digest: str


def capture_source(*, using, identity_policy, expected_artifacts=None):
    from apps.knowledge_graph.models import CollectionEntity, GraphArtifact

    artifacts = c._active_collection_artifact_snapshot(
        using=using, identity_policy=identity_policy
    )
    if expected_artifacts is not None and artifacts != expected_artifacts:
        raise c._CanonicalSnapshotChanged("publication source set changed")
    # The provenance adapter locks links and document artifacts. Take the whole
    # collection prefix first, as lifecycle cleanup does, in both transactions.
    _lock_source_artifacts(artifacts, using=using)
    if (
        c._active_collection_artifact_snapshot(
            using=using, identity_policy=identity_policy
        )
        != artifacts
    ):
        raise c._CanonicalSnapshotChanged("source changed before capture")
    ids = tuple(a for a, _ in artifacts)
    artifact_rows = tuple(
        GraphArtifact.objects.using(using)
        .filter(pk__in=ids)
        .order_by("pk")
        .values(
            "pk",
            "collection_scope_id",
            "scope_id",
            "source_hash",
            "build_key",
            "build_generation",
            "ontology_checksum",
            "resolver_version",
            "resolution_config_checksum",
            "embedding_model_signature",
            "representation_digest",
        )
    )
    fields = tuple(CanonicalSourceEntityV2.__dataclass_fields__)
    raw = c._bounded_locked_rows(
        CollectionEntity.objects.using(using)
        .filter(
            artifact_id__in=ids,
            artifact__status="active",
            status="active",
            collection_id=F("artifact__collection_scope_id"),
        )
        .order_by("pk")
        .values(*fields)
        .iterator(chunk_size=1000),
        maximum=c.MAX_CANONICAL_REBUILD_ENTITIES,
        label="canonical source entities",
    )
    entities = tuple(
        CanonicalSourceEntityV2(
            **{
                **row,
                "embedding": None
                if row["embedding"] is None
                else tuple(float(v) for v in row["embedding"]),
            }
        )
        for row in raw
    )
    inputs = c._load_locked_canonical_inputs(
        entity_rows=entities, active_artifact_ids=ids, using=using
    )
    if (
        c._active_collection_artifact_snapshot(
            using=using, identity_policy=identity_policy
        )
        != artifacts
    ):
        raise c._CanonicalSnapshotChanged("source changed during capture")
    payload = {
        "version": "canonical-source-v2",
        "policy": identity_policy,
        "artifacts": artifact_rows,
        "entities": [asdict(e) for e in entities],
        "inputs": [asdict(i) for i in inputs],
    }
    digest = sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            ensure_ascii=True,
        ).encode()
    ).hexdigest()
    return CanonicalSourceSnapshotV2(
        artifacts,
        tuple((r["pk"], r["embedding_model_signature"]) for r in artifact_rows),
        entities,
        inputs,
        identity_policy,
        digest,
    )


def _lock_source_artifacts(artifacts, *, using):
    from django.db import connections

    from apps.collections.models import Collection
    from apps.knowledge_graph.graph.assembly import _lock_collection_scope
    from apps.knowledge_graph.models import CollectionEntity, GraphArtifact

    collection_ids = tuple(sorted(cid for _, cid in artifacts))
    artifact_ids = tuple(sorted(aid for aid, _ in artifacts))
    with connections[using].cursor() as cursor:
        for cid in collection_ids:
            _lock_collection_scope(cursor, cid)
    locked = tuple(
        Collection.objects.using(using)
        .select_for_update()
        .filter(pk__in=collection_ids)
        .order_by("pk")
        .values_list("pk", flat=True)
    )
    if locked != collection_ids:
        raise c._CanonicalSnapshotChanged("collection source changed")
    locked = tuple(
        GraphArtifact.objects.using(using)
        .select_for_update()
        .filter(pk__in=artifact_ids)
        .order_by("pk")
        .values_list("pk", flat=True)
    )
    if locked != artifact_ids:
        raise c._CanonicalSnapshotChanged("artifact source changed")
    c._bounded_locked_rows(
        CollectionEntity.objects.using(using)
        .select_for_update()
        .filter(artifact_id__in=artifact_ids)
        .order_by("pk")
        .values_list("pk", flat=True),
        maximum=c.MAX_CANONICAL_REBUILD_ENTITIES,
        label="canonical publication entities",
    )
