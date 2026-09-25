"""Canonical lock protocol for immutable collection artifact manifests."""

from __future__ import annotations

import uuid
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

_QUERY_BATCH_SIZE = 5_000
_MAX_MANIFEST_ROWS = 10_000
_MAX_GRAPH_RUN_ROWS = 10_000


class ManifestLockError(RuntimeError):
    """The collection manifest could not be fenced as one exact snapshot."""


@dataclass(frozen=True, slots=True)
class LockedCollectionManifest(Sequence[object]):
    """Manifest rows plus every document-side row locked before them."""

    rows: tuple[object, ...]
    document_artifacts: tuple[object, ...]
    document_runs: tuple[object, ...]
    documents: tuple[object, ...]

    def __iter__(self) -> Iterator[object]:
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


@dataclass(frozen=True, slots=True)
class _DocumentRowRef:
    model_label: str
    pkid: int
    document_id: uuid.UUID


def _batches(values: Iterable[object]) -> Iterator[tuple[object, ...]]:
    ordered = tuple(values)
    for offset in range(0, len(ordered), _QUERY_BATCH_SIZE):
        yield ordered[offset : offset + _QUERY_BATCH_SIZE]


def _bounded_rows(queryset, maximum: int, label: str) -> tuple[object, ...]:
    rows = tuple(queryset[: maximum + 1])
    if len(rows) > maximum:
        raise ManifestLockError(f"{label} exceeds the bounded row limit")
    return rows


def _manifest_row_identity(row: object) -> tuple[object, ...]:
    return (
        row.pk,
        row.artifact_id,
        row.collection_id,
        row.document_id,
        row.document_artifact_id,
        row.membership_signature,
        row.source_signature,
        row.build_signature,
    )


def _manifest_identity(rows: Iterable[object]) -> tuple[tuple[object, ...], ...]:
    return tuple(_manifest_row_identity(row) for row in rows)


def _assert_manifest_identity_unchanged(
    expected: tuple[tuple[object, ...], ...],
    current: tuple[tuple[object, ...], ...],
) -> None:
    if current != expected:
        raise ManifestLockError("collection manifest changed during source locking")


def _ordered_document_ids(values: Iterable[uuid.UUID]) -> tuple[uuid.UUID, ...]:
    from apps.knowledge_graph.services.builds import document_graph_advisory_lock_key

    document_ids = tuple(values)
    if any(type(value) is not uuid.UUID for value in document_ids):
        raise ManifestLockError("manifest document identity is not an exact UUID")
    if len(document_ids) != len(set(document_ids)):
        raise ManifestLockError("collection manifest repeats a document identity")
    return tuple(
        sorted(
            document_ids,
            key=lambda value: (document_graph_advisory_lock_key(value), value.int),
        )
    )


def _discover_exact_document_refs(
    document_ids: tuple[uuid.UUID, ...],
) -> tuple[_DocumentRowRef, ...]:
    from apps.documents.models import DESCENDED_FROM_DOCUMENT

    expected = set(document_ids)
    matches: dict[uuid.UUID, list[_DocumentRowRef]] = defaultdict(list)
    total = 0
    for model in sorted(DESCENDED_FROM_DOCUMENT, key=lambda value: value._meta.label_lower):
        for document_id_batch in _batches(document_ids):
            remaining = _MAX_MANIFEST_ROWS - total
            rows = _bounded_rows(
                model._base_manager.filter(id__in=document_id_batch)
                .order_by("pkid", "id")
                .values_list("pkid", "id"),
                remaining,
                "manifest concrete document identity",
            )
            total += len(rows)
            for pkid, document_id in rows:
                matches[document_id].append(
                    _DocumentRowRef(model._meta.label_lower, pkid, document_id)
                )
    if set(matches) != expected or any(len(rows) != 1 for rows in matches.values()):
        raise ManifestLockError(
            "manifest document UUID is absent or ambiguous across concrete models"
        )
    from apps.knowledge_graph.services.builds import document_graph_advisory_lock_key

    return tuple(
        sorted(
            (rows[0] for rows in matches.values()),
            key=lambda row: (
                document_graph_advisory_lock_key(row.document_id),
                row.document_id.int,
                row.model_label,
                row.pkid,
            ),
        )
    )


def _lock_graph_artifacts(artifact_ids: tuple[int, ...]) -> tuple[object, ...]:
    from apps.knowledge_graph.models import GraphArtifact

    locked = []
    for artifact_id_batch in _batches(artifact_ids):
        locked.extend(
            GraphArtifact.objects.select_for_update()
            .filter(pk__in=artifact_id_batch)
            .order_by("pk")
        )
    rows = tuple(sorted(locked, key=lambda row: row.pk))
    if tuple(row.pk for row in rows) != artifact_ids:
        raise ManifestLockError("manifest document artifact changed before locking")
    return rows


def _lock_graph_runs(artifact_ids: tuple[int, ...]) -> tuple[object, ...]:
    from apps.knowledge_graph.models import GraphBuildRun

    run_ids: list[int] = []
    for artifact_id_batch in _batches(artifact_ids):
        remaining = _MAX_GRAPH_RUN_ROWS - len(run_ids)
        run_ids.extend(
            _bounded_rows(
                GraphBuildRun.objects.filter(artifact_id__in=artifact_id_batch)
                .order_by("pk")
                .values_list("pk", flat=True),
                remaining,
                "manifest document build runs",
            )
        )
    ordered_ids = tuple(sorted(run_ids))
    locked = []
    for run_id_batch in _batches(ordered_ids):
        locked.extend(
            GraphBuildRun.objects.select_for_update()
            .filter(pk__in=run_id_batch)
            .order_by("pk")
        )
    rows = tuple(sorted(locked, key=lambda row: row.pk))
    if tuple(row.pk for row in rows) != ordered_ids:
        raise ManifestLockError("manifest document build runs changed before locking")
    return rows


def _lock_exact_document_rows(
    refs: tuple[_DocumentRowRef, ...],
) -> tuple[object, ...]:
    from django.apps import apps as django_apps

    refs_by_model: dict[str, list[_DocumentRowRef]] = defaultdict(list)
    for ref in refs:
        refs_by_model[ref.model_label].append(ref)
    locked = []
    for model_label in sorted(refs_by_model):
        model = django_apps.get_model(model_label)
        ordered_refs = tuple(
            sorted(
                refs_by_model[model_label],
                key=lambda ref: (ref.pkid, ref.document_id.int),
            )
        )
        for ref_batch in _batches(ordered_refs):
            expected = {(ref.pkid, ref.document_id) for ref in ref_batch}
            rows = tuple(
                model._base_manager.select_for_update()
                .filter(pkid__in=tuple(ref.pkid for ref in ref_batch))
                .order_by("pkid", "id")
            )
            if {(row.pkid, row.id) for row in rows} != expected:
                raise ManifestLockError(
                    "manifest concrete document changed before locking"
                )
            locked.extend(rows)
    return tuple(locked)


def _read_manifest(artifact: object, maximum: int, *, for_update: bool):
    from apps.knowledge_graph.models import CollectionArtifactInput

    query = CollectionArtifactInput.objects
    if for_update:
        query = query.select_for_update(of=("self",))
    return _bounded_rows(
        query.select_related("document_artifact", "collection")
        .filter(artifact=artifact)
        .order_by("document_artifact_id", "pk"),
        maximum,
        "collection manifest",
    )


def lock_collection_manifest_sources(
    artifact: object,
    *,
    maximum: int,
    label: str = "collection manifest",
) -> LockedCollectionManifest:
    """Lock a C-artifact-owned manifest in the global C→D→manifest order.

    The caller must already hold the owning collection GraphArtifact row (and
    its collection scope prefix). That owner row fences manifest inserts and
    public manifest mutation is prohibited by the child model manager.
    """

    from django.db import connection

    from apps.knowledge_graph.models import GraphArtifact
    from apps.knowledge_graph.services.builds import (
        lock_document_graph_advisory_scope,
    )

    if type(maximum) is not int or not 1 <= maximum <= _MAX_MANIFEST_ROWS:
        raise ValueError("manifest maximum must be a bounded positive integer")
    if (
        not isinstance(artifact, GraphArtifact)
        or artifact.pk is None
        or artifact.scope_type != GraphArtifact.ScopeType.COLLECTION
    ):
        raise ValueError("manifest owner must be a persisted collection artifact")
    if not connection.in_atomic_block:
        raise ManifestLockError("manifest locking requires an atomic transaction")

    initial_rows = _read_manifest(artifact, maximum, for_update=False)
    initial_identity = _manifest_identity(initial_rows)
    document_ids = _ordered_document_ids(row.document_id for row in initial_rows)
    artifact_ids = tuple(sorted(row.document_artifact_id for row in initial_rows))
    if len(artifact_ids) != len(set(artifact_ids)):
        raise ManifestLockError("collection manifest repeats a document artifact")
    refs = _discover_exact_document_refs(document_ids)

    for document_id in document_ids:
        lock_document_graph_advisory_scope(document_id)
    document_artifacts = _lock_graph_artifacts(artifact_ids)
    document_runs = _lock_graph_runs(artifact_ids)
    documents = _lock_exact_document_rows(refs)
    if _discover_exact_document_refs(document_ids) != refs:
        raise ManifestLockError("manifest concrete document identity changed")

    locked_rows = _read_manifest(artifact, maximum, for_update=True)
    _assert_manifest_identity_unchanged(initial_identity, _manifest_identity(locked_rows))
    if tuple(sorted(row.document_artifact_id for row in locked_rows)) != artifact_ids:
        raise ManifestLockError(f"{label} source artifacts changed during locking")
    return LockedCollectionManifest(
        rows=locked_rows,
        document_artifacts=document_artifacts,
        document_runs=document_runs,
        documents=documents,
    )
