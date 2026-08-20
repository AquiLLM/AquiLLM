"""Transaction-safe document and collection graph lifecycle invalidation."""

from __future__ import annotations

import re
import uuid
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import islice

import structlog
from django.apps import apps as django_apps
from django.core.exceptions import ValidationError
from django.db import DEFAULT_DB_ALIAS, connections, transaction
from django.db.models.deletion import Collector
from django.db.models.signals import post_delete, post_save, pre_delete, pre_save
from django.utils import timezone

logger = structlog.stdlib.get_logger(__name__)

_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_MODEL_LABEL_PATTERN = re.compile(r"^[a-z0-9_]+\.[a-z0-9_]+$")
_DOCUMENT_DELETE_SNAPSHOT = "_kg_document_delete_snapshot"
_DOCUMENT_SAVE_SNAPSHOT = "_kg_document_save_snapshot"
_DOCUMENT_PENDING_EVENT = "_kg_document_pending_event"
_DOCUMENT_ROW_LOCK_CONTEXT = "_kg_document_lifecycle_row_lock"
_ORIGIN_DELETE_CONTEXT = "_kg_origin_delete_context"
_ACTIVE_OR_BUILDING = ("active", "building")
_MAX_LIFECYCLE_ROWS = 10_000
_QUERY_PREDICATE_BATCH_SIZE = 5_000
_MAX_LOCK_SET_RETRIES = 4
_NO_COLLECTION_PARENT = object()
_NO_FIGURE_PARENT_IDENTITY = object()
_FIGURE_PARENT_OWNER_ATTNAMES = (
    "parent_handwritten_notes_document_id",
    "parent_image_upload_document_id",
    "parent_media_upload_document_id",
    "parent_pdf_document_id",
    "parent_raw_text_document_id",
    "parent_tex_document_id",
    "parent_vtt_document_id",
)
_FIGURE_PARENT_MODEL_LABEL_BY_ATTNAME = {
    "parent_handwritten_notes_document_id": (
        "apps_documents.handwrittennotesdocument"
    ),
    "parent_image_upload_document_id": "apps_documents.imageuploaddocument",
    "parent_media_upload_document_id": "apps_documents.mediauploaddocument",
    "parent_pdf_document_id": "apps_documents.pdfdocument",
    "parent_raw_text_document_id": "apps_documents.rawtextdocument",
    "parent_tex_document_id": "apps_documents.texdocument",
    "parent_vtt_document_id": "apps_documents.vttdocument",
}


class _ExpandedCollectionLockSet(RuntimeError):
    def __init__(self, collection_ids: tuple[int, ...]):
        super().__init__("document dependencies expanded while lifecycle locks were acquired")
        self.collection_ids = collection_ids


def _source_hash(value: object, label: str = "source hash") -> str:
    if type(value) is not str or _HASH_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _collection_id(value: object) -> int:
    if type(value) is not int or not 0 < value < 2**63:
        raise ValueError("collection id must be a positive database integer")
    return value


def _collection_ids(values: Iterable[object]) -> tuple[int, ...]:
    canonical = tuple(sorted({_collection_id(value) for value in values}))
    if len(canonical) > _MAX_LIFECYCLE_ROWS:
        raise RuntimeError("lifecycle collection scope exceeds the bounded row limit")
    return canonical


def _predicate_batches(values: Iterable[object]):
    ordered = tuple(values)
    for offset in range(0, len(ordered), _QUERY_PREDICATE_BATCH_SIZE):
        yield ordered[offset : offset + _QUERY_PREDICATE_BATCH_SIZE]


def _bounded_values(
    queryset,
    *,
    label: str,
    maximum: int = _MAX_LIFECYCLE_ROWS,
) -> tuple[object, ...]:
    if type(maximum) is not int or not 0 <= maximum <= _MAX_LIFECYCLE_ROWS:
        raise ValueError("bounded lifecycle maximum is invalid")
    values = tuple(queryset[: maximum + 1])
    if len(values) > maximum:
        raise RuntimeError(f"{label} exceeds the bounded lifecycle row limit")
    return values


def _bounded_batched_values(
    values: Iterable[object],
    query_factory,
    *,
    label: str,
    row_key,
    sort_key,
) -> tuple[object, ...]:
    rows = []
    for value_batch in _predicate_batches(values):
        remaining = _MAX_LIFECYCLE_ROWS - len(rows)
        rows.extend(
            _bounded_values(
                query_factory(value_batch),
                label=label,
                maximum=remaining,
            )
        )
    ordered = tuple(sorted(rows, key=sort_key))
    keys = tuple(row_key(row) for row in ordered)
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"{label} repeats a lifecycle row")
    return ordered


def _bounded_batched_distinct_values(
    values: Iterable[object],
    query_factory,
    *,
    label: str,
    sort_key=None,
) -> tuple[object, ...]:
    rows = set()
    for value_batch in _predicate_batches(values):
        batch_rows = _bounded_values(
            query_factory(value_batch),
            label=label,
            maximum=_MAX_LIFECYCLE_ROWS - len(rows),
        )
        rows.update(batch_rows)
        if len(rows) > _MAX_LIFECYCLE_ROWS:
            raise RuntimeError(f"{label} exceeds the bounded lifecycle row limit")
    return tuple(sorted(rows, key=sort_key))


def _figure_parent_identity(
    value: object,
) -> tuple[object, ...]:
    expected_size = 3 + len(_FIGURE_PARENT_OWNER_ATTNAMES)
    if type(value) is not tuple or len(value) != expected_size:
        raise ValueError("figure parent identity must include provenance and typed owner")
    content_type_id, parent_pkid, parent_id, *owner_ids = value
    populated_owner_ids = [owner_id for owner_id in owner_ids if owner_id is not None]
    if (
        content_type_id is None
        and parent_pkid is None
        and parent_id is None
        and not populated_owner_ids
    ):
        return value
    if (
        type(content_type_id) is not int
        or not 0 < content_type_id < 2**63
        or type(parent_pkid) is not int
        or not 0 < parent_pkid < 2**63
        or type(parent_id) is not uuid.UUID
        or parent_id.version is None
        or len(populated_owner_ids) != 1
        or type(populated_owner_ids[0]) is not int
        or not 0 < populated_owner_ids[0] < 2**63
        or populated_owner_ids[0] != parent_pkid
    ):
        raise ValueError("figure parent identity must be all-null or exact typed keys")
    return value


def _database_alias(value: object) -> str:
    if value is None:
        return DEFAULT_DB_ALIAS
    if type(value) is not str or not value or value not in connections:
        raise ValueError("database alias must name a configured connection")
    return value


@dataclass(frozen=True, slots=True)
class DocumentLifecycleRef:
    """Exact concrete document row identity; UUID alone is not globally unique."""

    concrete_model_label: str
    document_pkid: int
    document_id: uuid.UUID

    def __post_init__(self) -> None:
        if (
            type(self.concrete_model_label) is not str
            or _MODEL_LABEL_PATTERN.fullmatch(self.concrete_model_label) is None
        ):
            raise ValueError("concrete model label must be canonical")
        if type(self.document_pkid) is not int or self.document_pkid <= 0:
            raise ValueError("document pkid must be a positive database integer")
        if type(self.document_id) is not uuid.UUID or self.document_id.version is None:
            raise ValueError("document id must be an exact RFC 4122 UUID")


@dataclass(frozen=True, slots=True)
class DocumentLifecycleEvent:
    """Immutable before/after state captured around one concrete save."""

    document: DocumentLifecycleRef
    old_source_hash: str
    committed_source_hash: str
    old_collection_id: int
    committed_collection_id: int

    def __post_init__(self) -> None:
        if type(self.document) is not DocumentLifecycleRef:
            raise ValueError("document lifecycle reference must be exact")
        _source_hash(self.old_source_hash, "old source hash")
        _source_hash(self.committed_source_hash, "committed source hash")
        _collection_id(self.old_collection_id)
        _collection_id(self.committed_collection_id)


@dataclass(frozen=True, slots=True)
class DocumentGraphCleanupResult:
    affected_collection_ids: tuple[int, ...]
    current_collection_id: int
    source_hash: str
    ingestion_complete: bool
    has_active_document_artifact: bool

    def __post_init__(self) -> None:
        _collection_ids(self.affected_collection_ids)
        _collection_id(self.current_collection_id)
        _source_hash(self.source_hash)
        if type(self.ingestion_complete) is not bool:
            raise ValueError("ingestion_complete must be a boolean")
        if type(self.has_active_document_artifact) is not bool:
            raise ValueError("active document artifact state must be a boolean")


class DocumentChunkState:
    STALE = "stale"
    COMMITTED = "committed"
    NEEDS_REPLACEMENT = "needs_replacement"


def _row_ids(values: Iterable[object], *, label: str) -> tuple[int, ...]:
    canonical = tuple(sorted(set(values)))
    if any(type(value) is not int or not 0 < value < 2**63 for value in canonical):
        raise ValueError(f"{label} must be positive database row IDs")
    if len(canonical) > _MAX_LIFECYCLE_ROWS:
        raise RuntimeError(f"{label} exceeds the bounded lifecycle row limit")
    return canonical


@dataclass(frozen=True, slots=True)
class GraphRowLockSet:
    """Immutable identity of graph orchestration rows locked in one phase."""

    artifact_ids: tuple[int, ...] = ()
    run_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.artifact_ids != _row_ids(
            self.artifact_ids, label="graph artifact lock rows"
        ):
            raise ValueError("graph artifact lock rows must be sorted and unique")
        if self.run_ids != _row_ids(self.run_ids, label="graph run lock rows"):
            raise ValueError("graph run lock rows must be sorted and unique")


@dataclass(frozen=True, slots=True)
class DeleteDocumentSnapshot:
    document: DocumentLifecycleRef
    source_hash: str
    collection_id: int
    figure_parent_identity: tuple[object, ...] | None = None

    def __post_init__(self) -> None:
        if type(self.document) is not DocumentLifecycleRef:
            raise ValueError("delete document reference must be exact")
        _source_hash(self.source_hash)
        _collection_id(self.collection_id)
        if self.document.concrete_model_label == "apps_documents.documentfigure":
            if self.figure_parent_identity is None:
                raise ValueError("DocumentFigure deletion requires typed parent identity")
            _figure_parent_identity(self.figure_parent_identity)
        elif self.figure_parent_identity is not None:
            raise ValueError("only DocumentFigure rows have parent identity")


@dataclass(frozen=True, slots=True)
class DeleteScopeSnapshot:
    collection_rows: tuple[tuple[int, int | None], ...]
    documents: tuple[DeleteDocumentSnapshot, ...]
    locked_collection_ids: tuple[int, ...]
    fence_documents: tuple[DeleteDocumentSnapshot, ...] = ()

    def __post_init__(self) -> None:
        canonical_rows = tuple(
            sorted(
                (
                    _collection_id(collection_id),
                    None if parent_id is None else _collection_id(parent_id),
                )
                for collection_id, parent_id in self.collection_rows
            )
        )
        if canonical_rows != self.collection_rows or len(
            {row[0] for row in canonical_rows}
        ) != len(canonical_rows):
            raise ValueError("delete collection snapshot must be sorted and unique")
        if any(type(row) is not DeleteDocumentSnapshot for row in self.documents):
            raise ValueError("delete document snapshot rows must be exact")
        canonical_documents = tuple(
            sorted(
                self.documents,
                key=lambda row: (
                    row.document.concrete_model_label,
                    row.document.document_pkid,
                    row.document.document_id.int,
                ),
            )
        )
        if canonical_documents != self.documents or len(
            {row.document for row in canonical_documents}
        ) != len(canonical_documents):
            raise ValueError("delete document snapshot must be sorted and unique")
        fence_documents = self.fence_documents or self.documents
        canonical_fence_documents = tuple(
            sorted(
                fence_documents,
                key=lambda row: (
                    row.document.concrete_model_label,
                    row.document.document_pkid,
                    row.document.document_id.int,
                ),
            )
        )
        if canonical_fence_documents != fence_documents or len(
            {row.document for row in canonical_fence_documents}
        ) != len(canonical_fence_documents):
            raise ValueError("delete fence documents must be sorted and unique")
        if not set(self.documents).issubset(canonical_fence_documents):
            raise ValueError("collected documents must be included in the fence set")
        object.__setattr__(self, "fence_documents", canonical_fence_documents)
        canonical_locked = _collection_ids(self.locked_collection_ids)
        if canonical_locked != self.locked_collection_ids:
            raise ValueError("delete collection lock scope must be canonical")
        required = {
            *(row[0] for row in canonical_rows),
            *(row.collection_id for row in canonical_fence_documents),
        }
        if not required.issubset(canonical_locked):
            raise ValueError("delete lock scope must cover every collected source")


@dataclass(frozen=True, slots=True)
class OriginDeleteContext:
    using: str
    snapshot: DeleteScopeSnapshot
    collection_graph_rows: GraphRowLockSet = GraphRowLockSet()
    document_graph_rows: GraphRowLockSet = GraphRowLockSet()
    transaction_token: object | None = None

    def __post_init__(self) -> None:
        _database_alias(self.using)
        if type(self.snapshot) is not DeleteScopeSnapshot:
            raise ValueError("origin delete context requires an exact snapshot")
        if type(self.collection_graph_rows) is not GraphRowLockSet:
            raise ValueError("origin collection graph locks must be exact")
        if type(self.document_graph_rows) is not GraphRowLockSet:
            raise ValueError("origin document graph locks must be exact")


def _document_ref(sender: object, instance: object) -> DocumentLifecycleRef:
    meta = getattr(sender, "_meta", None)
    return DocumentLifecycleRef(
        concrete_model_label=getattr(meta, "label_lower", ""),
        document_pkid=getattr(instance, "pkid", None),
        document_id=getattr(instance, "id", None),
    )


def _concrete_document_model(document: DocumentLifecycleRef):
    try:
        model = django_apps.get_model(document.concrete_model_label)
    except (LookupError, ValueError) as exc:
        raise ValueError("concrete document model is not registered") from exc
    from apps.documents.models import DESCENDED_FROM_DOCUMENT

    if model not in DESCENDED_FROM_DOCUMENT:
        raise ValueError("lifecycle model is not a concrete Document type")
    return model


def load_current_document_state(
    document: DocumentLifecycleRef,
    *,
    using: str = DEFAULT_DB_ALIAS,
) -> tuple[str, int] | None:
    """Reload exactly one concrete source row without the polymorphic UUID cache."""

    current = load_current_document_lifecycle_state(document, using=using)
    return None if current is None else current[:2]


def load_current_document_lifecycle_state(
    document: DocumentLifecycleRef,
    *,
    using: str = DEFAULT_DB_ALIAS,
) -> tuple[str, int, bool] | None:
    """Reload exact graph identity, membership, and chunk-readiness state."""

    alias = _database_alias(using)
    model = _concrete_document_model(document)
    row = (
        model._base_manager.using(alias)
        .filter(pkid=document.document_pkid, id=document.document_id)
        .values("full_text_hash", "collection_id", "ingestion_complete")
        .first()
    )
    if row is None:
        return None
    return (
        _source_hash(row["full_text_hash"]),
        _collection_id(row["collection_id"]),
        bool(row["ingestion_complete"]),
    )


def _assert_unambiguous_document_ref(
    document: DocumentLifecycleRef,
    *,
    using: str,
) -> None:
    _assert_unambiguous_document_refs((document,), using=using)


def _assert_unambiguous_document_refs(
    documents: Iterable[DocumentLifecycleRef],
    *,
    using: str,
) -> None:
    """Batch-audit exact UUID ownership across every concrete document table."""

    from apps.documents.models import DESCENDED_FROM_DOCUMENT

    refs = tuple(documents)
    if any(type(document) is not DocumentLifecycleRef for document in refs):
        raise ValueError("document ambiguity audit requires exact lifecycle references")
    expected = {
        document.document_id: (
            document.concrete_model_label,
            document.document_pkid,
        )
        for document in refs
    }
    if len(expected) != len(refs):
        raise ValueError("document UUID is ambiguous inside the lifecycle scope")
    document_ids = tuple(sorted(expected, key=lambda value: value.int))
    matches: dict[uuid.UUID, list[tuple[str, int]]] = {
        document_id: [] for document_id in document_ids
    }
    match_count = 0
    for model in sorted(DESCENDED_FROM_DOCUMENT, key=lambda row: row._meta.label_lower):
        rows = _bounded_batched_values(
            document_ids,
            lambda document_id_batch, model=model: (
                model._base_manager.using(using)
                .filter(id__in=document_id_batch)
                .order_by("pkid", "id")
                .values_list("pkid", "id")
            ),
            label="document UUID ambiguity rows",
            row_key=lambda row, label=model._meta.label_lower: (label, row[0]),
            sort_key=lambda row: (row[0], row[1].int),
        )
        match_count += len(rows)
        if match_count > _MAX_LIFECYCLE_ROWS:
            raise RuntimeError("document UUID ambiguity rows exceed the bounded limit")
        for pkid, document_id in rows:
            if document_id in matches:
                matches[document_id].append((model._meta.label_lower, pkid))
    if any(matches[document_id] != [expected[document_id]] for document_id in expected):
        raise ValueError("document UUID is absent or ambiguous across concrete models")


class _BoundedDeleteCollector(Collector):
    """Collector variant that caps materialization before Django builds sets."""

    def __init__(self, *, using: str, origin: object):
        super().__init__(using=using, origin=origin)
        self._bounded_source_count = 0

    def add(self, objs, source=None, nullable=False, reverse_dependency=False):
        from django.db.models import QuerySet

        if isinstance(objs, QuerySet):
            bounded = tuple(objs[: _MAX_LIFECYCLE_ROWS + 1])
        else:
            bounded = tuple(islice(iter(objs), _MAX_LIFECYCLE_ROWS + 1))
        if len(bounded) > _MAX_LIFECYCLE_ROWS:
            raise RuntimeError("delete origin exceeds the bounded lifecycle row limit")
        new_objects = super().add(
            bounded,
            source=source,
            nullable=nullable,
            reverse_dependency=reverse_dependency,
        )
        self._bounded_source_count += len(new_objects)
        if self._bounded_source_count > _MAX_LIFECYCLE_ROWS:
            raise RuntimeError("delete origin exceeds the bounded lifecycle row limit")
        return new_objects


def _figure_parent_identity_from_row(row: object) -> tuple[object, ...]:
    def value(name: str):
        return row[name] if isinstance(row, dict) else getattr(row, name)

    return _figure_parent_identity(
        (
            value("parent_content_type_id"),
            value("parent_object_pkid"),
            value("parent_object_id"),
            *(value(attname) for attname in _FIGURE_PARENT_OWNER_ATTNAMES),
        )
    )


def _delete_document_snapshot_from_row(model: type, row: object) -> DeleteDocumentSnapshot:
    def value(name: str):
        return row[name] if isinstance(row, dict) else getattr(row, name)

    label = model._meta.label_lower
    return DeleteDocumentSnapshot(
        document=DocumentLifecycleRef(
            concrete_model_label=label,
            document_pkid=value("pkid"),
            document_id=value("id"),
        ),
        source_hash=value("full_text_hash"),
        collection_id=value("collection_id"),
        figure_parent_identity=(
            _figure_parent_identity_from_row(row)
            if label == "apps_documents.documentfigure"
            else None
        ),
    )


def _collected_document_rows(collector: Collector, *, using: str):
    from apps.documents.models import DESCENDED_FROM_DOCUMENT, DocumentFigure

    snapshots = []
    base_fields = ("pkid", "id", "full_text_hash", "collection_id")
    for model in DESCENDED_FROM_DOCUMENT:
        collected = collector.data.get(model, ())
        pkids = tuple(sorted({row.pkid for row in collected}))
        if not pkids:
            continue
        fields = base_fields
        if model is DocumentFigure:
            fields = (
                *fields,
                "parent_content_type_id",
                "parent_object_pkid",
                "parent_object_id",
                *_FIGURE_PARENT_OWNER_ATTNAMES,
            )
        rows = _bounded_batched_values(
            pkids,
            lambda pkid_batch: (
                model._base_manager.using(using)
                .filter(pkid__in=pkid_batch)
                .order_by("pkid")
                .values(*fields)
            ),
            label=f"{model._meta.label_lower} delete rows",
            row_key=lambda row: row["pkid"],
            sort_key=lambda row: row["pkid"],
        )
        if len(rows) != len(pkids):
            raise RuntimeError("delete origin changed before lifecycle locking")
        snapshots.extend(_delete_document_snapshot_from_row(model, row) for row in rows)
        if len(snapshots) > _MAX_LIFECYCLE_ROWS:
            raise RuntimeError("delete document scope exceeds the bounded row limit")
    return tuple(
        sorted(
            snapshots,
            key=lambda row: (
                row.document.concrete_model_label,
                row.document.document_pkid,
                row.document.document_id.int,
            ),
        )
    )


def _expand_delete_fence_documents(
    documents: tuple[DeleteDocumentSnapshot, ...],
    *,
    using: str,
) -> tuple[DeleteDocumentSnapshot, ...]:
    """Include the exact typed owner of every collected Figure in the fence."""

    snapshots = {snapshot.document: snapshot for snapshot in documents}
    expected_by_model: dict[str, dict[int, uuid.UUID]] = {}
    for snapshot in documents:
        identity = snapshot.figure_parent_identity
        if identity is None or all(value is None for value in identity):
            continue
        _content_type_id, parent_pkid, parent_id, *owner_ids = identity
        selected = [
            (attname, owner_id)
            for attname, owner_id in zip(
                _FIGURE_PARENT_OWNER_ATTNAMES,
                owner_ids,
                strict=True,
            )
            if owner_id is not None
        ]
        if len(selected) != 1 or selected[0][1] != parent_pkid:
            raise RuntimeError("collected Figure has incoherent typed ownership")
        model_label = _FIGURE_PARENT_MODEL_LABEL_BY_ATTNAME[selected[0][0]]
        expected = expected_by_model.setdefault(model_label, {})
        if parent_pkid in expected and expected[parent_pkid] != parent_id:
            raise RuntimeError("collected Figures disagree about their typed owner")
        expected[parent_pkid] = parent_id

    for model_label, expected in sorted(expected_by_model.items()):
        model = django_apps.get_model(model_label)
        rows = _bounded_batched_values(
            tuple(sorted(expected)),
            lambda pkid_batch: (
                model._base_manager.using(using)
                .filter(pkid__in=pkid_batch)
                .order_by("pkid")
                .values("pkid", "id", "full_text_hash", "collection_id")
            ),
            label=f"{model_label} Figure owner fence rows",
            row_key=lambda row: row["pkid"],
            sort_key=lambda row: row["pkid"],
        )
        if len(rows) != len(expected):
            raise RuntimeError("collected Figure typed owner no longer exists")
        for row in rows:
            if row["id"] != expected[row["pkid"]]:
                raise RuntimeError("collected Figure typed owner identity changed")
            owner = _delete_document_snapshot_from_row(model, row)
            prior = snapshots.get(owner.document)
            if prior is not None and prior != owner:
                raise RuntimeError("collected Figure owner state is inconsistent")
            snapshots[owner.document] = owner
        if len(snapshots) > _MAX_LIFECYCLE_ROWS:
            raise RuntimeError("delete document fence exceeds the bounded row limit")

    return tuple(
        sorted(
            snapshots.values(),
            key=lambda row: (
                row.document.concrete_model_label,
                row.document.document_pkid,
                row.document.document_id.int,
            ),
        )
    )


def _collected_collection_rows(collector: Collector, *, using: str):
    from apps.collections.models import Collection

    collected = collector.data.get(Collection, ())
    collection_ids = tuple(sorted({row.pk for row in collected}))
    if not collection_ids:
        return ()
    rows = _bounded_batched_values(
        collection_ids,
        lambda collection_id_batch: (
            Collection.objects.using(using)
            .filter(pk__in=collection_id_batch)
            .order_by("pk")
            .values_list("pk", "parent_id")
        ),
        label="delete collection rows",
        row_key=lambda row: row[0],
        sort_key=lambda row: row[0],
    )
    if len(rows) != len(collection_ids):
        raise RuntimeError("delete origin changed before lifecycle locking")
    return rows


def _delete_dependency_collection_scopes(
    document_ids: tuple[uuid.UUID, ...],
    *,
    using: str,
) -> tuple[int, ...]:
    if not document_ids:
        return ()
    from apps.knowledge_graph.models import CollectionArtifactInput, GraphArtifact

    document_scope_ids = tuple(str(document_id) for document_id in document_ids)
    document_artifact_ids = _bounded_batched_values(
        document_scope_ids,
        lambda scope_id_batch: (
            GraphArtifact.objects.using(using)
            .filter(
                scope_type=GraphArtifact.ScopeType.DOCUMENT,
                scope_id__in=scope_id_batch,
            )
            .order_by("pk")
            .values_list("pk", flat=True)
        ),
        label="delete document artifact rows",
        row_key=lambda value: value,
        sort_key=lambda value: value,
    )
    by_document = _bounded_batched_distinct_values(
        document_ids,
        lambda document_id_batch: (
            CollectionArtifactInput.objects.using(using)
            .filter(document_id__in=document_id_batch)
            .order_by("artifact_id")
            .values_list("artifact_id", flat=True)
            .distinct()
        ),
        label="delete collection input rows",
    )
    by_artifact = _bounded_batched_distinct_values(
        document_artifact_ids,
        lambda document_artifact_id_batch: (
            CollectionArtifactInput.objects.using(using)
            .filter(document_artifact_id__in=document_artifact_id_batch)
            .order_by("artifact_id")
            .values_list("artifact_id", flat=True)
            .distinct()
        ),
        label="delete collection input rows",
    )
    collection_artifact_ids = tuple(sorted({*by_document, *by_artifact}))
    if len(collection_artifact_ids) > _MAX_LIFECYCLE_ROWS:
        raise RuntimeError("delete collection input rows exceed the bounded row limit")
    scopes = _bounded_batched_distinct_values(
        collection_artifact_ids,
        lambda artifact_id_batch: (
            GraphArtifact.objects.using(using)
            .filter(
                pk__in=artifact_id_batch,
                scope_type=GraphArtifact.ScopeType.COLLECTION,
            )
            .exclude(collection_scope_id=None)
            .order_by("collection_scope_id")
            .values_list("collection_scope_id", flat=True)
            .distinct()
        ),
        label="delete dependent collection scopes",
    )
    return _collection_ids(scopes)


def _snapshot_origin_delete_scope(
    origin: object,
    *,
    using: str,
) -> DeleteScopeSnapshot:
    from django.db.models import Model, QuerySet

    from apps.collections.models import Collection
    from apps.documents.models import DESCENDED_FROM_DOCUMENT

    if origin is None:
        raise RuntimeError("source lifecycle deletion requires a valid origin")
    if isinstance(origin, Model):
        roots = (origin,)
    elif isinstance(origin, QuerySet):
        roots = origin
    else:
        raise RuntimeError("source lifecycle deletion requires a valid origin")
    collector = _BoundedDeleteCollector(using=using, origin=origin)
    collector.collect(roots)
    relevant_models = {Collection, *DESCENDED_FROM_DOCUMENT}
    if any(
        getattr(queryset, "model", None) in relevant_models
        for queryset in collector.fast_deletes
    ):
        raise RuntimeError("delete origin fast-deleted a lifecycle source model")
    collection_rows = _collected_collection_rows(collector, using=using)
    documents = _collected_document_rows(collector, using=using)
    fence_documents = _expand_delete_fence_documents(documents, using=using)
    dependencies = _delete_dependency_collection_scopes(
        tuple(
            sorted(
                {row.document.document_id for row in fence_documents},
                key=lambda value: value.int,
            )
        ),
        using=using,
    )
    locked_collection_ids = _collection_ids(
        (
            *(row[0] for row in collection_rows),
            *(row.collection_id for row in fence_documents),
            *dependencies,
        )
    )
    return DeleteScopeSnapshot(
        collection_rows=collection_rows,
        documents=documents,
        locked_collection_ids=locked_collection_ids,
        fence_documents=fence_documents,
    )


def _lock_exact_document_row(
    document: DocumentLifecycleRef,
    *,
    using: str,
    identity_prevalidated: bool = False,
):
    model = _concrete_document_model(document)
    row = (
        model._base_manager.using(using)
        .select_for_update()
        .filter(pkid=document.document_pkid, id=document.document_id)
        .first()
    )
    if row is not None and not identity_prevalidated:
        _assert_unambiguous_document_ref(document, using=using)
    return row


def _lock_exact_document_rows(
    documents: Iterable[DocumentLifecycleRef],
    *,
    using: str,
) -> tuple[object, ...]:
    """Lock an exact document union by concrete model in bounded PK batches."""

    refs = tuple(documents)
    refs_by_model: dict[str, list[DocumentLifecycleRef]] = {}
    for document in refs:
        if type(document) is not DocumentLifecycleRef:
            raise ValueError("document row locks require exact lifecycle references")
        refs_by_model.setdefault(document.concrete_model_label, []).append(document)
    locked = []
    for model_label in sorted(refs_by_model):
        model = django_apps.get_model(model_label)
        ordered_refs = tuple(
            sorted(
                refs_by_model[model_label],
                key=lambda row: (row.document_pkid, row.document_id.int),
            )
        )
        for ref_batch in _predicate_batches(ordered_refs):
            expected = {
                (document.document_pkid, document.document_id)
                for document in ref_batch
            }
            rows = tuple(
                model._base_manager.using(using)
                .select_for_update()
                .filter(
                    pkid__in=tuple(
                        document.document_pkid for document in ref_batch
                    )
                )
                .order_by("pkid", "id")
            )
            if {(row.pkid, row.id) for row in rows} != expected:
                raise RuntimeError(
                    "delete document changed while lifecycle locks acquired"
                )
            locked.extend(rows)
    return tuple(locked)


def _lock_collection_scopes(collection_ids: tuple[int, ...], *, using: str) -> None:
    from apps.knowledge_graph.graph.assembly import _lock_collection_scope

    connection = connections[using]
    if connection.vendor != "postgresql":
        return
    with connection.cursor() as cursor:
        for collection_id in collection_ids:
            _lock_collection_scope(cursor, collection_id)


def _lock_collection_rows(
    collection_ids: tuple[int, ...],
    *,
    using: str,
) -> tuple[tuple[int, int | None], ...]:
    from apps.collections.models import Collection

    return _bounded_batched_values(
        collection_ids,
        lambda collection_id_batch: (
            Collection.objects.using(using)
            .select_for_update()
            .filter(pk__in=collection_id_batch)
            .order_by("pk")
            .values_list("pk", "parent_id")
        ),
        label="collection lock rows",
        row_key=lambda row: row[0],
        sort_key=lambda row: row[0],
    )


def _document_scope_lock_key(document_id: uuid.UUID) -> int:
    from hashlib import sha256

    return (
        int.from_bytes(sha256(document_id.bytes).digest()[:4], "big", signed=False)
        & 0x7FFFFFFF
    )


def _lock_document_scope(document_id: uuid.UUID, *, using: str) -> None:
    from apps.knowledge_graph.services.builds import _DOCUMENT_LOCK_NAMESPACE

    connection = connections[using]
    if connection.vendor != "postgresql":
        return
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT pg_advisory_xact_lock(%s, %s)",
            [_DOCUMENT_LOCK_NAMESPACE, _document_scope_lock_key(document_id)],
        )


def _graph_run_ids(
    artifact_ids: tuple[int, ...],
    *,
    using: str,
    label: str,
) -> tuple[int, ...]:
    from apps.knowledge_graph.models import GraphBuildRun

    return _bounded_batched_values(
        artifact_ids,
        lambda artifact_id_batch: (
            GraphBuildRun.objects.using(using)
            .filter(artifact_id__in=artifact_id_batch)
            .order_by("pk")
            .values_list("pk", flat=True)
        ),
        label=f"{label} build runs",
        row_key=lambda value: value,
        sort_key=lambda value: value,
    )


def _lock_graph_row_ids(
    artifact_ids: tuple[int, ...],
    *,
    using: str,
    label: str,
) -> GraphRowLockSet:
    """Lock graph artifacts globally by PK, followed by their runs globally by PK."""

    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun

    locked_artifacts = _bounded_batched_values(
        artifact_ids,
        lambda artifact_id_batch: (
            GraphArtifact.objects.using(using)
            .select_for_update()
            .filter(pk__in=artifact_id_batch)
            .order_by("pk")
        ),
        label=f"{label} artifacts",
        row_key=lambda row: row.pk,
        sort_key=lambda row: row.pk,
    )
    locked_artifact_ids = _row_ids(
        (artifact.pk for artifact in locked_artifacts),
        label=f"{label} artifact rows",
    )
    if locked_artifact_ids != artifact_ids:
        raise RuntimeError(f"{label} artifact rows changed before locking")
    run_ids = _graph_run_ids(artifact_ids, using=using, label=label)
    locked_runs = _bounded_batched_values(
        run_ids,
        lambda run_id_batch: (
            GraphBuildRun.objects.using(using)
            .select_for_update()
            .filter(pk__in=run_id_batch)
            .order_by("pk")
        ),
        label=f"{label} build runs",
        row_key=lambda row: row.pk,
        sort_key=lambda row: row.pk,
    )
    locked_run_ids = _row_ids(
        (run.pk for run in locked_runs),
        label=f"{label} run rows",
    )
    if locked_run_ids != run_ids:
        raise RuntimeError(f"{label} build runs changed before locking")
    return GraphRowLockSet(artifact_ids=artifact_ids, run_ids=run_ids)


def _collection_graph_artifact_ids(
    collection_ids: tuple[int, ...],
    *,
    using: str,
) -> tuple[int, ...]:
    from apps.knowledge_graph.models import GraphArtifact

    return _bounded_batched_values(
        _collection_ids(collection_ids),
        lambda collection_id_batch: (
            GraphArtifact.objects.using(using)
            .filter(
                scope_type=GraphArtifact.ScopeType.COLLECTION,
                collection_scope_id__in=collection_id_batch,
            )
            .order_by("pk")
            .values_list("pk", flat=True)
        ),
        label="collection graph artifacts",
        row_key=lambda value: value,
        sort_key=lambda value: value,
    )


def _lock_collection_graph_rows(
    collection_ids: tuple[int, ...],
    *,
    using: str,
) -> GraphRowLockSet:
    artifact_ids = _collection_graph_artifact_ids(collection_ids, using=using)
    return _lock_graph_row_ids(
        artifact_ids,
        using=using,
        label="collection graph",
    )


def _document_graph_artifact_ids(
    documents: tuple[DocumentLifecycleRef, ...],
    *,
    using: str,
) -> tuple[int, ...]:
    from apps.knowledge_graph.models import GraphArtifact

    if any(type(document) is not DocumentLifecycleRef for document in documents):
        raise ValueError("document graph locks require exact lifecycle references")
    scope_ids = tuple(sorted({str(document.document_id) for document in documents}))
    if len(scope_ids) > _MAX_LIFECYCLE_ROWS:
        raise RuntimeError("document graph scope exceeds the bounded row limit")
    return _bounded_batched_values(
        scope_ids,
        lambda scope_id_batch: (
            GraphArtifact.objects.using(using)
            .filter(
                scope_type=GraphArtifact.ScopeType.DOCUMENT,
                scope_id__in=scope_id_batch,
            )
            .order_by("pk")
            .values_list("pk", flat=True)
        ),
        label="document graph artifacts",
        row_key=lambda value: value,
        sort_key=lambda value: value,
    )


def _lock_document_graph_rows(
    documents: tuple[DocumentLifecycleRef, ...],
    *,
    using: str,
) -> GraphRowLockSet:
    artifact_ids = _document_graph_artifact_ids(documents, using=using)
    return _lock_graph_row_ids(
        artifact_ids,
        using=using,
        label="document graph",
    )


def _discover_collection_graph_rows(
    collection_ids: tuple[int, ...],
    *,
    using: str,
) -> GraphRowLockSet:
    artifact_ids = _collection_graph_artifact_ids(collection_ids, using=using)
    return GraphRowLockSet(
        artifact_ids=artifact_ids,
        run_ids=_graph_run_ids(
            artifact_ids,
            using=using,
            label="collection graph",
        ),
    )


def _discover_document_graph_rows(
    documents: tuple[DocumentLifecycleRef, ...],
    *,
    using: str,
) -> GraphRowLockSet:
    artifact_ids = _document_graph_artifact_ids(documents, using=using)
    return GraphRowLockSet(
        artifact_ids=artifact_ids,
        run_ids=_graph_run_ids(
            artifact_ids,
            using=using,
            label="document graph",
        ),
    )


def _lock_delete_document_scopes(
    documents: tuple[DeleteDocumentSnapshot, ...],
    *,
    using: str,
) -> None:
    document_ids = tuple(
        sorted(
            {row.document.document_id for row in documents},
            key=lambda value: (_document_scope_lock_key(value), value.int),
        )
    )
    for document_id in document_ids:
        _lock_document_scope(document_id, using=using)


def _lock_delete_document_rows(
    documents: tuple[DeleteDocumentSnapshot, ...],
    *,
    using: str,
) -> None:
    _assert_unambiguous_document_refs(
        tuple(snapshot.document for snapshot in documents),
        using=using,
    )
    _lock_exact_document_rows(
        tuple(snapshot.document for snapshot in documents),
        using=using,
    )


def _lock_document_graph_phases(
    document: DocumentLifecycleRef,
    collection_ids: tuple[int, ...],
    *,
    using: str,
) -> tuple[GraphRowLockSet, GraphRowLockSet, object | None]:
    """Apply the Task10-prefix/Task11-suffix lifecycle lock order."""

    if type(document) is not DocumentLifecycleRef:
        raise ValueError("document graph locking requires an exact lifecycle reference")
    canonical_collections = _collection_ids(collection_ids)
    _lock_collection_scopes(canonical_collections, using=using)
    _lock_collection_rows(canonical_collections, using=using)
    collection_graph = _lock_collection_graph_rows(
        canonical_collections,
        using=using,
    )
    _lock_document_scope(document.document_id, using=using)
    document_graph = _lock_document_graph_rows((document,), using=using)
    row = _lock_exact_document_row(document, using=using)
    return collection_graph, document_graph, row


def _delete_transaction_token(*, using: str) -> object:
    connection = connections[using]
    if not connection.in_atomic_block or not connection.atomic_blocks:
        raise RuntimeError("source lifecycle delete must run in its Collector transaction")
    return connection.atomic_blocks[-1]


def _ensure_origin_delete_scope_locked(
    origin: object,
    *,
    using: str = DEFAULT_DB_ALIAS,
) -> OriginDeleteContext:
    alias = _database_alias(using)
    if origin is None:
        raise RuntimeError("source lifecycle deletion requires a valid origin")
    token = _delete_transaction_token(using=alias)
    existing = getattr(origin, _ORIGIN_DELETE_CONTEXT, None)
    if (
        type(existing) is OriginDeleteContext
        and existing.using == alias
        and existing.transaction_token is token
    ):
        return existing

    snapshot = _snapshot_origin_delete_scope(origin, using=alias)
    _lock_collection_scopes(snapshot.locked_collection_ids, using=alias)
    locked_collection_rows = _lock_collection_rows(
        snapshot.locked_collection_ids,
        using=alias,
    )
    if {row[0] for row in locked_collection_rows} != set(
        snapshot.locked_collection_ids
    ):
        raise RuntimeError("delete collection changed while lifecycle locks acquired")
    collection_graph_rows = _lock_collection_graph_rows(
        snapshot.locked_collection_ids,
        using=alias,
    )
    _lock_delete_document_scopes(snapshot.fence_documents, using=alias)
    document_graph_rows = _lock_document_graph_rows(
        tuple(row.document for row in snapshot.fence_documents),
        using=alias,
    )
    _lock_delete_document_rows(snapshot.fence_documents, using=alias)
    locked_snapshot = _snapshot_origin_delete_scope(origin, using=alias)
    if locked_snapshot != snapshot:
        raise RuntimeError("delete scope changed while lifecycle locks were acquired")
    if _discover_collection_graph_rows(
        snapshot.locked_collection_ids,
        using=alias,
    ) != collection_graph_rows:
        raise RuntimeError("collection graph changed while lifecycle locks were acquired")
    if _discover_document_graph_rows(
        tuple(row.document for row in snapshot.fence_documents),
        using=alias,
    ) != document_graph_rows:
        raise RuntimeError("document graph changed while lifecycle locks were acquired")
    context = OriginDeleteContext(
        using=alias,
        snapshot=snapshot,
        collection_graph_rows=collection_graph_rows,
        document_graph_rows=document_graph_rows,
        transaction_token=token,
    )
    try:
        setattr(origin, _ORIGIN_DELETE_CONTEXT, context)
    except (AttributeError, TypeError) as exc:
        raise RuntimeError("source lifecycle deletion origin cannot retain lock state") from exc
    return context


def _assert_origin_graph_rows_current(context: OriginDeleteContext) -> None:
    if type(context) is not OriginDeleteContext:
        raise ValueError("origin graph assertions require an exact context")
    current_collection = _discover_collection_graph_rows(
        context.snapshot.locked_collection_ids,
        using=context.using,
    )
    current_document = _discover_document_graph_rows(
        tuple(row.document for row in context.snapshot.fence_documents),
        using=context.using,
    )
    if not set(current_collection.artifact_ids).issubset(
        context.collection_graph_rows.artifact_ids
    ) or not set(current_collection.run_ids).issubset(
        context.collection_graph_rows.run_ids
    ):
        raise RuntimeError("collection graph expanded outside the locked delete scope")
    if not set(current_document.artifact_ids).issubset(
        context.document_graph_rows.artifact_ids
    ) or not set(current_document.run_ids).issubset(
        context.document_graph_rows.run_ids
    ):
        raise RuntimeError("document graph expanded outside the locked delete scope")


def _assert_document_delete_scope(
    context: OriginDeleteContext,
    snapshot: DeleteDocumentSnapshot,
) -> None:
    if type(context) is not OriginDeleteContext or type(snapshot) is not DeleteDocumentSnapshot:
        raise ValueError("delete scope assertions require exact lifecycle snapshots")
    if snapshot not in context.snapshot.documents:
        raise RuntimeError("document is not part of the locked delete scope")
    if snapshot.collection_id not in context.snapshot.locked_collection_ids:
        raise RuntimeError("document collection is outside the locked delete scope")


def _assert_collection_delete_scope(
    context: OriginDeleteContext,
    collection_row: tuple[int, int | None],
) -> None:
    collection_id, parent_id = collection_row
    canonical = (
        _collection_id(collection_id),
        None if parent_id is None else _collection_id(parent_id),
    )
    if type(context) is not OriginDeleteContext:
        raise ValueError("delete scope assertion requires an exact origin context")
    if canonical not in context.snapshot.collection_rows:
        raise RuntimeError("collection is not part of the locked delete scope")


def _terminalize_artifact_runs(
    artifact_ids: tuple[int, ...],
    *,
    reason: str,
    using: str,
) -> None:
    if not artifact_ids:
        return
    run_model = django_apps.get_model("apps_knowledge_graph", "GraphBuildRun")
    runs = _bounded_batched_values(
        artifact_ids,
        lambda artifact_id_batch: (
            run_model.objects.using(using)
            .select_for_update()
            .filter(artifact_id__in=artifact_id_batch)
            .order_by("pk")
        ),
        label="artifact build runs",
        row_key=lambda row: row.pk,
        sort_key=lambda row: row.pk,
    )
    _terminalize_runs(runs, reason=reason, using=using)


def _terminalize_runs(
    runs: Iterable[object],
    *,
    reason: str,
    using: str,
) -> None:
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun
    from apps.knowledge_graph.services.builds import _apply_locked_terminal

    terminal = {
        GraphBuildRun.Stage.FAILED,
        GraphBuildRun.Stage.SUPERSEDED,
        GraphBuildRun.Stage.STALE,
    }
    now = timezone.now()
    for run in runs:
        if (
            run.orchestration_version
            == GraphArtifact.OrchestrationVersion.SCOPED_V1
        ):
            if run.stage not in terminal:
                _apply_locked_terminal(
                    run,
                    GraphBuildRun.Stage.STALE,
                    error_code=reason,
                )
            elif run.lease_owner or run.lease_expires_at is not None:
                run.lease_owner = ""
                run.lease_expires_at = None
                run.save(update_fields=["lease_owner", "lease_expires_at"])
            continue
        if run.status in {
            GraphBuildRun.Status.FAILED,
            GraphBuildRun.Status.CANCELLED,
        }:
            if run.lease_owner or run.lease_expires_at is not None:
                run.lease_owner = ""
                run.lease_expires_at = None
                run.save(update_fields=["lease_owner", "lease_expires_at"])
            continue
        run.stage = GraphBuildRun.Stage.STALE
        run.status = GraphBuildRun.Status.CANCELLED
        run.error_code = reason
        run.error_message = reason
        run.finished_at = run.finished_at or now
        run.lease_owner = ""
        run.lease_expires_at = None
        run.save(
            update_fields=[
                "stage",
                "status",
                "error_code",
                "error_message",
                "finished_at",
                "lease_owner",
                "lease_expires_at",
            ]
        )


def _load_graph_rows(
    locks: GraphRowLockSet,
    *,
    using: str,
    allow_missing: bool = False,
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    """Load rows already locked by this transaction without changing lock order."""

    if type(locks) is not GraphRowLockSet:
        raise ValueError("graph row loading requires an exact lock set")
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun

    artifacts = _bounded_batched_values(
        locks.artifact_ids,
        lambda artifact_id_batch: (
            GraphArtifact.objects.using(using)
            .filter(pk__in=artifact_id_batch)
            .order_by("pk")
        ),
        label="locked graph artifacts",
        row_key=lambda row: row.pk,
        sort_key=lambda row: row.pk,
    )
    runs = _bounded_batched_values(
        locks.run_ids,
        lambda run_id_batch: (
            GraphBuildRun.objects.using(using)
            .filter(pk__in=run_id_batch)
            .order_by("pk")
        ),
        label="locked graph runs",
        row_key=lambda row: row.pk,
        sort_key=lambda row: row.pk,
    )
    if not allow_missing and (
        tuple(artifact.pk for artifact in artifacts) != locks.artifact_ids
        or tuple(run.pk for run in runs) != locks.run_ids
    ):
        raise RuntimeError("locked graph rows changed during lifecycle cleanup")
    return artifacts, runs


def _runs_for_artifacts(
    runs: Iterable[object],
    artifact_ids: Iterable[int],
) -> tuple[object, ...]:
    selected = frozenset(_row_ids(artifact_ids, label="terminal artifact rows"))
    return tuple(run for run in runs if run.artifact_id in selected)


def _delete_artifacts(artifacts: tuple[object, ...], *, using: str) -> None:
    if not artifacts:
        return
    collection_artifact_ids = tuple(sorted(artifact.pk for artifact in artifacts if artifact.scope_type == "collection")); collection_artifact_ids and __import__("apps.knowledge_graph.projection.lifecycle", fromlist=["supersede_artifact_projections_locked"]).supersede_artifact_projections_locked(artifact_ids=collection_artifact_ids, now=timezone.now(), using=using); collector = Collector(using=using)  # noqa: E501, E702
    collector.collect(artifacts)
    collector.delete()


def _document_artifact_ids(
    document_id: uuid.UUID,
    *,
    using: str,
) -> tuple[int, ...]:
    from apps.knowledge_graph.models import GraphArtifact

    return tuple(
        _bounded_values(
            GraphArtifact.objects.using(using)
            .filter(
                scope_type=GraphArtifact.ScopeType.DOCUMENT,
                scope_id=str(document_id),
            )
            .order_by("pk")
            .values_list("pk", flat=True),
            label="document graph artifacts",
        )
    )


def _dependent_collection_artifact_ids(
    document_id: uuid.UUID,
    document_artifact_ids: tuple[int, ...],
    *,
    using: str,
    active_or_building_only: bool,
) -> tuple[int, ...]:
    from apps.knowledge_graph.models import CollectionArtifactInput, GraphArtifact

    input_artifact_ids = set(
        _bounded_values(
            CollectionArtifactInput.objects.using(using)
            .filter(document_id=document_id)
            .order_by("artifact_id")
            .values_list("artifact_id", flat=True)
            .distinct(),
            label="document collection manifest inputs",
        )
    )
    input_artifact_ids.update(
        _bounded_batched_distinct_values(
            document_artifact_ids,
            lambda document_artifact_id_batch: (
                CollectionArtifactInput.objects.using(using)
                .filter(document_artifact_id__in=document_artifact_id_batch)
                .order_by("artifact_id")
                .values_list("artifact_id", flat=True)
                .distinct()
            ),
            label="document collection manifest inputs",
        )
    )
    if len(input_artifact_ids) > _MAX_LIFECYCLE_ROWS:
        raise RuntimeError("document manifest inputs exceed the bounded row limit")

    def artifact_query(artifact_id_batch):
        query = GraphArtifact.objects.using(using).filter(
            pk__in=artifact_id_batch,
            scope_type=GraphArtifact.ScopeType.COLLECTION,
        )
        if active_or_building_only:
            query = query.filter(status__in=_ACTIVE_OR_BUILDING)
        return query.order_by("pk").values_list("pk", flat=True)

    return _bounded_batched_values(
        tuple(sorted(input_artifact_ids)),
        artifact_query,
        label="dependent collection graph artifacts",
        row_key=lambda value: value,
        sort_key=lambda value: value,
    )


def _collection_scopes_for_artifacts(
    artifact_ids: tuple[int, ...],
    *,
    using: str,
) -> tuple[int, ...]:
    from apps.knowledge_graph.models import GraphArtifact

    values = _bounded_batched_distinct_values(
        artifact_ids,
        lambda artifact_id_batch: (
            GraphArtifact.objects.using(using)
            .filter(
                pk__in=artifact_id_batch,
                scope_type=GraphArtifact.ScopeType.COLLECTION,
            )
            .exclude(collection_scope_id=None)
            .order_by("collection_scope_id")
            .values_list("collection_scope_id", flat=True)
            .distinct()
        ),
        label="dependent collection graph scopes",
    )
    return _collection_ids(values)


def _discover_document_collection_scopes(
    document: DocumentLifecycleRef,
    *,
    using: str,
    active_or_building_only: bool,
) -> tuple[int, ...]:
    document_artifact_ids = _document_artifact_ids(
        document.document_id,
        using=using,
    )
    artifact_ids = _dependent_collection_artifact_ids(
        document.document_id,
        document_artifact_ids,
        using=using,
        active_or_building_only=active_or_building_only,
    )
    return _collection_scopes_for_artifacts(artifact_ids, using=using)


def _initial_document_lock_set(
    document: DocumentLifecycleRef,
    explicit_collection_ids: tuple[int, ...],
    *,
    using: str,
    active_or_building_only: bool,
) -> tuple[int, ...]:
    current = load_current_document_state(document, using=using)
    current_collection_ids = () if current is None else (current[1],)
    dependent = _discover_document_collection_scopes(
        document,
        using=using,
        active_or_building_only=active_or_building_only,
    )
    return _collection_ids(
        (*explicit_collection_ids, *current_collection_ids, *dependent)
    )


def _locked_document_scope_set(
    document: DocumentLifecycleRef,
    row: object,
    explicit_collection_ids: tuple[int, ...],
    *,
    using: str,
    active_or_building_only: bool,
) -> tuple[int, ...]:
    dependent = _discover_document_collection_scopes(
        document,
        using=using,
        active_or_building_only=active_or_building_only,
    )
    return _collection_ids(
        (*explicit_collection_ids, _collection_id(row.collection_id), *dependent)
    )


def _raise_if_lock_set_expanded(
    locked: tuple[int, ...],
    discovered: tuple[int, ...],
) -> None:
    if not set(discovered).issubset(locked):
        raise _ExpandedCollectionLockSet(_collection_ids((*locked, *discovered)))


@contextmanager
def locked_document_lifecycle_row(
    document: DocumentLifecycleRef,
    collection_ids: Iterable[int],
    *,
    using: str = DEFAULT_DB_ALIAS,
    active_or_building_only: bool = True,
):
    """Yield a leaf-only exact row under collection-before-document locks.

    Callers may write that row and its already locked collection FKs, but must
    never acquire graph artifact/run locks after entering this context.
    """

    if type(document) is not DocumentLifecycleRef:
        raise ValueError("document lifecycle reference must be exact")
    if type(active_or_building_only) is not bool:
        raise ValueError("active_or_building_only must be a boolean")
    alias = _database_alias(using)
    explicit = _collection_ids(collection_ids)
    lock_set = _initial_document_lock_set(
        document,
        explicit,
        using=alias,
        active_or_building_only=active_or_building_only,
    )
    for _attempt in range(_MAX_LOCK_SET_RETRIES):
        try:
            with transaction.atomic(using=alias):
                _lock_collection_scopes(lock_set, using=alias)
                _lock_collection_rows(lock_set, using=alias)
                _lock_document_scope(document.document_id, using=alias)
                row = _lock_exact_document_row(document, using=alias)
                if row is None:
                    raise ValueError("exact lifecycle document row no longer exists")
                affected = _locked_document_scope_set(
                    document,
                    row,
                    explicit,
                    using=alias,
                    active_or_building_only=active_or_building_only,
                )
                _raise_if_lock_set_expanded(lock_set, affected)
                marker = (
                    alias,
                    document.concrete_model_label,
                    document.document_pkid,
                    document.document_id,
                )
                previous_marker = getattr(row, _DOCUMENT_ROW_LOCK_CONTEXT, None)
                setattr(row, _DOCUMENT_ROW_LOCK_CONTEXT, marker)
                try:
                    yield row, affected
                finally:
                    if previous_marker is None:
                        try:
                            delattr(row, _DOCUMENT_ROW_LOCK_CONTEXT)
                        except AttributeError:
                            pass
                    else:
                        setattr(row, _DOCUMENT_ROW_LOCK_CONTEXT, previous_marker)
            return
        except _ExpandedCollectionLockSet as exc:
            lock_set = exc.collection_ids
    raise RuntimeError("document lifecycle dependencies did not stabilize")


def document_lifecycle_row_is_locked(
    instance: object,
    *,
    using: str = DEFAULT_DB_ALIAS,
) -> bool:
    """Return whether ``instance`` is the exact row yielded by the leaf lock seam."""

    alias = _database_alias(using)
    meta = getattr(type(instance), "_meta", None)
    expected = (
        alias,
        getattr(meta, "label_lower", None),
        getattr(instance, "pkid", None),
        getattr(instance, "id", None),
    )
    return getattr(instance, _DOCUMENT_ROW_LOCK_CONTEXT, None) == expected


@contextmanager
def locked_collection_parent_rows(
    collection_id: int,
    parent_ids: Iterable[int],
    *,
    using: str = DEFAULT_DB_ALIAS,
):
    """Yield a leaf-only parent snapshot under sorted collection locks."""

    alias = _database_alias(using)
    child_id = _collection_id(collection_id)
    lock_set = _collection_ids((child_id, *tuple(parent_ids)))
    for _attempt in range(_MAX_LOCK_SET_RETRIES):
        try:
            with transaction.atomic(using=alias):
                _lock_collection_scopes(lock_set, using=alias)
                locked_rows = dict(_lock_collection_rows(lock_set, using=alias))
                if set(locked_rows) != set(lock_set) or child_id not in locked_rows:
                    raise RuntimeError(
                        "collection parent scope changed while locks were acquired"
                    )
                current_parent_id = locked_rows[child_id]
                discovered = _collection_ids(
                    (
                        *lock_set,
                        *(() if current_parent_id is None else (current_parent_id,)),
                    )
                )
                _raise_if_lock_set_expanded(lock_set, discovered)
                yield current_parent_id, lock_set
            return
        except _ExpandedCollectionLockSet as exc:
            lock_set = exc.collection_ids
    raise RuntimeError("collection parent dependencies did not stabilize")


def cleanup_collection_graph_state(
    collection_ids: Iterable[int],
    *,
    reason: str,
    using: str = DEFAULT_DB_ALIAS,
    all_artifacts: bool = False,
    expected_parent_id: object = _NO_COLLECTION_PARENT,
    _origin_context: OriginDeleteContext | None = None,
) -> tuple[int, ...]:
    """Delete owned collection graph shadows/current state through a guarded Collector."""

    alias = _database_alias(using)
    affected = _collection_ids(collection_ids)
    if not affected:
        return ()
    if expected_parent_id is not _NO_COLLECTION_PARENT:
        if len(affected) != 1:
            raise ValueError("a collection parent fence requires exactly one collection")
        if expected_parent_id is not None:
            expected_parent_id = _collection_id(expected_parent_id)
    if _origin_context is not None:
        if type(_origin_context) is not OriginDeleteContext:
            raise ValueError("collection cleanup origin context must be exact")
        if _origin_context.using != alias:
            raise RuntimeError("collection cleanup origin uses another database")
        if not set(affected).issubset(
            _origin_context.snapshot.locked_collection_ids
        ):
            raise RuntimeError("collection cleanup exceeds the locked delete scope")
    from apps.collections.models import Collection
    from apps.knowledge_graph.models import GraphArtifact

    with transaction.atomic(using=alias):
        if _origin_context is None:
            _lock_collection_scopes(affected, using=alias)
            locked_rows = dict(_lock_collection_rows(affected, using=alias))
            graph_locks = _lock_collection_graph_rows(affected, using=alias)
            graph_artifacts, graph_runs = _load_graph_rows(
                graph_locks,
                using=alias,
            )
        else:
            _assert_origin_graph_rows_current(_origin_context)
            locked_rows = dict(
                _bounded_batched_values(
                    affected,
                    lambda collection_id_batch: (
                        Collection.objects.using(alias)
                        .filter(pk__in=collection_id_batch)
                        .order_by("pk")
                        .values_list("pk", "parent_id")
                    ),
                    label="locked collection cleanup rows",
                    row_key=lambda row: row[0],
                    sort_key=lambda row: row[0],
                )
            )
            graph_artifacts, graph_runs = _load_graph_rows(
                _origin_context.collection_graph_rows,
                using=alias,
                allow_missing=True,
            )
        if expected_parent_id is not _NO_COLLECTION_PARENT:
            collection_id = affected[0]
            if (
                collection_id not in locked_rows
                or locked_rows[collection_id] != expected_parent_id
            ):
                raise RuntimeError("collection changed during lifecycle deletion")
        artifacts = tuple(
            artifact
            for artifact in graph_artifacts
            if artifact.scope_type == GraphArtifact.ScopeType.COLLECTION
            and artifact.collection_scope_id in affected
            and (all_artifacts or artifact.status in _ACTIVE_OR_BUILDING)
        )
        artifact_ids = tuple(artifact.pk for artifact in artifacts)
        _terminalize_runs(
            _runs_for_artifacts(graph_runs, artifact_ids),
            reason=reason,
            using=alias,
        )
        _delete_artifacts(artifacts, using=alias)
    return affected


def cleanup_document_graph_state(
    document: DocumentLifecycleRef,
    collection_ids: Iterable[int],
    *,
    reason: str,
    using: str = DEFAULT_DB_ALIAS,
    expected_source_hash: str | None = None,
    expected_collection_id: int | None = None,
    fail_on_stale: bool = False,
    _skip_if_committed_chunks: bool = False,
    expected_parent_identity: object = _NO_FIGURE_PARENT_IDENTITY,
    _origin_context: OriginDeleteContext | None = None,
) -> DocumentGraphCleanupResult | tuple[()] | None:
    """Fence/delete dependent collection state, document artifacts, then chunks."""

    if type(document) is not DocumentLifecycleRef:
        raise ValueError("document lifecycle reference must be exact")
    alias = _database_alias(using)
    explicit_collection_ids = _collection_ids(collection_ids)
    expected_hash = (
        None
        if expected_source_hash is None
        else _source_hash(expected_source_hash, "expected source hash")
    )
    expected_collection = (
        None
        if expected_collection_id is None
        else _collection_id(expected_collection_id)
    )
    if type(fail_on_stale) is not bool:
        raise ValueError("fail_on_stale must be a boolean")
    if type(_skip_if_committed_chunks) is not bool:
        raise ValueError("_skip_if_committed_chunks must be a boolean")
    if expected_parent_identity is not _NO_FIGURE_PARENT_IDENTITY:
        if document.concrete_model_label != "apps_documents.documentfigure":
            raise ValueError("only DocumentFigure rows have a parent identity fence")
        expected_parent_identity = _figure_parent_identity(expected_parent_identity)
    if _origin_context is not None:
        if type(_origin_context) is not OriginDeleteContext:
            raise ValueError("document cleanup origin context must be exact")
        if _origin_context.using != alias:
            raise RuntimeError("document cleanup origin uses another database")

    from apps.documents.models import TextChunk
    from apps.knowledge_graph.models import GraphArtifact

    if _origin_context is None:
        lock_set = _initial_document_lock_set(
            document,
            explicit_collection_ids,
            using=alias,
            active_or_building_only=False,
        )
        attempts = _MAX_LOCK_SET_RETRIES
    else:
        lock_set = _origin_context.snapshot.locked_collection_ids
        if not set(explicit_collection_ids).issubset(lock_set):
            raise RuntimeError("document cleanup exceeds the locked delete scope")
        attempts = 1
    for _attempt in range(attempts):
        try:
            with transaction.atomic(using=alias):
                if _origin_context is None:
                    collection_locks, document_locks, row = (
                        _lock_document_graph_phases(
                            document,
                            lock_set,
                            using=alias,
                        )
                    )
                    locked_collection_artifacts, locked_collection_runs = (
                        _load_graph_rows(collection_locks, using=alias)
                    )
                    locked_document_artifacts, locked_document_runs = (
                        _load_graph_rows(document_locks, using=alias)
                    )
                else:
                    _assert_origin_graph_rows_current(_origin_context)
                    locked_collection_artifacts, locked_collection_runs = (
                        _load_graph_rows(
                            _origin_context.collection_graph_rows,
                            using=alias,
                            allow_missing=True,
                        )
                    )
                    locked_document_artifacts, locked_document_runs = (
                        _load_graph_rows(
                            _origin_context.document_graph_rows,
                            using=alias,
                            allow_missing=True,
                        )
                    )
                    row = _lock_exact_document_row(
                        document,
                        using=alias,
                        identity_prevalidated=True,
                    )
                if row is None:
                    if fail_on_stale:
                        raise RuntimeError("document changed during lifecycle deletion")
                    return ()
                locked_affected = _locked_document_scope_set(
                    document,
                    row,
                    explicit_collection_ids,
                    using=alias,
                    active_or_building_only=False,
                )
                _raise_if_lock_set_expanded(lock_set, locked_affected)
                stale = (
                    (expected_hash is not None and row.full_text_hash != expected_hash)
                    or (
                        expected_collection is not None
                        and row.collection_id != expected_collection
                    )
                    or (
                        expected_parent_identity is not _NO_FIGURE_PARENT_IDENTITY
                        and (
                            row.parent_content_type_id,
                            row.parent_object_pkid,
                            row.parent_object_id,
                            *(
                                getattr(row, attname)
                                for attname in _FIGURE_PARENT_OWNER_ATTNAMES
                            ),
                        )
                        != expected_parent_identity
                    )
                )
                if stale and fail_on_stale:
                    raise RuntimeError("document changed during lifecycle deletion")
                if stale:
                    return ()
                if (
                    _skip_if_committed_chunks
                    and row.ingestion_complete
                    and TextChunk.objects.using(alias)
                    .filter(doc_id=document.document_id)
                    .exists()
                ):
                    return None

                document_artifacts = tuple(
                    artifact
                    for artifact in locked_document_artifacts
                    if artifact.scope_type == GraphArtifact.ScopeType.DOCUMENT
                    and artifact.scope_id == str(document.document_id)
                )
                document_artifact_ids = tuple(
                    artifact.pk for artifact in document_artifacts
                )
                dependent_collection_artifact_ids = (
                    _dependent_collection_artifact_ids(
                        document.document_id,
                        document_artifact_ids,
                        using=alias,
                        active_or_building_only=False,
                    )
                )
                dependency_scopes = _collection_scopes_for_artifacts(
                    dependent_collection_artifact_ids,
                    using=alias,
                )
                final_affected = _collection_ids(
                    (*locked_affected, *dependency_scopes)
                )
                _raise_if_lock_set_expanded(lock_set, final_affected)

                collection_artifacts = tuple(
                    artifact
                    for artifact in locked_collection_artifacts
                    if artifact.scope_type == GraphArtifact.ScopeType.COLLECTION
                    and (
                        artifact.pk in dependent_collection_artifact_ids
                        or (
                            artifact.collection_scope_id in final_affected
                            and artifact.status in _ACTIVE_OR_BUILDING
                        )
                    )
                )
                collection_artifact_ids = tuple(
                    artifact.pk for artifact in collection_artifacts
                )
                if not set(dependent_collection_artifact_ids).issubset(
                    {artifact.pk for artifact in locked_collection_artifacts}
                ):
                    raise RuntimeError(
                        "dependent collection graph changed outside lifecycle locks"
                    )
                _terminalize_runs(
                    _runs_for_artifacts(
                        locked_collection_runs,
                        collection_artifact_ids,
                    ),
                    reason=reason,
                    using=alias,
                )
                _delete_artifacts(collection_artifacts, using=alias)
                _terminalize_runs(
                    _runs_for_artifacts(
                        locked_document_runs,
                        document_artifact_ids,
                    ),
                    reason=reason,
                    using=alias,
                )
                _delete_artifacts(document_artifacts, using=alias)
                TextChunk.objects.using(alias).filter(
                    doc_id=document.document_id
                ).delete()
                result = DocumentGraphCleanupResult(
                    affected_collection_ids=final_affected,
                    current_collection_id=row.collection_id,
                    source_hash=row.full_text_hash,
                    ingestion_complete=bool(row.ingestion_complete),
                    has_active_document_artifact=any(
                        artifact.status == GraphArtifact.Status.ACTIVE
                        and artifact.source_hash == row.full_text_hash
                        for artifact in document_artifacts
                    ),
                )
            return result
        except _ExpandedCollectionLockSet as exc:
            if _origin_context is not None:
                raise RuntimeError("document cleanup exceeds the locked delete scope")
            lock_set = exc.collection_ids
    raise RuntimeError("document lifecycle dependencies did not stabilize")


def prepare_document_chunk_replacement(
    document: DocumentLifecycleRef,
    collection_ids: Iterable[int],
    *,
    expected_source_hash: str,
    using: str = DEFAULT_DB_ALIAS,
) -> tuple[int, ...] | None:
    """Fence graph state before chunk replacement, with an idempotent fast path."""

    result = cleanup_document_graph_state(
        document,
        collection_ids,
        reason="document_chunks_replaced",
        using=using,
        expected_source_hash=expected_source_hash,
        _skip_if_committed_chunks=True,
    )
    if isinstance(result, DocumentGraphCleanupResult):
        return result.affected_collection_ids
    return result


def inspect_document_chunk_state(
    document: DocumentLifecycleRef,
    collection_ids: Iterable[int],
    *,
    expected_source_hash: str,
    using: str = DEFAULT_DB_ALIAS,
) -> str:
    """Probe exact committed chunks under lifecycle locks without provider work."""

    expected_hash = _source_hash(expected_source_hash, "expected source hash")
    alias = _database_alias(using)
    from apps.documents.models import TextChunk

    with locked_document_lifecycle_row(
        document,
        collection_ids,
        using=alias,
        active_or_building_only=True,
    ) as (row, _affected):
        if row.full_text_hash != expected_hash:
            return DocumentChunkState.STALE
        if (
            row.ingestion_complete
            and TextChunk.objects.using(alias)
            .filter(doc_id=document.document_id)
            .exists()
        ):
            return DocumentChunkState.COMMITTED
        return DocumentChunkState.NEEDS_REPLACEMENT


def cleanup_document_collection_graph_state(
    document: DocumentLifecycleRef,
    collection_ids: Iterable[int],
    *,
    reason: str,
    expected_source_hash: str,
    using: str = DEFAULT_DB_ALIAS,
) -> DocumentGraphCleanupResult | tuple[()]:
    """Invalidate collection graphs for a move while preserving document state."""

    if type(document) is not DocumentLifecycleRef:
        raise ValueError("document lifecycle reference must be exact")
    alias = _database_alias(using)
    explicit_collection_ids = _collection_ids(collection_ids)
    expected_hash = _source_hash(expected_source_hash, "expected source hash")
    from apps.knowledge_graph.models import GraphArtifact

    lock_set = _initial_document_lock_set(
        document,
        explicit_collection_ids,
        using=alias,
        active_or_building_only=True,
    )
    for _attempt in range(_MAX_LOCK_SET_RETRIES):
        try:
            with transaction.atomic(using=alias):
                collection_locks, document_locks, row = _lock_document_graph_phases(
                    document,
                    lock_set,
                    using=alias,
                )
                if row is None:
                    return ()
                locked_collection_artifacts, locked_collection_runs = _load_graph_rows(
                    collection_locks,
                    using=alias,
                )
                locked_document_artifacts, _locked_document_runs = _load_graph_rows(
                    document_locks,
                    using=alias,
                )
                final_affected = _locked_document_scope_set(
                    document,
                    row,
                    explicit_collection_ids,
                    using=alias,
                    active_or_building_only=True,
                )
                _raise_if_lock_set_expanded(lock_set, final_affected)
                if row.full_text_hash != expected_hash:
                    return ()
                artifacts = tuple(
                    artifact
                    for artifact in locked_collection_artifacts
                    if artifact.scope_type == GraphArtifact.ScopeType.COLLECTION
                    and artifact.collection_scope_id in final_affected
                    and artifact.status in _ACTIVE_OR_BUILDING
                )
                artifact_ids = tuple(artifact.pk for artifact in artifacts)
                _terminalize_runs(
                    _runs_for_artifacts(locked_collection_runs, artifact_ids),
                    reason=reason,
                    using=alias,
                )
                _delete_artifacts(artifacts, using=alias)
                has_active_document_artifact = any(
                    artifact.scope_type == GraphArtifact.ScopeType.DOCUMENT
                    and artifact.scope_id == str(document.document_id)
                    and artifact.source_hash == row.full_text_hash
                    and artifact.status == GraphArtifact.Status.ACTIVE
                    for artifact in locked_document_artifacts
                )
                result = DocumentGraphCleanupResult(
                    affected_collection_ids=final_affected,
                    current_collection_id=row.collection_id,
                    source_hash=row.full_text_hash,
                    ingestion_complete=bool(row.ingestion_complete),
                    has_active_document_artifact=has_active_document_artifact,
                )
            return result
        except _ExpandedCollectionLockSet as exc:
            lock_set = exc.collection_ids
    raise RuntimeError("document move dependencies did not stabilize")


def _enqueue_refreshes(collection_ids: tuple[int, ...]) -> None:
    from apps.knowledge_graph.services.builds import (
        enqueue_current_collection_refresh,
    )

    for collection_id in collection_ids:
        try:
            enqueue_current_collection_refresh(collection_id)
        except Exception as exc:
            logger.error(
                "obs.kg.collection_refresh_enqueue_failed",
                collection_id=collection_id,
                error_type=type(exc).__name__,
            )


def schedule_collection_graph_refreshes(
    collection_ids: Iterable[int],
    *,
    using: str = DEFAULT_DB_ALIAS,
) -> None:
    affected = _collection_ids(collection_ids)
    alias = _database_alias(using)
    if not affected:
        return
    transaction.on_commit(
        lambda: _enqueue_refreshes(affected),
        using=alias,
        robust=True,
    )


def schedule_post_chunk_graph_build(
    document_id: uuid.UUID,
    expected_source_hash: str,
    *,
    using: str = DEFAULT_DB_ALIAS,
) -> None:
    """Publish document extraction only after replacement chunks commit."""

    if type(document_id) is not uuid.UUID or document_id.version is None:
        raise ValueError("document id must be an exact RFC 4122 UUID")
    source_hash = _source_hash(expected_source_hash, "expected source hash")
    alias = _database_alias(using)

    def publish() -> None:
        from apps.knowledge_graph.services.builds import enqueue_document_build

        try:
            enqueue_document_build(document_id, source_hash)
        except Exception as exc:
            logger.error(
                "obs.kg.document_enqueue_failed",
                document_id=str(document_id),
                expected_source_hash=source_hash,
                error_type=type(exc).__name__,
            )

    transaction.on_commit(publish, using=alias, robust=True)


def schedule_document_content_invalidation(
    event: DocumentLifecycleEvent,
    *,
    using: str = DEFAULT_DB_ALIAS,
    after_cleanup=None,
) -> None:
    if type(event) is not DocumentLifecycleEvent:
        raise ValueError("document lifecycle event must be exact")
    alias = _database_alias(using)
    requested = _collection_ids(
        (event.old_collection_id, event.committed_collection_id)
    )

    def cleanup() -> None:
        try:
            result = cleanup_document_graph_state(
                event.document,
                requested,
                reason="document_content_changed",
                using=alias,
                expected_source_hash=event.committed_source_hash,
            )
        except Exception as exc:
            logger.error(
                "obs.kg.document_invalidation_failed",
                document_id=str(event.document.document_id),
                concrete_model=event.document.concrete_model_label,
                document_pkid=event.document.document_pkid,
                reason="document_content_changed",
                error_type=type(exc).__name__,
            )
            return
        if not isinstance(result, DocumentGraphCleanupResult):
            return
        _enqueue_refreshes(
            tuple(
                collection_id
                for collection_id in result.affected_collection_ids
                if collection_id != result.current_collection_id
            )
        )
        if after_cleanup is not None:
            after_cleanup()

    transaction.on_commit(cleanup, using=alias, robust=True)


def schedule_document_move_invalidation(
    event: DocumentLifecycleEvent,
    *,
    using: str = DEFAULT_DB_ALIAS,
) -> None:
    if type(event) is not DocumentLifecycleEvent:
        raise ValueError("document lifecycle event must be exact")
    alias = _database_alias(using)
    requested = _collection_ids(
        (event.old_collection_id, event.committed_collection_id)
    )

    def cleanup() -> None:
        try:
            result = cleanup_document_collection_graph_state(
                event.document,
                requested,
                reason="document_moved",
                using=alias,
                expected_source_hash=event.committed_source_hash,
            )
        except Exception as exc:
            logger.error(
                "obs.kg.collection_invalidation_failed",
                document_id=str(event.document.document_id),
                concrete_model=event.document.concrete_model_label,
                document_pkid=event.document.document_pkid,
                reason="document_moved",
                error_type=type(exc).__name__,
            )
            return
        if not isinstance(result, DocumentGraphCleanupResult):
            return
        current_collection_id = (
            result.current_collection_id
            if not (
                result.ingestion_complete
                and result.has_active_document_artifact
            )
            else None
        )
        _enqueue_refreshes(
            tuple(
                collection_id
                for collection_id in result.affected_collection_ids
                if collection_id != current_collection_id
            )
        )

    transaction.on_commit(cleanup, using=alias, robust=True)


def document_pre_save(
    sender,
    instance,
    using=DEFAULT_DB_ALIAS,
    raw=False,
    **_kwargs,
) -> None:
    alias = _database_alias(using)
    setattr(instance, _DOCUMENT_PENDING_EVENT, None)
    if raw:
        setattr(instance, _DOCUMENT_SAVE_SNAPSHOT, None)
        return
    if getattr(instance, "pkid", None) is None:
        setattr(instance, _DOCUMENT_SAVE_SNAPSHOT, None)
        return
    previous = (
        sender._base_manager.using(alias)
        .filter(pkid=instance.pkid)
        .values("id", "full_text_hash", "collection_id")
        .first()
    )
    if previous is not None and previous["id"] != instance.id:
        raise ValidationError({"id": "A persisted document UUID is immutable."})
    setattr(instance, _DOCUMENT_SAVE_SNAPSHOT, previous)


def document_post_save(
    sender,
    instance,
    created=False,
    using=DEFAULT_DB_ALIAS,
    raw=False,
    **_kwargs,
) -> None:
    if raw:
        return
    previous = getattr(instance, _DOCUMENT_SAVE_SNAPSHOT, None)
    if created or previous is None:
        return
    current = (
        sender._base_manager.using(using)
        .filter(pkid=instance.pkid, id=instance.id)
        .values("id", "full_text_hash", "collection_id")
        .first()
    )
    if current is None:
        raise ValidationError("The exact saved document row could not be reloaded.")
    event = DocumentLifecycleEvent(
        document=_document_ref(sender, instance),
        old_source_hash=previous["full_text_hash"],
        committed_source_hash=current["full_text_hash"],
        old_collection_id=previous["collection_id"],
        committed_collection_id=current["collection_id"],
    )
    if event.old_source_hash != event.committed_source_hash:
        setattr(instance, _DOCUMENT_PENDING_EVENT, ("content", event, using))
    elif event.old_collection_id != event.committed_collection_id:
        setattr(instance, _DOCUMENT_PENDING_EVENT, ("move", event, using))


def consume_document_save_lifecycle(instance) -> tuple[str, DocumentLifecycleEvent, str] | None:
    pending = getattr(instance, _DOCUMENT_PENDING_EVENT, None)
    setattr(instance, _DOCUMENT_PENDING_EVENT, None)
    return pending


def document_pre_delete(
    sender,
    instance,
    using=DEFAULT_DB_ALIAS,
    origin=None,
    **_kwargs,
) -> None:
    alias = _database_alias(using)
    context = _ensure_origin_delete_scope_locked(origin, using=alias)
    delete_snapshot = _delete_document_snapshot_from_row(sender, instance)
    try:
        _assert_document_delete_scope(context, delete_snapshot)
    except RuntimeError as exc:
        if str(exc) == "document is not part of the locked delete scope":
            raise RuntimeError("document changed during lifecycle deletion") from exc
        raise
    document = delete_snapshot.document
    # This is deliberately synchronous inside the source deletion transaction:
    # DO_NOTHING graph ownership FKs make cleanup failure abort deletion.
    guards = {}
    if delete_snapshot.figure_parent_identity is not None:
        guards["expected_parent_identity"] = (
            delete_snapshot.figure_parent_identity
        )
    result = cleanup_document_graph_state(
        document,
        (instance.collection_id,),
        reason="document_deleted",
        using=alias,
        expected_source_hash=instance.full_text_hash,
        expected_collection_id=instance.collection_id,
        fail_on_stale=True,
        _origin_context=context,
        **guards,
    )
    if not isinstance(result, DocumentGraphCleanupResult):
        raise RuntimeError("document deletion cleanup did not produce a locked result")
    setattr(
        instance,
        _DOCUMENT_DELETE_SNAPSHOT,
        (document, result.affected_collection_ids, alias),
    )


def document_post_delete(sender, instance, using=DEFAULT_DB_ALIAS, **_kwargs) -> None:
    snapshot = getattr(instance, _DOCUMENT_DELETE_SNAPSHOT, None)
    if snapshot is None:
        return
    _document, collection_ids, alias = snapshot
    transaction.on_commit(
        lambda: _enqueue_refreshes(collection_ids),
        using=alias,
        robust=True,
    )


def collection_pre_delete(
    sender,
    instance,
    using=DEFAULT_DB_ALIAS,
    origin=None,
    **_kwargs,
) -> None:
    context = _ensure_origin_delete_scope_locked(origin, using=using)
    try:
        _assert_collection_delete_scope(context, (instance.pk, instance.parent_id))
    except RuntimeError as exc:
        if str(exc) == "collection is not part of the locked delete scope":
            raise RuntimeError("collection changed during lifecycle deletion") from exc
        raise
    cleanup_collection_graph_state(
        (instance.pk,),
        reason="collection_deleted",
        using=using,
        all_artifacts=True,
        expected_parent_id=instance.parent_id,
        _origin_context=context,
    )
    __import__("apps.knowledge_graph.projection.lifecycle", fromlist=["tombstone_collection_projections_locked"]).tombstone_collection_projections_locked(collection_id=instance.pk, now=timezone.now(), using=using)  # noqa: E501

def collection_post_delete(sender, instance, **_kwargs) -> None:
    """Stable signal endpoint; deleted collections intentionally are not refreshed."""


def register_document_lifecycle_signals(document_models: Iterable[type]) -> None:
    receivers = (
        (pre_save, document_pre_save, "pre_save"),
        (post_save, document_post_save, "post_save"),
        (pre_delete, document_pre_delete, "pre_delete"),
        (post_delete, document_post_delete, "post_delete"),
    )
    for model in tuple(document_models):
        label = model._meta.label_lower
        for signal, receiver, phase in receivers:
            signal.connect(
                receiver,
                sender=model,
                weak=False,
                dispatch_uid=f"apps_knowledge_graph.{phase}.{label}",
            )


def register_collection_lifecycle_signals(collection_model: type) -> None:
    label = collection_model._meta.label_lower
    pre_delete.connect(
        collection_pre_delete,
        sender=collection_model,
        weak=False,
        dispatch_uid=f"apps_knowledge_graph.pre_delete.{label}",
    )
    post_delete.connect(
        collection_post_delete,
        sender=collection_model,
        weak=False,
        dispatch_uid=f"apps_knowledge_graph.post_delete.{label}",
    )


__all__ = [
    "DocumentLifecycleEvent",
    "DocumentLifecycleRef",
    "DocumentGraphCleanupResult",
    "DocumentChunkState",
    "cleanup_collection_graph_state",
    "cleanup_document_collection_graph_state",
    "cleanup_document_graph_state",
    "collection_post_delete",
    "collection_pre_delete",
    "document_post_delete",
    "document_post_save",
    "document_pre_delete",
    "document_pre_save",
    "consume_document_save_lifecycle",
    "load_current_document_state",
    "load_current_document_lifecycle_state",
    "locked_document_lifecycle_row",
    "inspect_document_chunk_state",
    "prepare_document_chunk_replacement",
    "register_collection_lifecycle_signals",
    "register_document_lifecycle_signals",
    "schedule_document_content_invalidation",
    "schedule_document_move_invalidation",
    "schedule_collection_graph_refreshes",
    "schedule_post_chunk_graph_build",
]
