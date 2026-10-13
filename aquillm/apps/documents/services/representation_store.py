"""Exact legacy-null representation scopes. Source hashes never grant access."""

from uuid import UUID
import re

from django.contrib.postgres.expressions import ArraySubquery
from django.db.models import BooleanField, F, Func, Q

from lib.embeddings.document_contracts import digest, require_digest
from .representation_contracts import (
    DocumentRefV1,
    RepresentationSelectionV1,
    RepresentationSnapshotV1,
    SourceScopeUnavailable,
)


class SourceSHA256(Func):
    template = "encode(sha256(convert_to(%(expressions)s, 'UTF8')), 'hex')"


class OwnerIdMembership(Func):
    """Parameter-safe UUID membership in one statement-local owner array."""

    def __init__(self, owners):
        super().__init__(F("doc_id"), ArraySubquery(owners), output_field=BooleanField())

    def as_sql(self, compiler, connection, **extra_context):
        lhs, lhs_params = compiler.compile(self.source_expressions[0])
        rhs, rhs_params = compiler.compile(self.source_expressions[1])
        return f"({lhs} = ANY({rhs}))", [*lhs_params, *rhs_params]


def _unavailable():
    raise SourceScopeUnavailable("source_scope_unavailable")


def document_payload(ref):
    if (
        type(ref) is not DocumentRefV1
        or type(ref.model_label) is not str
        or re.fullmatch(r"[a-z0-9_]+\.[a-z0-9_]+", ref.model_label) is None
        or type(ref.id) is not UUID
        or type(ref.pkid) is not int
        or ref.pkid <= 0
        or type(ref.collection_id) is not int
        or ref.collection_id <= 0
    ):
        _unavailable()
    try:
        require_digest(ref.source_hash)
    except ValueError:
        _unavailable()
    return {"model_label": ref.model_label, "pkid": ref.pkid, "id": str(ref.id),
            "collection_id": ref.collection_id, "source_hash": ref.source_hash}


def ordered_selections(snapshot):
    if (
        type(snapshot) is not RepresentationSnapshotV1
        or type(snapshot.selections) is not tuple
    ):
        _unavailable()
    seen = set()
    for s in snapshot.selections:
        if type(s) is not RepresentationSelectionV1:
            _unavailable()
        document_payload(s.document)
        if s.document.id in seen:
            _unavailable()
        seen.add(s.document.id)
        if s.generation_key is None:
            if s.source_revision_sha256 is not None:
                _unavailable()
        else:
            if type(s.generation_key) is not UUID:
                _unavailable()
            try:
                require_digest(s.source_revision_sha256)
            except ValueError:
                _unavailable()
        if s.canonical_sha256 != s.document.source_hash:
            _unavailable()
    return tuple(
        sorted(
            snapshot.selections,
            key=lambda s: (s.document.model_label, s.document.pkid, str(s.document.id)),
        )
    )


def representation_snapshot_digest(snapshot: RepresentationSnapshotV1) -> str:
    return digest(
        {
            "version": "representation-snapshot-v1",
            "selections": [
                {
                    "document": document_payload(s.document),
                    "generation_key": str(s.generation_key) if s.generation_key else None,
                    "canonical_sha256": s.canonical_sha256,
                    "source_revision_sha256": s.source_revision_sha256,
                }
                for s in ordered_selections(snapshot)
            ],
        }
    )


def validate_document_reference(ref, *, using):
    document_payload(ref)
    from apps.documents.models import DESCENDED_FROM_DOCUMENT

    owners = []
    for model in DESCENDED_FROM_DOCUMENT:
        rows = (
            model._base_manager.using(using)
            .filter(id=ref.id)
            .annotate(_canonical=SourceSHA256(F("full_text")))
            .values("pkid", "id", "collection_id", "full_text_hash", "_canonical")[
                :2
            ]
        )
        owners.extend((model._meta.label_lower, row) for row in rows)
    if len(owners) != 1:
        _unavailable()
    label, row = owners[0]
    if (
        label,
        row["pkid"],
        row["id"],
        row["collection_id"],
        row["full_text_hash"],
        row["_canonical"],
    ) != (
        ref.model_label,
        ref.pkid,
        ref.id,
        ref.collection_id,
        ref.source_hash,
        ref.source_hash,
    ):
        _unavailable()
    return ref


def validate_representation_snapshot(snapshot: RepresentationSnapshotV1, *, using: str) -> RepresentationSnapshotV1:
    from .representation_validation_rows import validate_document_rows, validate_state_rows
    selections = ordered_selections(snapshot)
    if selections:
        validate_document_rows(selections, using=using)
        validate_state_rows(selections, using=using)
    return snapshot

def freeze_active_representations(
    documents: tuple[DocumentRefV1, ...], *, using: str
) -> RepresentationSnapshotV1:
    if type(documents) is not tuple:
        _unavailable()
    from .representation_scope import selection_for_document
    snapshot = RepresentationSnapshotV1(tuple(selection_for_document(ref, using=using) for ref in documents))
    return validate_representation_snapshot(snapshot, using=using)


def active_chunk_scope(queryset, *, snapshot, document_id=None):
    """Lazy predicates also close a UUID collision after this function returns."""
    validate_representation_snapshot(snapshot, using=queryset.db)
    return _active_chunk_predicates(queryset, snapshot=snapshot, document_id=document_id)


def _active_chunk_predicates(queryset, *, snapshot, document_id=None):
    """Build unchanged SQL fences within a caller's fresh validation boundary."""
    from apps.documents.models import DESCENDED_FROM_DOCUMENT

    selections = ordered_selections(snapshot)
    if document_id is not None:
        if type(document_id) is not UUID:
            _unavailable()
        selections = tuple(s for s in selections if s.document.id == document_id)
        if len(selections) != 1:
            _unavailable()
    from .representation_scope import current_state_clause, _current_active_state_clause
    from .representation_tuple_predicate import ExactTupleMembership
    condition = Q(pk__in=[])
    represented, active = {}, []
    for s in selections:
        ref = s.document
        clause = Q(doc_id=ref.id, representation_generation_id=s.generation_key)
        if s.generation_key is None:
            condition |= clause & current_state_clause(s, using=queryset.db)
        else:
            active.append(s)
        represented.setdefault(ref.model_label, []).append(ref)
    if not selections:
        return queryset.filter(condition)
    if active:
        active_condition = Q(ExactTupleMembership(
            ("doc_id", "representation_generation_id"),
            tuple((s.document.id, s.generation_key) for s in active)))
        condition |= active_condition & _current_active_state_clause(tuple(active), using=queryset.db)

    exact_owner = Q(pk__in=[])
    unique_owner = Q()
    selected_ids = tuple(s.document.id for s in selections)
    for model in DESCENDED_FROM_DOCUMENT:
        owners = model._base_manager.using(queryset.db).filter(id__in=selected_ids).order_by()
        refs = represented.get(model._meta.label_lower, ())
        if refs:
            exact_tuples = ExactTupleMembership(
                ("id", "pkid", "collection_id", "full_text_hash"),
                tuple((ref.id, ref.pkid, ref.collection_id, ref.source_hash) for ref in refs))
            identities = ExactTupleMembership(
                ("id", "pkid"), tuple((ref.id, ref.pkid) for ref in refs))
            # ordered_selections binds each canonical digest to its source hash;
            # the complete tuple fixes the stored hash before this single SHA check.
            exact = owners.annotate(_canonical=SourceSHA256(F("full_text"))).filter(
                exact_tuples, _canonical=F("full_text_hash"))
            # A bounded uncorrelated membership read hashes each selected owner
            # once in this SQL statement, rather than its full body per chunk.
            # ARRAY creates a statement-local InitPlan. Plain IN can become a
            # nested semi-join that rescans even an uncorrelated owner query.
            # Exact PK/UUID tuples admit at most len(refs) rows, so this limit
            # cannot truncate valid owners or reduce lexical recall.
            exact_owner |= Q(OwnerIdMembership(exact.values("id")[:len(refs)]))
            # Keep UUID/PK pairs: a different selected PK can acquire this UUID
            # after construction, and must then count as an unexpected owner.
            owners = owners.exclude(identities)
        unique_owner &= ~Q(doc_id__in=owners.values("id"))
    return queryset.filter(condition & exact_owner & unique_owner)


def load_chunk_source(chunk_id, *, using):
    from .representation_citations import load_chunk_source as load
    return load(chunk_id, using=using)
