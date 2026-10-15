"""Fresh bounded scalar reads for one complete representation validation."""
from django.db.models import CharField, F, Value

from ._uuid_membership import uuid_membership
from .representation_contracts import SourceScopeUnavailable
from .representation_tuple_predicate import ExactTupleMembership


def _unavailable():
    raise SourceScopeUnavailable("source_scope_unavailable")


def validate_document_rows(selections, *, using):
    from apps.documents.models import DESCENDED_FROM_DOCUMENT
    from .representation_store import SourceSHA256

    expected = {s.document.id: s.document for s in selections}
    if len(expected) != len(selections):
        _unavailable()
    if not expected:
        return
    models = tuple(DESCENDED_FROM_DOCUMENT)
    membership = uuid_membership(
        ("id",), tuple((value,) for value in expected), using=using,
        fields=tuple(model._meta.get_field("id") for model in models),
    )
    # One current SQL statement still visits every allowlisted concrete type.
    # UNION ALL preserves duplicate UUIDs; a global sentinel bounds hydration.
    owners = []
    for model in models:
        owners.append(model._base_manager.using(using).filter(membership).annotate(
            _canonical=SourceSHA256(F("full_text")),
            _owner_model=Value(model._meta.label_lower, output_field=CharField()),
        ).order_by().values(
            "pkid", "id", "collection_id", "full_text_hash", "_canonical", "_owner_model",
        ))
    rows = owners[0].union(*owners[1:], all=True)[:len(expected) + 1]
    seen = set()
    for row in rows:
        ref = expected.get(row["id"])
        if ref is None or row["id"] in seen or len(seen) >= len(expected):
            _unavailable()
        if (row["_owner_model"], row["pkid"], row["id"], row["collection_id"],
            row["full_text_hash"], row["_canonical"]) != (
            ref.model_label, ref.pkid, ref.id, ref.collection_id,
            ref.source_hash, ref.source_hash,
        ):
            _unavailable()
        seen.add(row["id"])
    if seen != set(expected):
        _unavailable()


def validate_state_rows(selections, *, using):
    from apps.documents.models import DocumentRepresentationState

    expected = {(s.document.model_label, s.document.pkid, s.document.id): s
                for s in selections}
    if len(expected) != len(selections):
        _unavailable()
    if not expected:
        return
    condition = ExactTupleMembership(
        ("model_label", "document_pkid", "document_id"), tuple(expected))
    rows = DocumentRepresentationState.objects.using(using).filter(condition).order_by("pk").values(
        "pk", "model_label", "document_pkid", "document_id", "collection_id",
        "legacy_closed", "legacy_source_hash", "active_generation_id",
        "active_generation__owner_id", "active_generation__status",
        "active_generation__collection_id", "active_generation__canonical_sha256",
        "active_generation__source_revision_sha256",
    )[:len(expected) + 1]
    seen = set()
    for row in rows:
        key = row["model_label"], row["document_pkid"], row["document_id"]
        selected = expected.get(key)
        if selected is None or key in seen or len(seen) >= len(expected):
            _unavailable()
        seen.add(key)
        ref = selected.document
        if selected.generation_key is None:
            if (row["legacy_closed"] or row["active_generation_id"]
                or row["legacy_source_hash"] != ref.source_hash):
                _unavailable()
        elif not (
            row["active_generation_id"] == selected.generation_key
            and row["collection_id"] == ref.collection_id and row["legacy_closed"]
            and row["active_generation__owner_id"] == row["pk"]
            and row["active_generation__status"] == "active"
            and row["active_generation__collection_id"] == ref.collection_id
            and row["active_generation__canonical_sha256"] == ref.source_hash
            and row["active_generation__source_revision_sha256"] == selected.source_revision_sha256
        ):
            _unavailable()
    if any(key not in seen and s.generation_key is not None for key, s in expected.items()):
        _unavailable()
