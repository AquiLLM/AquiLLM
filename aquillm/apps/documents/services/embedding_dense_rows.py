"""Fresh scalar dense-state reads for exact bounded original owner tuples."""
from django.db.models import Q

from apps.documents.models.embedding_index import DocumentEmbeddingState

from .representation_contracts import SourceScopeUnavailable
from .representation_tuple_predicate import ExactTupleMembership
from .source_deadline import check_source_deadline

_BATCH_SIZE = 64
_IDENTITY = (
    "model_label", "document_pkid", "document_id", "collection_id",
    "representation_key", "space_id",
)
_FIELDS = (*_IDENTITY, "pk", "active_generation_id", "active_generation__key",
           "active_generation__owner_id", "active_generation__state",
           "active_generation__failure_kind", "active_generation__source_hash",
           "active_generation__canonical_sha256", "active_generation__manifest_sha256")


def _identity(selection, space_key):
    ref = selection.document
    return (ref.model_label, ref.pkid, ref.id, ref.collection_id,
            selection.generation_key, space_key)


def _batch_rows(identities, *, using):
    current = tuple(row for row in identities if row[4] is not None)
    legacy = tuple((*row[:4], row[5]) for row in identities if row[4] is None)
    predicate = Q(ExactTupleMembership(_IDENTITY, current))
    if legacy:
        # SQL tuple equality cannot match NULL; retain the exact other fields.
        predicate |= Q(representation_key__isnull=True) & Q(ExactTupleMembership(
            (*_IDENTITY[:4], _IDENTITY[5]), legacy))
    rows = (DocumentEmbeddingState.objects.using(using).filter(predicate)
            .order_by("pk").values(*_FIELDS)[:len(identities) + 1])
    check_source_deadline()
    result = tuple(rows)
    check_source_deadline()
    return result


def _validated_rows(rows, identities):
    if len(rows) > len(identities):
        raise SourceScopeUnavailable("source_scope_unavailable")
    expected, mapped, pks = set(identities), {}, set()
    for row in rows:
        if type(row) is not dict or row.keys() != set(_FIELDS):
            raise SourceScopeUnavailable("source_scope_unavailable")
        identity = tuple(row[name] for name in _IDENTITY)
        if identity not in expected or identity in mapped or row["pk"] in pks:
            raise SourceScopeUnavailable("source_scope_unavailable")
        mapped[identity] = row
        pks.add(row["pk"])
    return mapped


def dense_owner_rows(selections, *, space_key, using):
    """Preserve selection order; missing or invalid active generations stay absent."""
    for offset in range(0, len(selections), _BATCH_SIZE):
        batch = selections[offset:offset + _BATCH_SIZE]
        identities = tuple(_identity(selection, space_key) for selection in batch)
        if len(set(identities)) != len(identities):
            raise SourceScopeUnavailable("source_scope_unavailable")
        rows = _validated_rows(_batch_rows(identities, using=using), identities)
        for selection, identity in zip(batch, identities, strict=True):
            row = rows.get(identity)
            if (row is None or row["active_generation__key"] is None
                    or row["active_generation_id"] != row["active_generation__key"]
                    or row["active_generation__owner_id"] != row["pk"]
                    or row["active_generation__state"] != "ready"
                    or row["active_generation__failure_kind"]
                    or row["active_generation__source_hash"] != selection.document.source_hash
                    or row["active_generation__canonical_sha256"] != selection.canonical_sha256):
                continue
            yield selection, row["active_generation__key"], row["active_generation__manifest_sha256"]
