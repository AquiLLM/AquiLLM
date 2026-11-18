"""Read offline PDF item metadata inside the current authorized source lifetime."""

from collections import Counter
from dataclasses import dataclass
import json
from math import log1p, sqrt
from uuid import UUID

from django.conf import settings
from django.db import connections
from django.db.models import F, JSONField
from django.db.models.functions import Cast, Length, MD5

from .named_item_projection import (
    MAX_CANONICAL,
    MAX_MEMBERS,
    MAX_PAYLOAD,
    MIN_OVERLAP,
    label_terms,
    owner_binding,
)
from .representation_contracts import RepresentationSnapshotV1, SourceScopeUnavailable
from .representation_store import (
    SourceSHA256,
    _active_chunk_predicates,
    validate_representation_snapshot,
)

MAX_OWNERS = 4
MAX_PER_OWNER = 3
MAX_CANDIDATES = MAX_OWNERS * MAX_PER_OWNER
PDF_MODEL = "apps_documents.pdfdocument"


@dataclass(frozen=True, slots=True)
class NamedItemCandidate:
    chunk_id: int
    document_id: UUID
    generation_id: UUID | None
    chunk_number: int
    body_md5: str
    body_length: int
    body_sha256: str
    chunk_start: int
    chunk_end: int
    label_start: int
    label_end: int
    item_end: int
    covered_start: int
    covered_end: int
    binding: str


def _authorized_selections(scope, owner_ids):
    from .chunk_search_scope import FrozenRetrievalScope, current_scope
    from .graph_source_runtime import graph_source_runtime
    from .hybrid_graph_authorization import is_exact_authorization_context

    if (
        type(scope) is not FrozenRetrievalScope
        or scope is not current_scope()
        or scope.runtime is None
        or scope.runtime is not graph_source_runtime()
        or not is_exact_authorization_context(scope.authorization)
        or scope.runtime.authorization is not scope.authorization
        or scope.using != scope.authorization.database_alias
        or not set(owner_ids) <= scope.requested_document_ids
    ):
        raise SourceScopeUnavailable("named_item_scope_unavailable")
    selected = {
        s.document.id: s
        for s in scope.representation.selections
        if s.document.id in owner_ids
    }
    if len(selected) != len(owner_ids):
        raise SourceScopeUnavailable("named_item_scope_unavailable")
    return selected


def _validate_sources(scope, selected):
    """Reauthorize the original selection and check only the requested owners' source bytes."""
    from .chunk_search_scope import validate_requested_authorization

    if not scope.runtime.budget.can_publish():
        raise SourceScopeUnavailable("named_item_scope_unavailable")
    if scope.proposal_binding is not None:
        from apps.chat.services.retrieval_query_runtime import (
            validate_bound_before_current_scope,
        )

        validate_bound_before_current_scope(scope.proposal_binding, scope)
    validate_requested_authorization(scope.runtime, scope.authorization, ())
    validate_representation_snapshot(
        RepresentationSnapshotV1(tuple(selected.values())), using=scope.using
    )
    if not scope.runtime.budget.can_publish():
        raise SourceScopeUnavailable("named_item_scope_unavailable")


def _legacy_membership(rows):
    """A single bounded digest of ordered integers. No body or body hash is read."""
    columns = rows.order_by("pk").values_list(
        "pk", "chunk_number", "start_position", "end_position"
    )[: MAX_MEMBERS + 1]
    sql, params = columns.query.get_compiler(using=rows.db).as_sql()
    with connections[rows.db].cursor() as cursor:
        cursor.execute(
            "SELECT encode(sha256(convert_to(string_agg(concat_ws(':', id, n, lo, hi), ',' ORDER BY id),"
            " 'UTF8')), 'hex') FROM (" + sql + ") AS members(id,n,lo,hi)",
            params,
        )
        return cursor.fetchone()[0]


def _ready_owner(scope, selected):
    from apps.documents.models import PDFNamedItemIndex, TextChunk

    if selected.document.model_label != PDF_MODEL:
        return None
    header = (
        PDFNamedItemIndex.objects.using(scope.using)
        .filter(
            document_id=selected.document.pkid,
            generation_id=selected.generation_key,
        )
        .annotate(size=Length("payload"), payload_sha=SourceSHA256(F("payload")))
        .filter(size__lte=MAX_PAYLOAD)
    )
    # Header strings and the optional legacy scalar have fixed maximum lengths.
    if not scope.runtime.budget.reserve_text(256, kind="materialized"):
        return None
    value = header.values_list(
        "binding", "member_digest", "payload_sha", "size"
    ).first()
    if value is None:
        return None
    binding, member_sha, payload_sha, size = value
    if owner_binding(selected, member_sha, payload_sha) != binding:
        return None
    rows = _active_chunk_predicates(
        TextChunk.objects.using(scope.using).filter(
            doc_id=selected.document.id, modality=TextChunk.Modality.TEXT
        ),
        snapshot=RepresentationSnapshotV1((selected,)),
    )
    if selected.generation_key is None and _legacy_membership(rows) != member_sha:
        return None
    return (
        header.filter(
            binding=binding, member_digest=member_sha, payload_sha=payload_sha
        ),
        rows,
        size,
        binding,
    )


def _choices(data, members, query_terms, limit):
    # Only this owner's literal labels influence its ordering, even while other
    # owners await a backfill. A rare matching word carries more weight.
    items = data["items"]
    matches = Counter()
    for term in sorted(query_terms):
        posting = data["postings"].get(term, ())
        if posting:
            weight = log1p((len(items) - len(posting) + 0.5) / (len(posting) + 0.5))
            for index in posting:
                matches[index] += weight
    rank = []
    for index, weight in matches.items():
        item = items[index]
        score = weight / sqrt(item["term_count"])
        for pk in item["refs"]:
            _, start, end, _ = members[pk]
            low, high = max(start, item["start"]), min(end, item["end"])
            key = (
                -score,
                low - high,
                not start <= item["start"] < end,
                pk,
                item["start"],
            )
            rank.append((key, pk, item, low, high))
    best = {}
    for _, pk, item, low, high in sorted(rank, key=lambda row: row[0]):
        best.setdefault(pk, (item, low, high))
        if len(best) == limit:
            break
    return best


def _revisions(rows, ids):
    # Body hashes are computed only for the at most three shortlisted chunks.
    data = (
        rows.filter(pk__in=ids)
        .annotate(
            body_sha=SourceSHA256(F("content")),
            native_md5=MD5("content"),
            body_size=Length("content"),
        )
        .filter(
            body_size__lte=MAX_CANONICAL,
            body_size=F("end_position") - F("start_position"),
        )
        .values_list(
            "pk",
            "chunk_number",
            "start_position",
            "end_position",
            "body_sha",
            "native_md5",
            "body_size",
        )
    )
    return {
        pk: (number, start, end, sha, md5, size)
        for pk, number, start, end, sha, md5, size in data
    }


def _nominate(scope, selected, terms, per_owner):
    ready = _ready_owner(scope, selected)
    if ready is None:
        return ()
    header, rows, size, binding = ready
    if not scope.runtime.budget.reserve_text(size, kind="materialized"):
        return ()
    payload = header.values_list("payload", flat=True).first()
    if payload is None:
        return ()
    data = json.loads(payload)
    members = {
        pk: (number, start, end, sha) for pk, number, start, end, sha in data["members"]
    }
    best = _choices(data, members, terms, per_owner)
    if not best or not scope.runtime.budget.reserve_text(
        96 * len(best), kind="materialized"
    ):
        return ()
    revisions = _revisions(rows, best)
    candidates = []
    for pk, (item, low, high) in best.items():
        current = revisions.get(pk)
        number, start, end, sha = members[pk]
        if (
            current is None
            or current[:4] != members[pk]
            or current[5] != end - start
            or high - low < MIN_OVERLAP
        ):
            return ()
        candidates.append(
            NamedItemCandidate(
                pk,
                selected.document.id,
                selected.generation_key,
                number,
                current[4],
                end - start,
                sha,
                start,
                end,
                item["start"],
                item["label_end"],
                item["end"],
                low,
                high,
                binding,
            )
        )
    return candidates


def lookup_named_item_candidates(scope, owner_ids, question, *, per_owner=3, limit=12):
    if not settings.RAG_NAMED_ITEM_INDEX_ENABLED:
        return ()
    if (
        type(owner_ids) not in (tuple, list)
        or not 0 < len(owner_ids) <= MAX_OWNERS
        or any(type(o) is not UUID for o in owner_ids)
        or len(set(owner_ids)) != len(owner_ids)
        or type(per_owner) is not int
        or not 1 <= per_owner <= MAX_PER_OWNER
        or type(limit) is not int
        or not 0 <= limit <= MAX_CANDIDATES
        or type(question) is not str
        or len(question) > 2048
    ):
        return ()
    selected = _authorized_selections(scope, owner_ids)
    pdfs = {
        owner: selected[owner]
        for owner in owner_ids
        if selected[owner].document.model_label == PDF_MODEL
    }
    terms = label_terms(question)
    if not pdfs or not terms or not limit:
        return ()
    with scope.database():
        _validate_sources(scope, pdfs)
        groups = [
            _nominate(scope, pdf, terms, min(per_owner, limit)) for pdf in pdfs.values()
        ]
        _validate_sources(scope, pdfs)
    return tuple(
        group[n] for n in range(per_owner) for group in groups if n < len(group)
    )[:limit]


def validate_named_item_candidates(scope, candidates):
    """Recheck just the captured receipts before publication, without source text."""
    if type(candidates) not in (list, tuple) or len(candidates) > MAX_CANDIDATES:
        return False
    if not candidates:
        return True
    if not settings.RAG_NAMED_ITEM_INDEX_ENABLED or any(
        type(n) is not NamedItemCandidate for n in candidates
    ):
        return False
    owners = dict.fromkeys(n.document_id for n in candidates)
    if len(owners) > MAX_OWNERS or len({n.chunk_id for n in candidates}) != len(
        candidates
    ):
        return False
    selections = _authorized_selections(scope, owners)
    with scope.database():
        _validate_sources(scope, selections)
        for owner in owners:
            group = [n for n in candidates if n.document_id == owner]
            selected = selections[owner]
            if len(group) > MAX_PER_OWNER or any(
                n.generation_id != selected.generation_key for n in group
            ):
                return False
            ready = _ready_owner(scope, selected)
            if ready is None or not scope.runtime.budget.reserve_text(
                96 * len(group), kind="materialized"
            ):
                return False
            header, rows, _, binding = ready
            check = header.annotate(index_data=Cast("payload", JSONField()))
            for n in group:
                if (
                    n.binding != binding
                    or (n.covered_start, n.covered_end)
                    != (max(n.chunk_start, n.label_start), min(n.chunk_end, n.item_end))
                    or n.covered_end - n.covered_start < MIN_OVERLAP
                ):
                    return False
                check = check.filter(
                    index_data__contains={
                        "items": [
                            {
                                "refs": [n.chunk_id],
                                "start": n.label_start,
                                "label_end": n.label_end,
                                "end": n.item_end,
                            }
                        ],
                        "members": [
                            [
                                n.chunk_id,
                                n.chunk_number,
                                n.chunk_start,
                                n.chunk_end,
                                n.body_sha256,
                            ]
                        ],
                    }
                )
            if not check.exists():
                return False
            current = _revisions(rows, [n.chunk_id for n in group])
            if any(
                current.get(n.chunk_id)
                != (
                    n.chunk_number,
                    n.chunk_start,
                    n.chunk_end,
                    n.body_sha256,
                    n.body_md5,
                    n.body_length,
                )
                for n in group
            ):
                return False
        _validate_sources(scope, selections)
    return True
