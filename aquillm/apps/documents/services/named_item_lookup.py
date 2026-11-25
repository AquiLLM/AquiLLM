"""Read offline PDF item metadata inside the current authorized source lifetime."""

from collections import Counter
from dataclasses import dataclass
import json
from math import log1p, sqrt
from uuid import UUID

from django.conf import settings
from django.db.models import JSONField, TextField
from django.db.models.fields.json import KeyTransform
from django.db.models.functions import Cast, Length

from .named_item_projection import MIN_OVERLAP, label_terms
from .named_item_reads import (
    PDF_MODEL,
    authorized_selections,
    ready_owner,
    current_revisions,
    validate_sources,
)

MAX_OWNERS = 4
MAX_PER_OWNER = 3
MAX_CANDIDATES = MAX_OWNERS * MAX_PER_OWNER
MAX_ABSTRACT_OWNERS = 2


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
    kind: str = "named"


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


def _candidate(selected, binding, pk, item, member, current, kind):
    if current is None or current[:4] != tuple(member):
        return None
    number, start, end, sha = member
    low, high = max(start, item["start"]), min(end, item["end"])
    if current[5] != end - start or high - low < MIN_OVERLAP:
        return None
    return NamedItemCandidate(
        pk, selected.document.id, selected.generation_key, number, current[4], end - start,
        sha, start, end, item["start"], item["label_end"], item["end"], low, high, binding, kind,
    )


def _named_candidates(scope, selected, terms, per_owner, pending):
    ready = ready_owner(scope, selected)
    if ready is None:
        return ()
    header, rows, size, binding = ready
    if not scope.runtime.budget.reserve_text(size, kind="materialized"):
        return ()
    payload = header.values_list("payload", flat=True).first()
    if payload is None:
        return ()
    data = json.loads(payload)
    members = {pk: (number, lo, hi, sha) for pk, number, lo, hi, sha in data["members"]}
    best = _choices(data, members, terms, per_owner)
    item = data.get("abstract") if data.get("version") == 2 else None
    if item is not None and selected.document.id in pending:
        pk, = item["refs"]
        pending[selected.document.id] = (ready, item, members[pk])
    if not best or not scope.runtime.budget.reserve_text(96 * len(best), kind="materialized"):
        return ()
    revisions = current_revisions(rows, best)
    hits = []
    for pk, (item, _, _) in best.items():
        hit = _candidate(selected, binding, pk, item, members[pk], revisions.get(pk), "named")
        if hit is None:
            return ()
        hits.append(hit)
    return hits


def _abstract_candidate(scope, selected, prepared=None):
    if prepared is None:
        ready = ready_owner(scope, selected)
        if ready is None:
            return None
        header, rows, _, binding = ready
        # Only fetch this short integer/coordinate record. Large unrelated list
        # postings and member arrays remain in PostgreSQL.
        data = Cast("payload", JSONField())
        header = header.annotate(index_data=data, abstract_record=KeyTransform("abstract", data)).filter(
            index_data__version=2
        )
        size = header.annotate(abstract_size=Length(Cast("abstract_record", TextField()))).values_list(
            "abstract_size", flat=True
        ).first()
        if not size or size > 192 or not scope.runtime.budget.reserve_text(size, kind="materialized"):
            return None
        item = header.values_list("abstract_record", flat=True).first()
        if item is None:
            return None
        member = None
    else:
        (header, rows, _, binding), item, member = prepared
    pk, = item["refs"]
    if not scope.runtime.budget.reserve_text(96, kind="materialized"):
        return None
    current = current_revisions(rows, (pk,)).get(pk)
    if current is None:
        return None
    if member is None:
        member = current[:4]
        if not header.filter(index_data__contains={"members": [[pk, *member]]}).exists():
            return None
    return _candidate(selected, binding, pk, item, member, current, "abstract")


def lookup_named_item_candidates(
    scope, owner_ids, question, *, per_owner=3, limit=12, abstract_owners=()
):
    if not settings.RAG_NAMED_ITEM_INDEX_ENABLED:
        return ()
    if (
        type(owner_ids) not in (tuple, list)
        or len(owner_ids) > MAX_OWNERS
        or any(type(o) is not UUID for o in owner_ids)
        or len(set(owner_ids)) != len(owner_ids)
        or type(abstract_owners) not in (tuple, list)
        or len(abstract_owners) > MAX_ABSTRACT_OWNERS
        or any(type(owner) is not UUID for owner in abstract_owners)
        or len(set(abstract_owners)) != len(abstract_owners)
        or type(per_owner) is not int
        or not 1 <= per_owner <= MAX_PER_OWNER
        or type(limit) is not int
        or not 0 <= limit <= MAX_CANDIDATES
        or type(question) is not str
        or len(question) > 2048
    ):
        return ()
    if not getattr(settings, "RAG_PARENT_ABSTRACT_RETRIEVAL_ENABLED", False):
        abstract_owners = ()
    owners = tuple(dict.fromkeys((*owner_ids, *abstract_owners)))
    if not owners or not limit:
        return ()
    selected = authorized_selections(scope, owners)
    pdfs = {
        owner: selected[owner]
        for owner in owners
        if selected[owner].document.model_label == PDF_MODEL
    }
    terms = label_terms(question)
    if not pdfs or not (terms or abstract_owners):
        return ()
    with scope.database():
        validate_sources(scope, pdfs)
        named_owners = [owner for owner in owner_ids if owner in pdfs and terms]
        abstracts = [owner for owner in abstract_owners if owner in pdfs]
        pending = dict.fromkeys(abstracts)
        groups = [_named_candidates(scope, pdfs[owner], terms, min(per_owner, limit), pending)
                  for owner in named_owners]
        ranked = [group[i] for i in range(per_owner) for group in groups if i < len(group)][:limit]
        selected_ids = {n.chunk_id for n in ranked}
        for owner in abstracts:
            if len(ranked) >= limit:
                break
            prepared = pending[owner]
            if owner in named_owners and (prepared is None or prepared[1]["refs"][0] in selected_ids):
                continue
            candidate = _abstract_candidate(scope, pdfs[owner], prepared)
            if candidate is not None:
                ranked.append(candidate)
        validate_sources(scope, pdfs)
    return tuple(ranked)


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
    abstract = [n for n in candidates if n.kind == "abstract"]
    named = [n for n in candidates if n.kind == "named"]
    if (len(named) + len(abstract) != len(candidates)
            or len({n.document_id for n in named}) > MAX_OWNERS
            or len(abstract) > MAX_ABSTRACT_OWNERS
            or (abstract and not getattr(settings, "RAG_PARENT_ABSTRACT_RETRIEVAL_ENABLED", False))
            or len({n.chunk_id for n in candidates}) != len(candidates)):
        return False
    selections = authorized_selections(scope, owners)
    with scope.database():
        validate_sources(scope, selections)
        for owner in owners:
            group = [n for n in candidates if n.document_id == owner]
            selected = selections[owner]
            if sum(n.kind == "named" for n in group) > MAX_PER_OWNER or sum(
                n.kind == "abstract" for n in group
            ) > 1 or any(
                n.generation_id != selected.generation_key for n in group
            ):
                return False
            ready = ready_owner(scope, selected)
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
                item = dict(refs=[n.chunk_id], start=n.label_start, label_end=n.label_end, end=n.item_end)
                witness = {"version": 2, "abstract": item} if n.kind == "abstract" else {"items": [item]}
                check = check.filter(
                    index_data__contains={
                        **witness,
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
            current = current_revisions(rows, [n.chunk_id for n in group])
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
        validate_sources(scope, selections)
    return True
