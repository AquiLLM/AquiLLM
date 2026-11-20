"""Native preparation obligations, independent of scoped source provenance."""

from dataclasses import dataclass
from hashlib import sha256

from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.source_loading import current_source_runtime


def pending_initial(records, identities):
    """An accepted envelope must exactly match its one pending native result."""
    if not records:
        return None
    if len(records) != 1 or records[0][0] != identities:
        raise SourceScopeUnavailable("recovery_result_changed")
    initial = records[0][1]
    if (type(initial) is not tuple or len(set(initial)) != len(initial)
            or any(value not in identities for value in initial)):
        raise SourceScopeUnavailable("recovery_result_changed")
    return initial


def record_native_preparation(captured, rows, initial_ids):
    from .rag_recovery_origins import _record_native

    _record_native(captured, rows, initial_ids, preparation=True)


def accepted_sources(origins, pool, question, authorization):
    """Shared accepted-input validation; current source authority stays fresh."""
    from .rag_recovery_origins import _identity, _payload_rows

    origins.check()
    if (authorization is not origins.authorization
            or origins.query != sha256(question.encode()).hexdigest()):
        raise SourceScopeUnavailable("recovery_original_scope_unavailable")
    sources = {}
    for occurrence in origins.occurrences:
        if _payload_rows(occurrence.result) != occurrence.rows:
            raise SourceScopeUnavailable("recovery_result_changed")
        for source in occurrence.sources:
            identity = _identity(source)
            if source.chunk_id in sources and sources[source.chunk_id] != identity:
                raise SourceScopeUnavailable("recovery_source_changed")
            sources[source.chunk_id] = identity
    ids = tuple(row.get("chunk_id", row.get("i")) for row in pool.rows)
    for row, pk in zip(pool.rows, ids):
        if pk not in sources or not any(row == original for occurrence in origins.occurrences
                                        for original in occurrence.rows):
            raise SourceScopeUnavailable("recovery_result_changed")
    origins.check()
    return ids, sources


def ordinary_votes(origins, ids, pool):
    """Keep accepted scoped occurrence votes separate from work priority."""
    scores, first = {}, {}
    for occurrence in origins.occurrences:
        for rank, identity in enumerate(occurrence.ordinary, 1):
            pk = identity[0]
            first.setdefault(pk, len(first))
            scores[pk] = scores.get(pk, 0) + 1 / (60 + rank)
    ordinary = tuple(pk for pk in sorted(scores, key=lambda pk: (-scores[pk], first[pk])) if pk in ids)
    optional = tuple(pk for pk in ids if pk not in scores)
    citations = {row.get("chunk_id", row.get("i")): row.get("citation", row.get("ref")) for row in pool.rows}
    return ordinary, optional, tuple((citations[pk], scores[pk]) for pk in ordinary)


@dataclass(frozen=True, slots=True)
class PreparationOrder:
    origins: object
    initial_ids: tuple
    deferred_ids: tuple
    identities: tuple
    fused_scores: tuple
    fallback_ids: tuple
    fallback_scores: tuple
    scoped_only_ids: tuple

    def check(self):
        self.origins.check()


def take_preparation_order(pool, question, authorization):
    """Consume actual completion obligations without changing ordinary rank."""
    from .rag_recovery_origins import _Origins

    runtime = current_source_runtime()
    if runtime is None:
        return None
    with runtime.lock:
        origins = getattr(runtime, "_recovery_origins", None)
        if type(origins) is not _Origins or not any(
                occurrence.preparation is not None for occurrence in origins.occurrences):
            return None  # Leave the established scoped consumer its own record.
        runtime._recovery_origins = None
        ids, sources = accepted_sources(origins, pool, question, authorization)
        initial = {value[0] for occurrence in origins.occurrences
                   for value in (occurrence.preparation if occurrence.preparation is not None
                                 else occurrence.ordinary)}
        fallback, scoped_only, fallback_scores = ordinary_votes(origins, ids, pool)
        if not scoped_only:
            fallback, fallback_scores = ids, pool.fused_scores
        protected = tuple(pk for pk in fallback if pk in initial)
        deferred = tuple(pk for pk in ids if pk not in initial)
        if not protected or not deferred:
            return None
        return PreparationOrder(origins, protected, deferred,
                                tuple(sources[pk] for pk in ids), pool.fused_scores,
                                fallback, fallback_scores, scoped_only)
