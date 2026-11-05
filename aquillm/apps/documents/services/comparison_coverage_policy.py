"""Bounded comparison policy and immutable, same-turn source coverage records."""

from dataclasses import dataclass, field
from hashlib import md5, sha256
from os import getenv

from django.core.exceptions import ImproperlyConfigured

POLICY = "version-comparison-coverage-v1"
_SEAL = object()


def comparison_coverage_enabled():
    name = "RAG_COMPARISON_COVERAGE_ENABLED"
    value = getenv(name, "0").strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise ImproperlyConfigured(f"invalid {name}")


def comparison_policy_identity():
    return POLICY + (":enabled" if comparison_coverage_enabled() else ":disabled")


def _ranking_controls():
    from .chunk_search_rank_policy import controls

    return tuple(controls().items())


@dataclass(frozen=True, slots=True)
class CoverageSource:
    chunk_id: int
    document_id: str
    chunk_number: int
    generation: object
    revision: str
    length: int
    body_sha256: str


@dataclass(frozen=True, slots=True)
class CoverageGroup:
    sources: tuple[CoverageSource, ...]
    document: object
    overlap: tuple | None


@dataclass(frozen=True, slots=True)
class CoverageRecord:
    policy: str
    query_sha256: str
    groups: tuple[CoverageGroup, ...]
    runtime: object = field(repr=False, compare=False)
    budget: object = field(repr=False, compare=False)
    authorization: object = field(repr=False, compare=False)
    representation: object = field(repr=False)
    dense_manifest: str
    ranking_controls: tuple
    seal: object = field(repr=False, compare=False)


def mixed_dense(original, left, right):
    """Keep nine original places, allocate at most four per paper, then refill."""
    if any(len(lane) > 17 or len(set(lane)) != len(lane) for lane in (original, left, right)):
        raise ValueError("invalid comparison candidate lane")
    result = list(original[:9])
    iterators = (iter(left), iter(right))
    for _ in range(4):
        for candidates in iterators:
            if len(result) >= len(original):
                break
            candidate = next((pk for pk in candidates if pk not in result), None)
            if candidate is not None:
                result.append(candidate)
    for pk in original:
        if len(result) < len(original) and pk not in result:
            result.append(pk)
    return tuple(result)


def mixed_lexical(original, heads):
    if len(heads) != 2 or len(set(heads)) != 2 or len(original) < 2:
        return None
    return tuple(dict.fromkeys((*original[:-2], *heads, *original)))[:len(original)]


def native_successor(head, refs, representation):
    """A captured coordinate and concrete owner are required even for legacy data."""
    if (representation.document.id != head.doc_id or not representation.document.source_hash
            or not representation.canonical_sha256 or representation.generation_key != head.generation):
        return None
    by_pk = {}
    for ref in refs:
        previous = by_pk.get(ref.pk)
        if previous is not None and (previous.fingerprint != ref.fingerprint or previous.length != ref.length):
            return None
        by_pk[ref.pk] = ref
    candidates = [ref for ref in by_pk.values() if ref.doc_id == head.doc_id
                  and ref.generation == head.generation and ref.chunk_number == head.chunk_number + 1]
    return candidates[0] if len(candidates) == 1 else None


def stored_overlap(head, after, head_text, after_text):
    """Prove one substantial local overlap while retaining both source identities."""
    if (head.doc_id != after.doc_id or head.generation != after.generation
            or after.chunk_number != head.chunk_number + 1 or len(head_text) != head.length
            or len(after_text) != after.length or md5(head_text.encode()).hexdigest() != head.revision
            or md5(after_text.encode()).hexdigest() != after.revision):
        return None
    sizes = [size for size in range(64, min(len(head_text), len(after_text)) // 2 + 1)
             if head_text[-size:] == after_text[:size]]
    if len(sizes) != 1:
        return None
    size = sizes[0]
    shared = after_text[:size]
    if (len(set(shared)) < 12 or len(shared.split()) < 8
            or head_text.count(shared) != 1 or after_text.count(shared) != 1):
        return None
    return (len(head_text) - size, len(head_text), 0, size, sha256(shared.encode()).hexdigest())


def capture_source(ref, row):
    from .representation_contracts import SourceScopeUnavailable

    text = row.__dict__.get("content")
    if (not isinstance(text, str) or row.pk != ref.pk or row.doc_id != ref.doc_id
            or row.chunk_number != ref.chunk_number or len(text) != ref.length
            or md5(text.encode()).hexdigest() != ref.revision):
        raise SourceScopeUnavailable("comparison_source_revision_changed")
    return CoverageSource(ref.pk, str(ref.doc_id), ref.chunk_number, ref.generation,
                          ref.revision, ref.length, sha256(text.encode()).hexdigest())


def bind_coverage(scope, query, groups):
    return CoverageRecord(policy=comparison_policy_identity(), query_sha256=sha256(query.encode()).hexdigest(),
        groups=tuple(groups), runtime=scope.runtime, budget=scope.runtime.budget,
        authorization=scope.runtime.authorization, representation=scope.representation,
        dense_manifest=scope.dense.manifest_sha256, ranking_controls=_ranking_controls(), seal=_SEAL)


def publish_coverage(record, plan):
    """Install hints only after all native preparation and source publication succeed."""
    if record is None:
        return
    runtime = record.runtime

    def install():
        plan.check()
        present = {row.pk: row for row in runtime.cache.values()}
        for group in record.groups:
            for source in group.sources:
                row = present.get(source.chunk_id)
                if row is None or source.chunk_id not in plan.initial_ids:
                    return
                ref = plan.references[source.chunk_id]
                if capture_source(ref, row) != source:
                    return
        runtime.comparison_coverage = record

    runtime.budget.publish(install)


def selection_coverage_groups(runtime, prepared, pool, query):
    """Resolve same-turn hints only against complete original-query scored evidence."""
    from .source_loading import current_source_runtime

    if runtime is None or current_source_runtime() is not runtime or not comparison_coverage_enabled():
        return ()
    record = runtime.comparison_coverage
    frozen = getattr(runtime, "_frozen_retrieval_scope", None)
    if (type(record) is not CoverageRecord or record.seal is not _SEAL or record.runtime is not runtime
            or record.budget is not runtime.budget or record.authorization is not runtime.authorization or frozen is None
            or record.representation != frozen.representation
            or record.dense_manifest != getattr(frozen.dense, "manifest_sha256", None)
            or record.policy != comparison_policy_identity()
            or record.ranking_controls != _ranking_controls()
            or record.query_sha256 != sha256(query.encode()).hexdigest()
            or not runtime.budget.can_publish() or prepared.score_status != "model"
            or prepared.fallback_reason or len(record.groups) != 2):
        return ()
    expected_ids = {row.get("chunk_id", row.get("i")) for row in pool.rows}
    candidates = {candidate.chunk_id: candidate for candidate in prepared.candidates}
    if (set(candidates) != expected_ids
            or len({candidate.scorer_fingerprint for candidate in prepared.candidates}) != 1
            or any(candidate.raw_score is None or not candidate.scorer_fingerprint
                   or candidate.scoring_kind not in ("pointwise", "listwise")
                   for candidate in prepared.candidates)):
        return ()
    # The normal final preparation already rehydrates and revalidates the pool.
    # Recheck current permission and representation before applying this policy.
    with frozen.database():
        frozen.validate()
    groups = []
    for group in record.groups:
        resolved = []
        for source in group.sources:
            candidate = candidates.get(source.chunk_id)
            if (candidate is None or candidate.doc_id != source.document_id
                    or candidate.chunk_number != source.chunk_number
                    or candidate.source_fingerprint != source.body_sha256
                    or candidate.raw_score is None or not candidate.scorer_fingerprint
                    or candidate.scoring_kind not in ("pointwise", "listwise")):
                return ()
            evidence = candidate.prepared_evidence
            if (evidence is None or evidence.source.document_id != source.document_id
                    or evidence.source.chunk_id != source.chunk_id
                    or evidence.source.chunk_number != source.chunk_number
                    or evidence.source.source_fingerprint != source.body_sha256
                    or sha256(evidence.source.text.encode()).hexdigest() != source.body_sha256
                    or tuple((span.start, span.end) for span in evidence.spans) != ((0, source.length),)):
                return ()
            resolved.append(candidate)
        groups.append(tuple(resolved))
    if (runtime.comparison_coverage is not record or current_source_runtime() is not runtime
            or runtime.budget is not record.budget
            or runtime.authorization is not record.authorization
            or getattr(runtime, "_frozen_retrieval_scope", None) is not frozen
            or frozen.representation != record.representation
            or getattr(frozen.dense, "manifest_sha256", None) != record.dense_manifest
            or record.ranking_controls != _ranking_controls()
            or record.policy != comparison_policy_identity() or not runtime.budget.can_publish()):
        return ()
    return tuple(groups)
