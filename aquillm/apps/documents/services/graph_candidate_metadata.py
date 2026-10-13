"""Body-free native rank capture inside fresh original source boundaries."""

from dataclasses import dataclass
from json import dumps
from time import perf_counter
from uuid import UUID

from django.contrib.postgres.search import TrigramSimilarity
from django.db.models import Case, IntegerField, When
from django.db.models.functions import Length, MD5

from .chunk_search_rank_policy import controls
from .lexical_candidates import extract_exact_terms, ranked_exact_queryset
from .lexical_readiness import FullTextCoverage
from .representation_contracts import SourceScopeUnavailable
from .source_loading import _prepare_source_queryset


@dataclass(frozen=True, slots=True)
class SourceRankRef:
    pk: int
    doc_id: object
    chunk_number: int
    generation: object
    revision: str
    length: int
    fts_binding: object = None
    exact_coverage: int = 0

    def __post_init__(self):
        if (type(self.pk) is not int or self.pk <= 0 or type(self.doc_id) is not UUID
                or type(self.chunk_number) is not int or self.chunk_number < 0
                or (self.generation is not None and type(self.generation) is not UUID)
                or type(self.length) is not int or self.length <= 0
                or type(self.revision) is not str or len(self.revision) != 32
                or any(c not in "0123456789abcdef" for c in self.revision)):
            raise SourceScopeUnavailable("source_admission_invalid_metadata")

    @property
    def identity(self):
        return self.pk, self.revision

    @property
    def fingerprint(self):
        return self.doc_id, self.chunk_number, self.revision, self.generation


def rank_refs(queryset, *, exact=False, full_text=False):
    """Scalar values only: no deferred model can access uncharged content."""
    names = ["pk", "doc_id", "chunk_number", "representation_generation",
             "_admission_revision", "_admission_length"]
    if full_text:
        names.append("search_vector_binding")
    if exact:
        names.append("_exact_coverage")
    records = queryset.annotate(
        _admission_revision=MD5("content"), _admission_length=Length("content")
    ).values_list(*names)[:45]
    result = []
    for row in records:
        if row[5] <= 0:
            continue
        ref = SourceRankRef(
            *row[:6], fts_binding=dumps(row[6], sort_keys=True, separators=(",", ":")) if full_text else None,
            exact_coverage=row[6] if exact else 0,
        )
        result.append(ref)
    return tuple(result)


def captured_query(model, refs, scope):
    """Conjunction per captured tuple; independent IN sets cannot cross owners."""
    from django.db.models import Q

    predicate = Q(pk__in=[])
    for ref in refs:
        predicate |= Q(
            pk=ref.pk, doc_id=ref.doc_id, chunk_number=ref.chunk_number,
            representation_generation=ref.generation,
            _admission_revision=ref.revision, _admission_length=ref.length,
        )
    order = Case(*(When(pk=r.pk, then=i) for i, r in enumerate(refs)),
                 output_field=IntegerField())
    return model.objects.using(scope.using).annotate(
        _admission_revision=MD5("content"), _admission_length=Length("content")
    ).filter(predicate, doc_id__in=scope.requested_document_ids).order_by(order)


def _full_text_metadata(model, query, scope, limit):
    # Share the public readiness proof, only replacing its private eager reader.
    from .lexical_readiness import _full_text_candidates

    return _full_text_candidates(
        model, query, scope, limit=limit,
        reader=lambda rows, *_args, **_kw: rank_refs(rows, full_text=True),
    )


def capture_metadata(model, query, scope, limits, observation, check):
    from .embedding_index_query import rank_compatible_chunks

    original = scope.runtime._frozen_retrieval_scope
    channels, dense, timings = {}, {}, {}
    check()
    initial_keys = set(scope.validate())
    started = perf_counter()
    vectors = ()
    vector_error = None
    if observation is not None and scope.dense is not None:
        try:
            ranked = rank_compatible_chunks(
                snapshot=scope.representation, dense_snapshot=scope.dense,
                query=observation, limit=min(45, limits.vector), using=scope.using,
                budget=scope.runtime.budget,
                requested_document_ids=scope.requested_document_ids,
            )
            ranked = tuple(r for r in ranked if r.embedding_generation_key in initial_keys)
            query_rows = _prepare_source_queryset(model.objects.using(scope.using).filter(
                pk__in=[r.chunk_id for r in ranked],
                doc_id__in=scope.requested_document_ids,
            ), original)
            found = {r.pk: r for r in rank_refs(query_rows)}
            vectors = tuple(found[r.chunk_id] for r in ranked if r.chunk_id in found
                            and found[r.chunk_id].revision == r.source_md5
                            and found[r.chunk_id].generation == r.representation_key)
            dense = {r.chunk_id: r for r in ranked if r.chunk_id in {v.pk for v in vectors}}
        except SourceScopeUnavailable:
            raise
        except Exception:
            vector_error = "upstream_unavailable"
    channels["dense"] = vectors
    timings["vector"] = (perf_counter() - started) * 1000
    check()
    text = _prepare_source_queryset(model.objects.using(scope.using).filter(
        doc_id__in=scope.requested_document_ids, modality=model.Modality.TEXT,
    ), original)
    full_text, coverage = (), FullTextCoverage("disabled")
    trigram_limit = limits.trigram
    if controls()["full_text"]:
        try:
            full_text, coverage = _full_text_metadata(model, query, scope, limits.trigram // 2)
        except Exception:
            full_text, coverage = (), FullTextCoverage("unavailable")
        check()
        if coverage.status == "ready":
            trigram_limit -= limits.trigram // 2
    channels["full_text"] = full_text
    started = perf_counter()
    channels["trigram"] = rank_refs(text.annotate(
        similarity=TrigramSimilarity("content", query)
    ).filter(similarity__gt=limits.trigram_similarity_min).order_by(
        "-similarity", "pk"
    )[:trigram_limit])
    timings["trigram"] = (perf_counter() - started) * 1000
    check()
    started = perf_counter()
    terms = extract_exact_terms(query)
    channels["exact"] = rank_refs(ranked_exact_queryset(
        text, terms, limit=limits.exact
    ), exact=True) if terms else ()
    timings["exact"] = (perf_counter() - started) * 1000
    check()
    final_keys = initial_keys & set(scope.validate())
    channels["dense"] = tuple(r for r in vectors
                              if dense[r.pk].embedding_generation_key in final_keys)
    dense = {pk: r for pk, r in dense.items() if r.embedding_generation_key in final_keys}
    check()
    # A single PK cannot silently adopt a later revision from another channel.
    captured = {}
    for refs in channels.values():
        for ref in refs:
            if ref.pk in captured and captured[ref.pk].fingerprint != ref.fingerprint:
                raise SourceScopeUnavailable("source_scope_unavailable")
            captured[ref.pk] = ref
    return channels, dense, terms, timings, coverage, vector_error
