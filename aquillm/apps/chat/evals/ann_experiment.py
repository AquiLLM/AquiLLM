"""Unactivated physical-ranking experiment; production remains the exact arm."""

from django.db.models import F
from django.db.models.functions import MD5
from pgvector.django import L2Distance

from apps.documents.models import TextChunk
from apps.documents.models.embedding_index import ChunkEmbedding
from apps.documents.services.embedding_index_query import (
    rank_compatible_chunks,
    validate_dense_snapshot,
)
from apps.documents.services.embedding_index_query_scope import (
    _authorized_runtime,
    _dense_runtime,
    requested_dense_scope,
)
from apps.documents.services.embedding_query_payloads import bounded_dense_payloads
from apps.documents.services.embedding_index_store import load_embedding_space
from apps.documents.services.representation_store import active_chunk_scope
from apps.documents.services.source_loading import (
    SourcePreparationLimited,
    current_source_runtime,
)
from lib.embeddings.document_binding import validate_query_observation
from lib.embeddings.document_contracts import digest

from .ann_candidate_validation import validate_candidate

ARMS = ("production_exact", "experimental_direct_distance")


def _original_runtime(scope):
    runtime = current_source_runtime()
    if runtime is None:
        from apps.documents.services.representation_manual_scope import private_acquisition_runtime
        runtime = private_acquisition_runtime()
    original = getattr(runtime, "_frozen_retrieval_scope", None)
    if (
        runtime is None
        or scope.runtime is not runtime
        or original is None
        or scope.authorization is not runtime.authorization
        or scope.representation is not original.representation
        or scope.dense is not original.dense
    ):
        raise ValueError("experiment requires the retained original source runtime")
    if _dense_runtime(scope.representation, runtime.budget, scope.using) is not runtime:
        raise ValueError("experiment requires the retained original source runtime")
    return runtime


def experimental_candidates(scope, query):
    """Same eligible predicates as serving; direct distance ordering is experimental."""
    runtime = _original_runtime(scope)
    _authorized_runtime(scope.representation, runtime.budget, scope.using, runtime=runtime)
    keys = validate_dense_snapshot(scope.representation, scope.dense, using=scope.using)
    requested, keys = requested_dense_scope(
        scope.representation, scope.dense, keys, scope.requested_document_ids
    )
    if (
        query.compatibility_key is None
        or not keys
        or query.compatibility_key != scope.dense.space_key
    ):
        return None, None
    space = load_embedding_space(scope.dense.space_key, using=scope.using)
    if validate_query_observation(query, space=space) != query.compatibility_key:
        return None, None
    eligible = active_chunk_scope(
        TextChunk.objects.using(scope.using).filter(doc_id__in=requested),
        snapshot=scope.representation,
    ).values("pk")
    rows = (
        ChunkEmbedding.objects.using(scope.using)
        .filter(
            generation_id__in=keys,
            chunk_id__in=eligible,
        )
        .annotate(
            current_md5=MD5("chunk__content"),
            distance=L2Distance("vector", list(query.vector)),
        )
        .filter(source_md5=F("current_md5"))
    )
    return rows.order_by("distance"), space


def rank_experiment(scope, query, *, limit, arm):
    """Return eligible rows without collecting timing or plan evidence."""
    if arm not in ARMS or type(limit) is not int or not 0 < limit <= 45:
        raise ValueError("invalid experimental arm or candidate limit")
    runtime = _original_runtime(scope)
    signature = digest(
        {
            "experiment": "r5-physical-ranking-v1",
            "arm": arm,
            "query": query.prepared_input_sha256,
            "requested": sorted(map(str, scope.requested_document_ids)),
        }
    )
    if not runtime.budget.reserve_action(signature):
        raise SourcePreparationLimited("experimental action allowance")
    if arm == "production_exact":
        return rank_compatible_chunks(
            snapshot=scope.representation,
            dense_snapshot=scope.dense,
            query=query,
            limit=limit,
            using=scope.using,
            budget=runtime.budget,
            requested_document_ids=scope.requested_document_ids,
        )
    with scope.database():
        rows, space = experimental_candidates(scope, query)
        if rows is None:
            return ()
        representations = {
            s.generation_key: s.representation_key for s in scope.dense.selections
        }
        ranked = []
        # Same pre-read producer as serving, retaining this SQL's eligibility.
        # Numeric vector width is schema bounded, never serialized as text.
        for candidate in bounded_dense_payloads(rows[:limit], budget=runtime.budget):
            row = validate_candidate(
                candidate, space=space, representation_keys=representations
            )
            if row is not None:
                ranked.append(row)
        valid = set(
            validate_dense_snapshot(
                scope.representation, scope.dense, using=scope.using
            )
        )
        _authorized_runtime(scope.representation, runtime.budget, scope.using, runtime=runtime)
    if not runtime.budget.can_publish():
        raise SourcePreparationLimited("source deadline")
    return tuple(row for row in ranked if row.embedding_generation_key in valid)
