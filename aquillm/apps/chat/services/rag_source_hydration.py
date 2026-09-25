"""Hydrate current sources separately from public clipped search previews."""

from copy import copy
from types import MappingProxyType, SimpleNamespace

from apps.chat.services.rag_source_preparation import prepare_evidence, render_evidence
from apps.collections.services.retrieval_authorization import (
    revalidate_retrieval_authorization_context,
)
from apps.documents.models.chunks import TextChunk
from apps.documents.services.source_loading import (
    SourcePreparationLimited,
    bounded_source_database,
    current_source_runtime,
    source_query_rows,
)
from lib.llm.evidence_guard import estimate_request_tokens
from lib.retrieval.evidence import SourceEvidence, fingerprint_source


def hydrate_source_rows(
    rows,
    authorization,
    *,
    question,
    token_ceiling,
    turn_budget,
    source_windows=None,
    chunk_loader=None,
):
    from apps.chat.services.rag_retrieval import _verified_row_coordinates
    from apps.chat.services.rag_selection_hydration import HydratedRow

    if turn_budget is None:
        raise SourcePreparationLimited("source preparation requires shared turn ledger")
    scope = _current_scope(authorization)
    ids = tuple(row.get("chunk_id", row.get("i")) for row in rows)
    if chunk_loader is None:
        chunks = source_query_rows(
            TextChunk.objects.using(authorization.database_alias).filter(pk__in=ids)
        )
    else:
        chunks = chunk_loader(authorization, ids)
    current = {chunk.pk: chunk for chunk in chunks}
    hydrated = []
    runtime = current_source_runtime()
    windows = (
        source_windows
        if source_windows is not None
        else (runtime.windows if runtime else {})
    )
    for row in rows:
        verified = _verified_row_coordinates(dict(row))
        if verified is None:
            continue
        pk, doc, number, _ = verified
        chunk = current.get(pk)
        if (
            chunk is None
            or chunk.doc_id not in scope.document_ids
            or str(chunk.doc_id) != doc
            or chunk.chunk_number != number
        ):
            continue
        source = SourceEvidence(
            pk, doc, number, fingerprint_source(chunk.content), chunk.content
        )
        prepared = prepare_evidence(
            source,
            question=question,
            windows=windows.get(pk, ()),
            token_ceiling=token_ceiling,
            turn_budget=turn_budget,
        )
        if not prepared.spans:
            continue
        rendered = render_evidence(prepared)
        frozen_chunk = copy(chunk)
        frozen_chunk.content = rendered
        # Unchanged full text retains acquisition's exact score/cache identity.
        if rendered != source.text:
            frozen_chunk.source_revision = source.source_fingerprint
        frozen_row = dict(row)
        frozen_row["x" if "x" in row and "text" not in row else "text"] = rendered
        hydrated.append(
            HydratedRow(
                MappingProxyType(frozen_row),
                frozen_chunk,
                rendered,
                source.source_fingerprint,
                prepared,
            )
        )
    return tuple(hydrated)


def source_candidate_cost(item, turn_budget=None):
    # Count exact rendered text, row/citation metadata and any figure together.
    return estimate_request_tokens(dict(item.row), turn_budget=turn_budget)


def revalidate_prepared_rows(rows, authorization, *, chunk_loader=None):
    identities = tuple(
        SimpleNamespace(
            chunk_id=r.chunk.pk,
            doc_id=str(r.chunk.doc_id),
            chunk_number=r.chunk.chunk_number,
            source_fingerprint=r.source_fingerprint,
        )
        for r in rows
    )
    retained = revalidate_source_candidates(
        identities, authorization, chunk_loader=chunk_loader
    )
    ids = {item.chunk_id for item in retained}
    return tuple(row for row in rows if row.chunk.pk in ids)


def revalidate_source_candidates(candidates, authorization, *, chunk_loader=None):
    scope = _current_scope(authorization)
    ids = tuple(c.chunk_id for c in candidates)
    if chunk_loader is None:
        chunks = source_query_rows(
            TextChunk.objects.using(authorization.database_alias).filter(pk__in=ids)
        )
    else:
        chunks = chunk_loader(authorization, ids)
    current = {chunk.pk: chunk for chunk in chunks}
    return tuple(
        candidate
        for candidate in candidates
        if (chunk := current.get(candidate.chunk_id)) is not None
        and chunk.doc_id in scope.document_ids
        and str(chunk.doc_id) == candidate.doc_id
        and chunk.chunk_number == candidate.chunk_number
        and fingerprint_source(chunk.content) == candidate.source_fingerprint
    )


def _current_scope(authorization):
    runtime = current_source_runtime()
    if runtime is None:
        return revalidate_retrieval_authorization_context(context=authorization)
    with bounded_source_database(runtime, authorization.database_alias):
        return revalidate_retrieval_authorization_context(context=authorization)
