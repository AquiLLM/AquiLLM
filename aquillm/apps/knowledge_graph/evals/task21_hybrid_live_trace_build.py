"""Build privacy-safe symbolic trace rows for the live Task21 executor."""

from __future__ import annotations

import json
from hashlib import sha256


def candidate_maps(trace):
    from apps.knowledge_graph.retrieval.branch_contracts import BranchStatusV1

    result: dict[str, dict[int, tuple[int, float]]] = {
        "direct": {},
        "extended": {},
    }
    for name in result:
        envelope = getattr(trace.runtime, name)
        if envelope is None or envelope.status is not BranchStatusV1.SUCCEEDED:
            continue
        for row in envelope.result.candidates:
            identifier = trace.materializer.by_key.get(row.chunk_key)
            if identifier is not None:
                result[name][identifier] = (row.rank, row.score)
    return result


def projected_symbols(trace, symbols: dict[int, str]):
    maps = candidate_maps(trace)
    ordered: list[str] = []
    for name in ("direct", "extended"):
        for identifier, _rank_score in sorted(
            maps[name].items(), key=lambda row: row[1][0]
        ):
            symbol = symbols[identifier]
            if symbol not in ordered:
                ordered.append(symbol)
    return tuple(ordered), maps


def comparison_signature(prepared) -> str:
    payload = {
        "authorization": prepared.authorization.authorization_context_signature,
        "baseline": [
            prepared.chunk_symbols_by_pk[row.pk]
            for row in prepared.snapshot.baseline_candidates
        ],
        "case": prepared.case["id"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def mapped_seed_symbols(trace, prepared) -> tuple[str, ...]:
    from apps.knowledge_graph.projection.identifiers import ProjectionIdentifierDomain
    from apps.knowledge_graph.retrieval.production_extended import (
        _projection_repository,
    )

    if trace.runtime.shared is None:
        return ()
    runtime = trace.runtime.delegate
    scope = trace.runtime.shared.scope
    authority_by_document = {
        document: authority
        for authority in scope.projections
        for document, _artifact in authority.documents
    }
    candidates = {row.pk: row for row in prepared.snapshot.baseline_candidates}
    requested = {}
    for seed in prepared.snapshot.graph_seeds:
        candidate = candidates.get(seed.chunk_id)
        authority = authority_by_document.get(getattr(candidate, "doc_id", None))
        if authority is None:
            continue
        key = runtime.codec.encode(
            ProjectionIdentifierDomain.CHUNK,
            generation=authority.generation_id,
            source=seed.chunk_id,
        ).value
        requested[key] = prepared.chunk_symbols_by_pk[seed.chunk_id]
    repository = _projection_repository(runtime)
    mentioned = {
        row.chunk_key
        for authority in scope.projections
        for row in repository.load_projection_bundle(
            projection_id=authority.projection_id,
            batch_size=runtime.settings.projection_batch_size,
            purpose="audit",
        ).entity_mentions
    }
    return tuple(
        requested[key] for key in requested if key in mentioned
    )


__all__ = [
    "candidate_maps",
    "comparison_signature",
    "mapped_seed_symbols",
    "projected_symbols",
]
