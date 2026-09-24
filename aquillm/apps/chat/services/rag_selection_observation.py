"""Opt-in exact exclusion causes; never changes selector decisions or text."""

from lib.evidence_observation import active, publish


def observe_exclusions(remaining, selected, document_counts, tokens, limits):
    if not active():
        return
    from .rag_selection import candidate_token_cost

    for candidate in remaining:
        prepared = candidate.prepared_evidence
        if prepared is None:
            continue  # Legacy offsets cannot be invented for source-loss proof.
        if len(selected) >= limits.max_passages:
            resource, used, limit, required = (
                "passages",
                len(selected),
                limits.max_passages,
                1,
            )
        elif document_counts[candidate.doc_id] >= limits.max_per_document:
            resource, used, limit, required = (
                "document_passages",
                document_counts[candidate.doc_id],
                limits.max_per_document,
                1,
            )
        elif tokens + candidate_token_cost(candidate) > limits.token_budget:
            resource, used, limit, required = (
                "evidence_tokens",
                tokens,
                limits.token_budget,
                candidate_token_cost(candidate),
            )
        else:
            continue
        publish(
            "selection_excluded",
            {
                "phase": "final_selection",
                "resource": resource,
                "used": used,
                "limit": limit,
                "required": required,
                "source": {
                    "document_id": prepared.source.document_id,
                    "chunk_id": prepared.source.chunk_id,
                    "fingerprint": prepared.source.source_fingerprint,
                    "spans": [[s.start, s.end] for s in prepared.spans],
                },
            },
        )
