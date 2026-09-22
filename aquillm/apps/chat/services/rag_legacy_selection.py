"""Pure legacy document rotation shared by serving and offline replay."""


def diversify_evidence_chunks(
    chunks: list[dict],
    per_doc_limit: int,
) -> list[dict]:
    """Round-robin across documents so no single doc consumes all snippet slots.

    The strategy:
    1. Group chunks by ``doc_id`` preserving original ranking order within each group.
    2. Round-robin: take one chunk per doc per round until each doc hits its cap.

    This guarantees that when multiple docs are present, all get at least one
    snippet before any doc receives a second.
    """
    from collections import defaultdict

    doc_order: list[str] = []
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for chunk in chunks:
        doc_id = chunk.get("doc_id") or chunk.get("d", "")
        if doc_id not in doc_order:
            doc_order.append(doc_id)
        by_doc[doc_id].append(chunk)

    result: list[dict] = []
    doc_counts: dict[str, int] = {d: 0 for d in doc_order}
    doc_iters = {d: iter(by_doc[d]) for d in doc_order}
    exhausted: set[str] = set()

    while len(exhausted) < len(doc_order):
        for doc_id in doc_order:
            if doc_id in exhausted:
                continue
            if doc_counts[doc_id] >= per_doc_limit:
                exhausted.add(doc_id)
                continue
            try:
                chunk = next(doc_iters[doc_id])
                result.append(chunk)
                doc_counts[doc_id] += 1
            except StopIteration:
                exhausted.add(doc_id)

    return result
