"""Bounded retrieval memory and whole source views used only by the planner."""

from collections import deque
from dataclasses import asdict, dataclass, replace
from itertools import islice


def planner_views(evidence_views, max_chars=16000, latest_source_ids=()):
    """Round-robin documents, preserving exact source objects and stable order."""
    latest = set(latest_source_ids)
    groups, seen = {}, set()
    for source in evidence_views:
        if source.identity in seen:
            continue
        seen.add(source.identity)
        groups.setdefault(source.document_id, []).append(source)
    ordered = sorted(
        groups.values(),
        key=lambda group: not any(source.identity in latest for source in group),
    )
    queues = [
        deque(sorted(group, key=lambda source: source.identity not in latest))
        for group in ordered
    ]
    selected, remaining = [], max_chars
    while any(queues):
        for queue in queues:
            while queue:
                source = queue.popleft()
                if len(source.text) <= remaining:
                    selected.append(source)
                    remaining -= len(source.text)
                    break
                # Skip the whole view, then try a smaller view of this document.
    return tuple(selected)


@dataclass(frozen=True)
class AcquisitionAttempt:
    kind: str
    query: str | None
    document_id: str | None
    chunk_id: int | None
    outcome: str
    available_source_count: int
    new_source_count: int
    query_truncated: bool = False


def acquisition_attempt(action, before_ids, after_ids, *, outcome="completed"):
    """Counts describe available unique revisions, never an assumed result body."""
    before, after = set(before_ids), set(after_ids)
    query = action.query if action is not None else None
    return AcquisitionAttempt(
        kind=action.kind if action is not None else "external",
        query=query[:2000] if query is not None else None,
        document_id=action.document_id if action is not None else None,
        chunk_id=action.chunk_id if action is not None else None,
        outcome="no_progress"
        if outcome == "completed" and not after - before
        else outcome,
        available_source_count=len(after),
        new_source_count=len(after - before),
        query_truncated=query is not None and len(query) > 2000,
    )


def attempt_context(attempts, max_actions):
    """Only bounded server-owned fields enter the prompt; no source/error bodies."""
    records = []
    for attempt in islice(attempts, max_actions):
        query = attempt.query
        records.append(
            asdict(
                replace(
                    attempt,
                    query=query[:2000] if query is not None else None,
                    query_truncated=attempt.query_truncated
                    or (query is not None and len(query) > 2000),
                )
            )
        )
    return records
