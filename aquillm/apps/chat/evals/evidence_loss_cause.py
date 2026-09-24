"""Structural proof for supported loss causes, separate from human semantics."""

from .evidence_quality_eval import identity


def observed_loss_cause(row, fact, loss, bindings):
    cause = loss.get("cause", {})
    ref, acquired, sdk = (
        cause.get("event"),
        fact["acquisition_event"],
        loss["sdk_event"],
    )
    if (
        cause.get("type") != "selection_limit"
        or any(type(i) is not int for i in (ref, acquired, sdk))
        or not 0 <= acquired < ref < sdk < len(row["events"])
        or sdk
        != max(i for i, e in enumerate(row["events"]) if e["event"] == "sdk_start")
    ):
        return False
    event = row["events"][ref]
    resource = cause.get("resource")
    if (
        event.get("event") != "selection_excluded"
        or event.get("phase") != "final_selection"
        or resource not in ("passages", "document_passages", "evidence_tokens")
        or event.get("resource") != resource
    ):
        return False
    used, limit, required = (event.get(k) for k in ("used", "limit", "required"))
    if (
        any(type(v) is not int for v in (used, limit, required))
        or min(used, limit) < 0
        or required <= 0
        or used > limit
        or used + required <= limit
        or (resource != "evidence_tokens" and required != 1)
    ):
        return False
    affected = event["source"]
    source = bindings.get((affected["document_id"], affected["chunk_id"]))
    span = fact["span"]
    return bool(
        source
        and identity(source) == identity(span)
        and affected.get("fingerprint") == span["fingerprint"]
        and any(
            0 <= start <= span["start"] < span["end"] <= end <= len(source["text"])
            for start, end in affected["spans"]
        )
    )
