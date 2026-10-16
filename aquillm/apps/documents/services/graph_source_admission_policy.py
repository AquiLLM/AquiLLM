"""Private graph-on body admission; the original TurnBudget alone charges reads."""

POLICY = "native-v2-channel-admission-v1"
CHANNELS = ("dense", "exact", "trigram", "full_text")


def allocation(remaining, graph_cap, active_channels):
    if any(type(v) is not int for v in (remaining, graph_cap, active_channels)) or not (
        0 <= remaining <= 45 and 0 <= graph_cap <= 20 and 0 <= active_channels <= 4
    ):
        raise ValueError("invalid_source_admission_allocation")
    graph = min(graph_cap, max(0, remaining - active_channels))
    return remaining - graph, graph


def fair_prefix(channels, limit, *, protected=()):
    """Cycle independent channels; duplicates/exhausted lanes spend no slots."""
    if type(limit) is not int or not 0 <= limit <= 45:
        raise ValueError("invalid_source_admission_prefix")
    ranks = {name: iter(dict.fromkeys(channels.get(name, ()))) for name in CHANNELS}
    seen, selected = set(), []
    allowed = {pk for name in CHANNELS for pk in channels.get(name, ())}
    for pk in protected:
        if pk in allowed and pk not in seen and len(selected) < limit:
            seen.add(pk)
            selected.append(pk)
    while len(selected) < limit:
        progressed = False
        for name in CHANNELS:
            for pk in ranks[name]:
                if pk not in seen:
                    seen.add(pk)
                    selected.append(pk)
                    progressed = True
                    break
            if len(selected) >= limit:
                break
        if not progressed:
            break
    return tuple(selected)


def charged_sources(budget):
    """Read the same atomic ledger; never reserve, refund, or replace identities."""
    from lib.retrieval.turn_budget import TurnBudget

    if type(budget) is not TurnBudget:
        raise TypeError("original_source_ledger_required")
    with budget._lock:
        return frozenset((v.chunk_id, v.source_fingerprint) for v in budget._sources)
