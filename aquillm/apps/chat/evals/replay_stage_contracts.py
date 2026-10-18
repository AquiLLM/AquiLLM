"""Redacted schedule and final novelty contracts for private replay stages."""


def _label(value, allowed):
    return value if type(value) is str and value in allowed else None


def schedule_fields(stage):
    result = {
        key: _label(stage.get(key), allowed)
        for key, allowed in (
            ("acquisition_schedule_requested", ("scored", "native_closure")),
            ("acquisition_schedule_observed", ("scored", "native_closure")),
            ("acquisition_initial_ineligibility_reason", ("native_ineligible",)),
            ("acquisition_score_status", ("not_requested", "complete", "unavailable")),
        )
    }
    count = stage.get("preclosure_pool_count")
    result["preclosure_pool_count"] = (
        count if type(count) is int and count >= 0 else None
    )
    return result


def schedule_summary(stages):
    fields = [schedule_fields(stage) for stage in stages]
    observed = [stage["acquisition_schedule_observed"] for stage in fields]
    requested = [stage["acquisition_schedule_requested"] for stage in fields]
    available = bool(stages) and all(value is not None for value in observed)
    return {
        "status": "observed" if available else "unavailable",
        "observed": observed if available else None,
        "matches_requested": (
            observed == requested
            if available and all(value is not None for value in requested)
            else None
        ),
    }


def materialized_pool(stage):
    rows = stage.get("materialized_union")
    if (
        stage.get("acquisition_schedule_observed") != "native_closure"
        and stage.get("materialized_union_status") in (None, "observed")
        and type(rows) in (list, tuple)
    ):
        return rows
    return None


def reranked_pool(stage):
    # Native closure publishes an unscored handoff in the historical post_rerank
    # field. It is still useful for source provenance, never for rerank metrics.
    if (
        stage.get("acquisition_schedule_observed") == "native_closure"
        or stage.get("acquisition_score_status") in ("unavailable", "not_requested")
    ):
        return None
    rows = stage.get("post_rerank")
    return rows if type(rows) in (list, tuple) else None


def _valid_ids(values):
    return type(values) in (list, tuple) and all(
        type(pk) is int and 0 < pk <= 9_223_372_036_854_775_807 for pk in values
    )


def novel_final_provenance(stages, packet, graph_sources, identity):
    baseline, finalized = set(), set()
    baseline_available = bool(stages)
    final_available = bool(stages)
    for stage in stages:
        values = stage.get("baseline_metadata_ids")
        if stage.get("baseline_metadata_status") == "captured" and _valid_ids(values):
            baseline.update(values)
        else:
            baseline_available = False
        rows = stage.get("post_rerank")
        if type(rows) in (list, tuple) and all(type(row) is dict for row in rows):
            ids = [identity(row) for row in rows]
            if _valid_ids(ids):
                finalized.update(ids)
            else:
                final_available = False
        else:
            final_available = False
    graph_off = bool(stages) and all(
        stage.get("graph_overlay_enabled") is False for stage in stages
    )
    available = (
        baseline_available
        and final_available
        and (graph_sources is not None or graph_off)
    )
    novel = (graph_sources or set()) & finalized - baseline
    return {
        "status": "captured" if available else "unavailable",
        "baseline_metadata_status": (
            "captured" if baseline_available else "unavailable"
        ),
        "baseline_metadata_count": len(baseline) if baseline_available else None,
        "packet_count": (
            sum(type(identity(row)) is int and identity(row) in novel for row in packet)
            if available
            else None
        ),
    }
