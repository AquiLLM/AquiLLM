"""Activation-v2 actual invariants; frozen scenario oracles remain diagnostics."""


def exhaustion(row):
    events = row.get("events", [])
    starts = [e for e in events if e["event"] == "ledger_start"]
    result = {"pairs": [], "deadline": []}
    if len(starts) != 1:
        return result
    limits = starts[0]["limits"]
    for index, event in enumerate(events):
        if event["event"] != "ledger":
            continue
        before, after = event["before"], event["after"]
        phase = event.get("keywords", {}).get("phase")
        if (
            event["operation"] == "reserve_pairs"
            and event["result"] is False
            and phase == "acquisition"
            and before["closed"] is None
            and before["pairs"][phase] >= limits["acquisition_pairs"]
            and limits["acquisition_pairs"] + limits["final_pairs"] <= 135
        ):
            result["pairs"].append(index)
        if after["closed"] == "deadline" and after["remaining_ms"] == 0:
            result["deadline"].append(index)
    return result


def actual_safety(row):
    events = row.get("events", [])
    starts = [e for e in events if e["event"] == "ledger_start"]
    complete = [e for e in events if e["event"] == "turn_complete"]
    required = (
        "publication_observation_complete",
        "provenance_complete",
        "dispatch_accounting_complete",
        "source_validation_complete",
    )
    if (
        len(starts) != 1
        or len(complete) != 1
        or any(row.get(key) is not True for key in required)
    ):
        return {"passed": None, "unknown": "incomplete observation", "evidence": []}
    ledger_id, limits = starts[0]["ledger_id"], starts[0]["limits"]
    failures, references = [], []
    closed = terminal = None
    sealed, synthesis_calls = None, 0
    for index, event in enumerate(events):
        if event["event"] == "ledger":
            references.append(index)
            state = event["after"]
            if event["ledger_id"] != ledger_id:
                failures.append("budget reset")
            counts = {
                "actions": state["actions"],
                "unique_sources": state["sources"],
                "planner_calls": state["planner_calls"],
                "in_flight_pairs": state["inflight"],
                **{f"{k}_pairs": v for k, v in state["pairs"].items()},
                **{f"{k}_codepoints": v for k, v in state["text"].items()},
            }
            if any(v < 0 or v > limits[k] for k, v in counts.items()):
                failures.append("configured bound exceeded")
            if (
                closed
                and event["operation"]
                in (
                    "reserve_action",
                    "reserve_planner",
                    "reserve_pairs",
                    "admit_source",
                )
                and event["result"] is True
            ):
                failures.append("new retrieval after closure")
            closed, terminal = state["closed"], state["terminal"]
        elif event["event"] == "synthesis_sealed":
            if sealed:
                failures.append("synthesis resealed")
            sealed, closed = event, "sealed"
            if event["remaining_retrieval_ms"] <= 0:
                failures.append("finalization reserve exhausted")
        elif event["event"] == "sdk_start":
            if terminal:
                failures.append("SDK dispatch after terminal closure")
            if sealed:
                synthesis_calls += 1
                if synthesis_calls > sealed["max_calls"]:
                    failures.append("synthesis dispatch bound exceeded")
            elif closed:
                failures.append("planner SDK after retrieval closure")
        elif event["event"] == "action_start" and closed:
            failures.append("action after retrieval closure")
    if row.get("late_publications", 0):
        failures.append("publication after disconnect")
    if row.get("citation_violations") or row.get("authorization_violations"):
        failures.append("authorization/citation violation")
    if sum(e["event"] == "final_selection" for e in events) > 1:
        failures.append("multiple final selectors")
    return {
        "passed": not failures,
        "failures": failures,
        "evidence": references,
        "exhaustion": exhaustion(row),
    }
