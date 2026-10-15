"""Diagnostic evidence only: never changes or replaces the production ledger."""
import json
from collections import Counter
from pathlib import Path

from apps.knowledge_graph.projection.frontier_index_contracts_v2 import decode_record
from apps.knowledge_graph.retrieval.topology.contracts import ProjectedSeedV1
from apps.knowledge_graph.retrieval.topology.frontier_acquisition_v2 import acquire_frontier
from apps.knowledge_graph.retrieval.topology.frontier_contracts_v2 import FrontierBudgetExceeded
from apps.knowledge_graph.tests.frontier_scope_window_support import MixedScopeFixture


def offline_copy_schedule(fixture, seeds, cached):
    """Count existing immutable paths, not a relaxed-budget execution."""
    directories, found = [], 0
    for seed in seeds:
        for generation in sorted(fixture.session.selected):
            current = fixture.driver.markers[generation]["frontier_root_key"]
            target = "copies/" + seed.identity_key
            while current is not None:
                pair = generation, current
                if pair not in cached:
                    directories.append(pair)
                    cached.add(pair)
                kind, node = decode_record(fixture.driver.index[current],
                                          expected_key=current, generation=generation)
                assert kind == "directory"
                if node["name"] == target:
                    found += 1
                    break
                current = node["left"] if target < node["name"] else node["right"]
    # Every selected seed is a different generation's real relation source.
    # Its single copy needs an entry, fresh entity/membership pair, relation head.
    return {"directory_reads": len(directories), "found_copy_streams": found,
            "copy_entry_reads": found, "copy_raw_reads": found,
            "relation_head_reads": found,
            "pre_admission_queries": len(directories) + 3 * found,
            "pre_admission_rows": len(directories) + 4 * found}


def offline_batched_directories(fixture, seeds, cached):
    """Design estimate only: cache-first seeks; same root paths and rows."""
    pending = {(generation, "copies/" + seed.identity_key):
               fixture.driver.markers[generation]["frontier_root_key"]
               for seed in seeds for generation in fixture.session.selected}
    waves = []
    while pending:
        waiting = {}
        for (generation, target), current in pending.items():
            while current is not None and (generation, current) in cached:
                kind, node = decode_record(fixture.driver.index[current],
                                          expected_key=current, generation=generation)
                assert kind == "directory"
                if node["name"] == target:
                    current = None
                    break
                current = node["left"] if target < node["name"] else node["right"]
            if current is not None:
                waiting[generation, target] = current
        missing = sorted({(g, current) for (g, _), current in waiting.items()})
        if missing:
            batches = [len(missing[i:i + 160]) for i in range(0, len(missing), 160)]
            waves.append({"unique_missing": len(missing), "batches": batches})
            cached.update(missing)
        pending = waiting
    return {"waves": waves, "queries": sum(len(w["batches"]) for w in waves),
            "rows": sum(w["unique_missing"] for w in waves)}


def run(seed_count):
    fixture = MixedScopeFixture()
    first = sorted(fixture.bundles, key=lambda b: b.generation.generation_key)[:seed_count]
    seeds = tuple(sorted((ProjectedSeedV1(b.relations[0].source_entity_key, 1.0)
                          for b in first), key=lambda s: s.identity_key))
    fixture.session.initialize()
    initial = {"queries": fixture.budget.queries, "rows": fixture.budget.source_rows}
    offline = offline_copy_schedule(fixture, seeds, set(fixture.session.reads._index_records))
    batched = offline_batched_directories(fixture, seeds, set(fixture.session.reads._index_records))
    visited = []
    original = fixture.session.copies

    def tracked(generation, identity):
        item = {"generation": generation, "identity": identity, "completed": False}
        visited.append(item)
        yield from original(generation, identity)
        item["completed"] = True

    fixture.session.copies = tracked
    fixture.driver.phase = "acquisition"
    error = None
    try:
        acquire_frontier(fixture.session, fixture.assembly, seeds, fixture.caps, fixture.budget)
    except FrontierBudgetExceeded as exc:
        error = str(exc)
    before_final = {"queries": fixture.budget.queries, "rows": fixture.budget.source_rows}
    fixture.driver.phase = "final"
    fixture.session.final_attestation()
    summary = {
        "seed_count": seed_count, "selected_generations": len(fixture.ready.selected_generations),
        "authorized_documents": len(fixture.ready.authorized_documents),
        "unrelated_copies_per_generation": 256,
        "initial": initial, "before_final": before_final,
        "total": {"queries": fixture.budget.queries, "rows": fixture.budget.source_rows},
        "acquisition_queries": before_final["queries"] - initial["queries"],
        "acquisition_rows": before_final["rows"] - initial["rows"],
        "copy_searches_started": len(visited),
        "copy_searches_completed": sum(v["completed"] for v in visited),
        "admitted_edges": fixture.budget.edges, "candidates": len(fixture.assembly.candidates),
        "error": error, "offline_first_round_schedule": offline,
        "offline_batched_directory_design_estimate": batched,
        "kind_counts": dict(Counter(t["kind"] for t in fixture.driver.transcript)),
        "max_physical_rows": max(t["rows"] for t in fixture.driver.transcript),
        "max_physical_bound": max(t["maximum"] for t in fixture.driver.transcript),
        "final_reservation_consumed": fixture.session.reads._final_authority is None,
    }
    return {"summary": summary, "transcript": fixture.driver.transcript,
            "copy_searches": visited}


if __name__ == "__main__":
    result = {str(count): run(count) for count in (1, 16)}
    Path("/tmp/scope-feasibility-counts.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v["summary"] for k, v in result.items()}, indent=2))
