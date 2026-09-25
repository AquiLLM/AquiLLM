"""Deterministic physical-copy and fanout selection for projected topology."""

from __future__ import annotations

from collections import defaultdict

from apps.knowledge_graph.retrieval.ppr import RetrievalDirection, raw_edge_weight


def select_topology_groups(
    *,
    transitions,
    evidence_by_relation,
    entities,
    entities_by_identity,
    identity_by_entity,
    identity_hops,
    max_nodes: int,
    max_edges: int,
):
    candidates = []
    for key in sorted(transitions):
        target, direction = key[2], key[3]
        ranked = []
        for physical in transitions[key]:
            rows = evidence_by_relation[physical.relation_key]
            confidence = max(row.confidence for row in rows)
            weight = raw_edge_weight(
                direction=RetrievalDirection(direction.value),
                confidence=confidence,
                support_count=len(rows),
                destination_retrieval_utility=max(
                    entities[item].retrieval_utility
                    for item in entities_by_identity[target]
                ),
            )
            ranked.append((-weight, physical.relation_key, weight, physical))
        _, _, weight, physical = min(ranked)
        selected_evidence = sorted(
            evidence_by_relation[physical.relation_key],
            key=lambda row: (-row.confidence, row.evidence_key, row.chunk_key),
        )[:3]
        selected_evidence.sort(key=lambda row: (row.evidence_key, row.chunk_key))
        candidates.append((key, weight, physical, selected_evidence))
    candidates.sort(key=lambda row: (-row[1], row[0]))
    accepted, endpoint_entities = [], set()
    fanout = defaultdict(int)
    for candidate in candidates:
        key, _weight, physical, _evidence = candidate
        source = key[0]
        if fanout[source] == 10:
            continue
        trial = endpoint_entities | {
            physical.source_entity_key,
            physical.target_entity_key,
        }
        covered = {identity_by_entity[entity_key] for entity_key in trial}
        minimum_memberships = len(trial) + len(set(identity_hops) - covered)
        if minimum_memberships > max_nodes:
            continue
        accepted.append(candidate)
        endpoint_entities = trial
        fanout[source] += 1
        if len(accepted) == max_edges:
            break
    accepted.sort(key=lambda row: row[0])
    selected_entities = set(endpoint_entities)
    covered = {identity_by_entity[entity_key] for entity_key in selected_entities}
    for identity in sorted(set(identity_hops) - covered):
        selected_entities.add(min(entities_by_identity[identity]))
    return tuple(accepted), tuple(sorted(selected_entities))


__all__ = ["select_topology_groups"]
