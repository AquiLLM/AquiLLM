"""Closed synthetic graph replay inputs; private corpora use separate files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from math import fsum, isfinite
from pathlib import Path

from apps.knowledge_graph.evals.projected_replay_snapshot import key, projected_snapshot
from apps.knowledge_graph.retrieval.ppr_policy import (
    PPRPolicySignalsV1,
    classify_ppr_intent,
)
from apps.knowledge_graph.retrieval.topology.contracts import (
    HybridBranchKind,
    ProjectedSeedV1,
)

SUMMARY_KEYS = frozenset(
    "deduplicated_spans resolved_spans ambiguous_spans "
    "retained_matches retained_seeds exact_tier_matches minimum_extraction_score "
    "cap_pressure requested_chunks mapped_top_chunks required_top_chunks "
    "rank_one_mapped rank_one_channel_agreement".split()
)


@dataclass(frozen=True)
class PPRReplayCase:
    raw: dict
    snapshot: object
    config: object
    seeds: tuple[ProjectedSeedV1, ...]
    signals: PPRPolicySignalsV1
    node_names: dict[str, str]
    chunk_ids: dict[str, int]


def parse_case(raw):
    if not isinstance(raw, dict):
        raise ValueError("case must be a mapping")
    for field in ("id", "question", "split"):
        if not isinstance(raw.get(field), str) or not raw[field].strip():
            raise ValueError(f"missing {field}")
    if raw["split"] not in ("regression", "development", "held_out"):
        raise ValueError("unsupported split")
    nodes = raw.get("nodes")
    if (
        not isinstance(nodes, list)
        or not 1 <= len(nodes) <= 200
        or any(not isinstance(node, str) or not node for node in nodes)
        or len(set(nodes)) != len(nodes)
    ):
        raise ValueError("invalid or duplicate nodes")
    edges = raw.get("edges")
    if not isinstance(edges, list) or len(edges) > 1000:
        raise ValueError("invalid edges")
    for edge in edges:
        if (
            not isinstance(edge, list)
            or len(edge) != 3
            or edge[0] not in nodes
            or edge[1] not in nodes
            or edge[0] == edge[1]
            or type(edge[2]) not in (int, float)
            or not isfinite(edge[2])
            or not 0 < edge[2] <= 2
        ):
            raise ValueError("invalid edge")
    if len({(edge[0], edge[1]) for edge in edges}) != len(edges):
        raise ValueError("duplicate edge groups")
    documents = raw.get("evidence_documents", ["document"] * len(edges))
    if documents != ["document"] * len(edges):
        raise ValueError("outside-scope evidence")
    seed_rows = raw.get("seeds")
    if not isinstance(seed_rows, list) or not 1 <= len(seed_rows) <= 64:
        raise ValueError("invalid seeds")
    for row in seed_rows:
        if (
            not isinstance(row, list)
            or len(row) != 2
            or row[0] not in nodes
            or type(row[1]) not in (int, float)
            or not isfinite(row[1])
            or row[1] <= 0
        ):
            raise ValueError("invalid seed weight")
    if len({row[0] for row in seed_rows}) != len(seed_rows):
        raise ValueError("duplicate seeds")
    total = fsum(row[1] for row in seed_rows)
    if not isfinite(total):
        raise ValueError("nonfinite seed mass")
    seeds = tuple(
        sorted(
            (ProjectedSeedV1(key(name), weight / total) for name, weight in seed_rows),
            key=lambda row: row.identity_key,
        )
    )
    summary = raw.get("support_summary", {})
    if (
        not isinstance(summary, dict)
        or len(summary) > 16
        or set(summary) - SUMMARY_KEYS
        or any(
            type(v) not in (int, float, bool) or not isfinite(v) or v < 0
            for v in summary.values()
        )
    ):
        raise ValueError("invalid numeric support summary")
    encoded = json.dumps(
        {k: v.hex() if type(v) is float else v for k, v in summary.items()},
        sort_keys=True,
    ).encode()
    signals = PPRPolicySignalsV1(
        HybridBranchKind(raw.get("branch", "direct")),
        classify_ppr_intent(raw["question"]),
        len(seeds),
        raw.get("support_status", "unknown"),
        raw.get("cap_pressure", False),
        0.0,
        sha256(encoded).hexdigest(),
    )
    for field in ("required_chunks", "irrelevant_chunks"):
        values = raw.get(field, [])
        if (
            not isinstance(values, list)
            or len(set(values)) != len(values)
            or any(type(value) is not int or value < 1 for value in values)
        ):
            raise ValueError("invalid chunk labels")
    groups = raw.get("required_groups", [])
    if not isinstance(groups, list) or any(
        not isinstance(group, list)
        or not group
        or any(type(value) is not int or value < 1 for value in group)
        for group in groups
    ):
        raise ValueError("invalid support-group labels")
    snapshot, config = projected_snapshot(
        nodes=nodes,
        edges=tuple(tuple(edge) for edge in edges),
        iterations=8,
        max_nodes=raw.get("max_nodes", 200),
    )
    return PPRReplayCase(
        raw,
        snapshot,
        config,
        seeds,
        signals,
        {key(node): node for node in nodes},
        {key(f"chunk-{i}"): i for i in range(1, len(edges) + 1)},
    )


def load_cases(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("unsupported PPR replay schema")
    if not isinstance(payload.get("cases"), list):
        raise ValueError("missing cases")
    cases = tuple(parse_case(raw) for raw in payload["cases"])
    if len({case.raw["id"] for case in cases}) != len(cases):
        raise ValueError("duplicate case IDs")
    return cases
