"""Converged recurrence for offline diagnostics; serving remains eight steps."""

from dataclasses import replace
from math import fsum

from apps.knowledge_graph.retrieval.ppr_kernel import run_ppr_kernel


def reference_run(*, snapshot, config, edges, seeds, restart):
    initial = run_ppr_kernel(
        nodes=snapshot.identity_keys,
        edges=edges,
        seeds={seed.identity_key: seed.mass for seed in seeds},
        config=replace(config, ppr_restart=restart, ppr_iterations=1),
        order_key=lambda value: value,
    )
    rows = dict(initial.transition_rows)
    seed_map = {seed.identity_key: seed.mass for seed in seeds}
    total = fsum(seed_map.values())
    base = {node: seed_map.get(node, 0.0) / total for node in snapshot.identity_keys}
    scores = dict(base)
    residual = float("inf")
    for iteration in range(1, 501):
        incoming = {node: [] for node in scores}
        dangling = fsum(scores[node] for node in scores if not rows[node])
        for source, row in rows.items():
            for target, weight in row:
                incoming[target].append(scores[source] * weight)
        updated = {
            node: restart * base[node]
            + (1 - restart) * (fsum(incoming[node]) + dangling * base[node])
            for node in scores
        }
        residual = fsum(abs(updated[node] - scores[node]) for node in scores)
        scores = updated
        if residual <= 1e-10:
            break
    return (
        initial.scores,
        scores,
        {
            "iterations": iteration,
            "residual_l1": residual,
            "stopping_reason": "tolerance" if residual <= 1e-10 else "iteration_limit",
        },
    )
