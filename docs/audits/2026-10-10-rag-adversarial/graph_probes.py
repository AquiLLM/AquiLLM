"""Offline adversarial graph probes; no Django setup, database, or HTTP access.

Run from any directory with:
    rtk proxy python -B docs/audits/2026-10-10-rag-adversarial/graph_probes.py

The resolver accepts identical inputs for same-name real-world homonyms because
its contract contains no contextual identity discriminator. The first probe
demonstrates the automatic merge, not a model's ability to extract those inputs.
The remaining probes exercise the shipping provider-neutral PPR primitives.
"""

from pathlib import Path
from types import SimpleNamespace
import sys


REPOSITORY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY / "aquillm"))

from apps.knowledge_graph.resolution.canonical import (  # noqa: E402
    CanonicalEntityInput,
    resolve_canonical_entities,
)
from apps.knowledge_graph.retrieval.ppr import (  # noqa: E402
    RetrievalDirection,
    raw_edge_weight,
)
from apps.knowledge_graph.retrieval.ppr_kernel import (  # noqa: E402
    WeightedEdge,
    run_ppr_kernel,
)


def exact_name_homonym_probe():
    """Distinct projects named Mercury have no identifier/version anchors."""
    rows = tuple(
        CanonicalEntityInput(
            entity_id=index,
            collection_id=index,
            artifact_id=index * 100,
            cluster_key=f"{index:064x}",
            label="Mercury",
            normalized_label="mercury",
            entity_type="model",
        )
        for index in (1, 2)
    )
    result = resolve_canonical_entities(rows)
    observed = tuple(component.entity_ids for component in result.components)
    assert observed == ((1, 2),)
    decision = result.decisions[0]
    assert (decision.method, decision.outcome.value, decision.score) == (
        "exact_name_or_alias", "automatic", 0.99
    )
    print("Homonym components:", observed)
    print("Homonym disposition:", decision.method, decision.outcome.value, decision.score)


def _rank(edges, nodes=("A", "B", "C")):
    return run_ppr_kernel(
        nodes=nodes,
        edges=edges,
        seeds={"A": 1.0},
        config=SimpleNamespace(
            max_nodes=3, max_edges=30, ppr_restart=0.25, ppr_iterations=20
        ),
        order_key=lambda key: key,
    )


def parallel_predicate_probe():
    """Projected input conversion drops predicate labels before this kernel."""
    base = (WeightedEdge("A", "B", 1.0), WeightedEdge("A", "C", 1.0))
    extra_predicates = base + tuple(WeightedEdge("A", "B", 1.0) for _ in range(8))
    baseline_row = dict(_rank(base).transition_rows)["A"]
    inflated_row = dict(_rank(extra_predicates).transition_rows)["A"]
    assert baseline_row == (("B", 0.5), ("C", 0.5))
    assert inflated_row == (("B", 0.9), ("C", 0.1))
    print("Two predicate transitions:", baseline_row)
    print("Ten predicate transitions:", inflated_row)


def reverse_direction_cancellation_probe():
    """A uniform row direction multiplier disappears during normalization."""
    def weight(direction):
        return raw_edge_weight(
            direction=direction,
            confidence=0.9,
            support_count=1,
            destination_retrieval_utility=1.0,
        )

    forward = _rank((WeightedEdge("A", "B", weight(RetrievalDirection.FORWARD)),))
    reverse = _rank((WeightedEdge("A", "B", weight(RetrievalDirection.REVERSE_DIRECTED)),))
    assert forward.transition_rows == reverse.transition_rows
    assert forward.scores == reverse.scores
    print("Forward/reverse uniform-row rankings identical:", forward.scores)


if __name__ == "__main__":
    exact_name_homonym_probe()
    parallel_predicate_probe()
    reverse_direction_cancellation_probe()
