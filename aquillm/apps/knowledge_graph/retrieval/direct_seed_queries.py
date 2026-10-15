# ruff: noqa: E501,E701,E702
"""Scoped database queries for direct seed lookup."""

from __future__ import annotations

from time import monotonic

from apps.knowledge_graph.retrieval.direct_seed_contracts import DirectResolutionTier

from .direct_seed_sql import bounded_seed_read
from .direct_seed_types import DirectSeedCandidateRow
from .graph_seed_reads_v2 import seed_query_rows


# fmt: off
def _load_membership_states(**options: object) -> tuple[dict[str, object], ...]:
    from apps.knowledge_graph.models import CollectionGraphMembershipState
    fields = ("collection_id", "active_artifact_id", "registry_epoch", "membership_checksum", "resolver_version", "resolution_config_checksum")
    with bounded_seed_read(using=options["using"], deadline=options.get("deadline"), clock=options.get("clock", monotonic)):
        query = CollectionGraphMembershipState.objects.using(options["using"]).filter(collection_id__in=options["collection_ids"]).order_by("collection_id").values(*fields)
        if options.get("read_budget") is None:
            return tuple(query)
        return seed_query_rows(query, maximum=len(options["collection_ids"]), budget=options["read_budget"])


def _load_candidate_rows(**options: object) -> tuple[DirectSeedCandidateRow, ...]:
    from .direct_seed_queryset import _candidate_queryset

    bounded = _candidate_queryset(**options)
    tier = options["tier"]
    with bounded_seed_read(using=options["using"], deadline=options.get("deadline"), alias=tier is DirectResolutionTier.ALIAS, clock=options.get("clock", monotonic)):
        rows = seed_query_rows(bounded, maximum=int(options["limit"]) + 1, budget=options.get("read_budget"))
    if len(rows) > int(options["limit"]):
        raise ValueError("candidate result exceeds its hard cap")
    if any((row["automatic_link_count"] or 0) > 1 for row in rows):
        raise ValueError("automatic membership is not unique")
    return tuple(DirectSeedCandidateRow(int(row["id"]), int(row["artifact_id"]), str(row["entity_type"]), row["automatic_canonical_entity_id"], float(row["similarity"])) for row in rows)

_NATIVE_CANDIDATE_LOADER = _load_candidate_rows
