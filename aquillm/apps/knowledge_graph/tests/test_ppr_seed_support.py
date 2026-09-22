"""Policy support is derived from private bounded source diagnostics."""

from dataclasses import replace

import pytest

from apps.knowledge_graph.retrieval.direct_seed_contracts import (
    DirectEntityMatchV1,
    DirectResolutionTier,
    DirectSeedAmbiguityV1,
    DirectSeedDiagnosticsV1,
    DirectSeedOutcomeV1,
    ResolvedDirectSeedV1,
)
from apps.knowledge_graph.retrieval.ppr_seed_support import (
    PPRSeedSupportV1,
    summarize_direct_support,
    summarize_extended_support,
)
from apps.knowledge_graph.retrieval.types import GraphExpansionSeed
from apps.knowledge_graph.tests.projected_ppr_fixtures import key


def _direct(confidence=0.9, tier=DirectResolutionTier.NAME, ambiguous=False):
    factor = {DirectResolutionTier.NAME: 0.95, DirectResolutionTier.EMBEDDING: 0.8}[
        tier
    ]
    match = DirectEntityMatchV1(
        0,
        key("entity"),
        key("component"),
        "person",
        tier,
        confidence,
        1.0,
        confidence * factor,
    )
    ambiguity = (
        (DirectSeedAmbiguityV1(1, DirectResolutionTier.NAME, 2, 2),)
        if ambiguous
        else ()
    )
    diagnostics = DirectSeedDiagnosticsV1(
        2 if ambiguous else 1,
        2 if ambiguous else 1,
        1,
        int(ambiguous),
        0,
        int(tier is DirectResolutionTier.EMBEDDING),
        int(tier is DirectResolutionTier.EMBEDDING),
    )
    return DirectSeedOutcomeV1(
        (match,),
        (ResolvedDirectSeedV1(key("component"), (key("entity"),), 1.0),),
        ambiguity,
        diagnostics,
        None,
    )


def test_direct_support_separates_absolute_confidence_from_same_normalized_seed():
    high = summarize_direct_support(_direct(0.9), max_seeds=32)
    low = summarize_direct_support(_direct(0.7), max_seeds=32)
    assert high.status == "supported"
    assert low.status == "insufficient"
    assert high.digest != low.digest
    assert dict(high.summary)["minimum_extraction_score"] == 0.9
    assert tuple(key for key, _ in high.summary) == tuple(sorted(dict(high.summary)))
    assert not any(key("entity") in str(value) for _, value in high.summary)


def test_direct_ambiguity_seed_cap_and_embedding_only_abstain():
    assert (
        summarize_direct_support(_direct(0.9, ambiguous=True), max_seeds=32).status
        == "insufficient"
    )
    capped = summarize_direct_support(_direct(0.9), max_seeds=1)
    assert capped.cap_pressure is True
    assert capped.status == "insufficient"
    embedded = summarize_direct_support(
        _direct(0.9, DirectResolutionTier.EMBEDDING), max_seeds=32
    )
    assert embedded.status == "insufficient"
    assert dict(embedded.summary)["exact_tier_matches"] == 0


def test_extended_support_needs_top_mapping_and_rank_one_channel_agreement():
    ranked = (GraphExpansionSeed(10, 1, 100.0), GraphExpansionSeed(20, 2, 0.01))
    common = dict(
        ranked_seeds=ranked,
        mapped_chunk_ids=frozenset((10, 20)),
        vector_chunk_ids=(10,),
        trigram_chunk_ids=(10,),
        exact_chunk_ids=(),
        retained_identity_count=2,
        max_seeds=64,
    )
    supported = summarize_extended_support(**common)
    assert supported.status == "supported"
    assert dict(supported.summary)["mapped_top_chunks"] == 2
    assert dict(supported.summary)["required_top_chunks"] == 2
    assert (
        summarize_extended_support(
            **(common | {"mapped_chunk_ids": frozenset((10,))})
        ).status
        == "insufficient"
    )
    assert (
        summarize_extended_support(**(common | {"trigram_chunk_ids": ()})).status
        == "insufficient"
    )
    assert (
        summarize_extended_support(**(common | {"trigram_chunk_ids": None})).status
        == "unknown"
    )
    assert (
        summarize_extended_support(
            **(common | {"retained_identity_count": 64})
        ).cap_pressure
        is True
    )
    # Rank weights and absolute seed mass are never treated as calibrated confidence.
    smaller = summarize_extended_support(
        **(
            common
            | {
                "ranked_seeds": (
                    replace(ranked[0], restart_weight=0.001),
                    replace(ranked[1], restart_weight=100.0),
                )
            }
        )
    )
    assert smaller.status == supported.status
    assert smaller.digest == supported.digest


def test_summary_rejects_mixed_branch_keys():
    with pytest.raises(ValueError, match="branch"):
        PPRSeedSupportV1(
            "unknown",
            False,
            (("deduplicated_spans", 1), ("mapped_top_chunks", 1)),
            "0" * 64,
        )
