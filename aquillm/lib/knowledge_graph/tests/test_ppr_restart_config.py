"""PageRank restart mode is an opt-in exact configuration value."""

import pytest

from lib.knowledge_graph.retrieval_config import (
    HybridRetrievalConfigError,
    load_hybrid_retrieval_settings,
)


def test_restart_mode_defaults_fixed_and_accepts_shadow_or_adaptive():
    assert load_hybrid_retrieval_settings({}).ppr_restart_mode == "fixed"
    assert (
        load_hybrid_retrieval_settings(
            {"KG_PPR_RESTART_MODE": "shadow"}
        ).ppr_restart_mode
        == "shadow"
    )
    assert (
        load_hybrid_retrieval_settings(
            {"KG_PPR_RESTART_MODE": "adaptive"}
        ).ppr_restart_mode
        == "adaptive"
    )


@pytest.mark.parametrize("invalid", ("", "ADAPTIVE", "on", "adaptive ", "fixed\n"))
def test_restart_mode_rejects_malformed_values(invalid):
    with pytest.raises(HybridRetrievalConfigError, match="KG_PPR_RESTART_MODE"):
        load_hybrid_retrieval_settings({"KG_PPR_RESTART_MODE": invalid})
