"""Independent projection budget configuration checks."""

import pytest

from lib.knowledge_graph.retrieval_config import (
    HybridRetrievalConfigError,
    load_hybrid_retrieval_settings,
)


def test_projection_timeout_changes_without_changing_retrieval_budget() -> None:
    settings = load_hybrid_retrieval_settings({"KG_PROJECTION_TIMEOUT_MS": "12000"})
    assert settings.projection_timeout_ms == 12000
    assert settings.graph_overall_timeout_ms == 300


@pytest.mark.parametrize("value", ("100", "60000"))
def test_projection_timeout_accepts_supported_edges(value: str) -> None:
    settings = load_hybrid_retrieval_settings({"KG_PROJECTION_TIMEOUT_MS": value})
    assert settings.projection_timeout_ms == int(value)


@pytest.mark.parametrize("value", ("99", "60001", "0100", "1.0"))
def test_projection_timeout_rejects_out_of_range_and_noncanonical_values(
    value: str,
) -> None:
    with pytest.raises(HybridRetrievalConfigError, match="KG_PROJECTION_TIMEOUT_MS"):
        load_hybrid_retrieval_settings({"KG_PROJECTION_TIMEOUT_MS": value})


def test_oversized_projection_integer_is_a_fixed_configuration_error() -> None:
    with pytest.raises(
        HybridRetrievalConfigError, match="KG_PROJECTION_BATCH_SIZE"
    ) as caught:
        load_hybrid_retrieval_settings({"KG_PROJECTION_BATCH_SIZE": "9" * 5000})
    assert "999999" not in str(caught.value)
