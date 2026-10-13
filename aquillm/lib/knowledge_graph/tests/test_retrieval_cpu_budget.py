"""CPU extractor timeout preserves the enclosing graph budgets."""

from __future__ import annotations

from lib.knowledge_graph import retrieval_config as config
from lib.knowledge_graph.tests.test_retrieval_config import _direct


def test_cpu_extractor_timeout_preserves_parent_budgets() -> None:
    settings = config.load_hybrid_retrieval_settings(
        {
            **_direct(),
            "KG_QUERY_EXTRACTOR_TIMEOUT_MS": "3000",
            "KG_GRAPH_OVERALL_TIMEOUT_MS": "5000",
            "KG_GRAPH_DIRECT_TIMEOUT_MS": "4500",
            "KG_GRAPH_EXTENDED_TIMEOUT_MS": "4500",
        }
    )
    assert settings.query_extractor_timeout_ms == 3000
    assert settings.graph_overall_timeout_ms == 5000
    assert settings.graph_direct_timeout_ms == 4500
    assert settings.graph_extended_timeout_ms == 4500

    defaults = config.load_hybrid_retrieval_settings(
        {"KG_QUERY_EXTRACTOR_TIMEOUT_MS": "3000"}
    )
    assert defaults.graph_overall_timeout_ms == 300
    assert defaults.graph_direct_timeout_ms == 125
