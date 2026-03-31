"""Quality fixtures for Memgraph compatibility relation filtering."""

from __future__ import annotations

import importlib
import sys
import types


def _load_memgraph_compat_module(monkeypatch):
    fake_memgraph_memory = types.ModuleType("mem0.memory.memgraph_memory")
    fake_memgraph_memory.MemoryGraph = type("MemoryGraph", (), {})

    fake_factory = types.ModuleType("mem0.utils.factory")
    fake_factory.EmbedderFactory = types.SimpleNamespace(create=lambda *_args, **_kwargs: None)
    fake_factory.LlmFactory = types.SimpleNamespace(create=lambda *_args, **_kwargs: None)

    fake_langchain_memgraph = types.ModuleType("langchain_memgraph.graphs.memgraph")
    fake_langchain_memgraph.Memgraph = type("Memgraph", (), {})

    monkeypatch.setitem(sys.modules, "mem0.memory.memgraph_memory", fake_memgraph_memory)
    monkeypatch.setitem(sys.modules, "mem0.utils.factory", fake_factory)
    monkeypatch.setitem(sys.modules, "langchain_memgraph.graphs.memgraph", fake_langchain_memgraph)

    from lib.memory.mem0 import memgraph_compat as module

    return importlib.reload(module)


def test_prepare_graph_relation_drops_self_referential_edges(monkeypatch):
    module = _load_memgraph_compat_module(monkeypatch)

    relation = module.prepare_graph_relation(
        {"source": "user", "relationship": "prefers", "destination": "user"}
    )

    assert relation is None


def test_prepare_graph_relation_drops_generic_identity_edges(monkeypatch):
    module = _load_memgraph_compat_module(monkeypatch)

    relation = module.prepare_graph_relation(
        {"source": "user", "relationship": "name", "destination": "user"}
    )

    assert relation is None


def test_prepare_graph_relation_preserves_useful_tooling_edges(monkeypatch):
    module = _load_memgraph_compat_module(monkeypatch)

    relation = module.prepare_graph_relation(
        {"source": "Jack", "relationship": "uses", "destination": "Memgraph"}
    )

    assert relation == {
        "source": "jack",
        "relationship": "USES",
        "destination": "memgraph",
    }


def test_prepare_graph_relation_normalizes_entities_and_relation_names(monkeypatch):
    module = _load_memgraph_compat_module(monkeypatch)

    relation = module.prepare_graph_relation(
        {"source": " The User ", "relationship": "works on", "destination": " AquiLLM "}
    )

    assert relation == {
        "source": "user",
        "relationship": "WORKS_ON",
        "destination": "aquillm",
    }


def test_search_graph_db_batches_edge_queries(monkeypatch):
    module = _load_memgraph_compat_module(monkeypatch)

    query_calls: list[dict[str, object]] = []

    def fake_query(_cypher, params=None):
        query_calls.append(params or {})
        if len(query_calls) == 1:
            return [{"source": "jack", "source_id": 1, "relationship": "USES", "relation_id": 11, "destination": "memgraph", "destination_id": 2}]
        return [{"source": "aquillm", "source_id": 7, "relationship": "WORKS_ON", "relation_id": 12, "destination": "jack", "destination_id": 3}]

    graph_obj = module.CompatibleMemgraphMemoryGraph.__new__(module.CompatibleMemgraphMemoryGraph)
    graph_obj.graph = types.SimpleNamespace(query=fake_query)
    graph_obj.embedding_model = types.SimpleNamespace(embed=lambda _node: [0.1, 0.2])
    graph_obj.threshold = 0.7
    graph_obj._nearest_nodes = lambda *_args, **_kwargs: [
        {"node_id": 1, "similarity": 0.95},
        {"node_id": 3, "similarity": 0.91},
    ]
    graph_obj._base_params = lambda filters: {"user_id": filters["user_id"]}
    graph_obj._node_props = lambda filters, name_param=None: "user_id: $user_id"

    results = graph_obj._search_graph_db(["jack"], {"user_id": "1"}, limit=5)

    assert len(query_calls) == 2
    assert query_calls[0]["node_ids"] == [1, 3]
    assert query_calls[1]["node_ids"] == [1, 3]
    assert [row["similarity"] for row in results] == [0.95, 0.91]


def test_search_graph_db_fetches_seeded_candidates_once(monkeypatch):
    module = _load_memgraph_compat_module(monkeypatch)

    fetch_calls: list[list[str]] = []

    def fake_query(_cypher, _params=None):
        return []

    graph_obj = module.CompatibleMemgraphMemoryGraph.__new__(module.CompatibleMemgraphMemoryGraph)
    graph_obj.graph = types.SimpleNamespace(query=fake_query)
    graph_obj.embedding_model = types.SimpleNamespace(embed=lambda _node: [0.1, 0.2])
    graph_obj.threshold = 0.7
    graph_obj._fetch_candidate_nodes = lambda filters, node_list=None: fetch_calls.append(list(node_list or [])) or []
    graph_obj._nearest_nodes = lambda *_args, **_kwargs: []
    graph_obj._base_params = lambda filters: {"user_id": filters["user_id"]}
    graph_obj._node_props = lambda filters, name_param=None: "user_id: $user_id"

    graph_obj._search_graph_db(["memgraph", "qdrant"], {"user_id": "1"}, limit=5)

    assert fetch_calls == [["memgraph", "qdrant"]]


def test_fetch_candidate_nodes_respects_candidate_cap(monkeypatch):
    monkeypatch.setenv("MEM0_GRAPH_SEARCH_CANDIDATE_LIMIT", "17")
    module = _load_memgraph_compat_module(monkeypatch)

    captured: dict[str, object] = {}

    def fake_query(cypher, params=None):
        captured["cypher"] = cypher
        captured["params"] = params or {}
        return []

    graph_obj = module.CompatibleMemgraphMemoryGraph.__new__(module.CompatibleMemgraphMemoryGraph)
    graph_obj.graph = types.SimpleNamespace(query=fake_query)
    graph_obj._base_params = lambda filters: {"user_id": filters["user_id"]}
    graph_obj._node_props = lambda filters, name_param=None: "user_id: $user_id"

    graph_obj._fetch_candidate_nodes({"user_id": "1"})

    assert "LIMIT $candidate_limit" in captured["cypher"]
    assert captured["params"]["candidate_limit"] == 17


def test_fetch_candidate_nodes_uses_query_seeds_when_available(monkeypatch):
    monkeypatch.setenv("MEM0_GRAPH_SEARCH_CANDIDATE_LIMIT", "17")
    module = _load_memgraph_compat_module(monkeypatch)

    captured: dict[str, object] = {}

    def fake_query(cypher, params=None):
        captured["cypher"] = cypher
        captured["params"] = params or {}
        if params and "seed_tokens" in params:
            return [{"name": "memgraph", "node_id": 1, "embedding": [0.1, 0.2]}]
        return []

    graph_obj = module.CompatibleMemgraphMemoryGraph.__new__(module.CompatibleMemgraphMemoryGraph)
    graph_obj.graph = types.SimpleNamespace(query=fake_query)
    graph_obj._base_params = lambda filters: {"user_id": filters["user_id"]}
    graph_obj._node_props = lambda filters, name_param=None: "user_id: $user_id"

    graph_obj._fetch_candidate_nodes({"user_id": "1"}, node_list=["What do you remember about Memgraph and Qdrant?"])

    assert "seed_tokens" in captured["params"]
    assert "memgraph" in captured["params"]["seed_tokens"]
    assert "qdrant" in captured["params"]["seed_tokens"]
    assert "any(token IN $seed_tokens" in captured["cypher"]


def test_fetch_candidate_nodes_falls_back_to_generic_scan_when_no_useful_seeds(monkeypatch):
    monkeypatch.setenv("MEM0_GRAPH_SEARCH_CANDIDATE_LIMIT", "17")
    module = _load_memgraph_compat_module(monkeypatch)

    captured: dict[str, object] = {}

    def fake_query(cypher, params=None):
        captured["cypher"] = cypher
        captured["params"] = params or {}
        return []

    graph_obj = module.CompatibleMemgraphMemoryGraph.__new__(module.CompatibleMemgraphMemoryGraph)
    graph_obj.graph = types.SimpleNamespace(query=fake_query)
    graph_obj._base_params = lambda filters: {"user_id": filters["user_id"]}
    graph_obj._node_props = lambda filters, name_param=None: "user_id: $user_id"

    graph_obj._fetch_candidate_nodes({"user_id": "1"}, node_list=["I", "A"])

    assert "seed_tokens" not in captured["params"]
    assert "any(token IN $seed_tokens" not in captured["cypher"]


def test_search_graph_db_logs_timing_breakdown(monkeypatch):
    module = _load_memgraph_compat_module(monkeypatch)

    info_logs: list[tuple[str, tuple[object, ...]]] = []

    graph_obj = module.CompatibleMemgraphMemoryGraph.__new__(module.CompatibleMemgraphMemoryGraph)
    graph_obj.graph = types.SimpleNamespace(query=lambda *_args, **_kwargs: [])
    graph_obj.embedding_model = types.SimpleNamespace(embed=lambda _node: [0.1, 0.2])
    graph_obj.threshold = 0.7
    graph_obj._fetch_candidate_nodes_with_stats = lambda filters, node_list=None: (
        [{"name": "memgraph", "node_id": 1, "embedding": [0.1, 0.2]}],
        {
            "strategy": "seeded",
            "candidate_limit": 17,
            "seed_name_count": 1,
            "seed_token_count": 2,
            "fetched_candidate_count": 1,
            "fetch_ms": 1.5,
        },
    )
    graph_obj._nearest_nodes_with_stats = lambda *_args, **_kwargs: (
        [{"node_id": 1, "similarity": 0.95}],
        {
            "mode": "numpy",
            "candidate_count": 1,
            "valid_candidate_count": 1,
            "passed_threshold_count": 1,
            "score_ms": 0.7,
        },
    )
    graph_obj._base_params = lambda filters: {"user_id": filters["user_id"]}
    graph_obj._node_props = lambda filters, name_param=None: "user_id: $user_id"
    monkeypatch.setattr(
        module.logger,
        "info",
        lambda message, *args: info_logs.append((message, args)),
    )

    graph_obj._search_graph_db(["memgraph"], {"user_id": "1"}, limit=5)

    assert info_logs[0][0] == (
        "Memgraph graph search stats: strategy=%s candidate_limit=%d seed_names=%d "
        "seed_tokens=%d fetched_candidates=%d score_mode=%s valid_candidates=%d "
        "passed_threshold=%d query_nodes=%d fetch_ms=%.2f score_ms=%.2f edge_query_ms=%.2f total_ms=%.2f"
    )
