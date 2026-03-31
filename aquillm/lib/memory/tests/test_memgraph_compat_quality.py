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
