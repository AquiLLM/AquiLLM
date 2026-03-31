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
