"""Focused regression tests for Mem0 graph payload normalization."""

from __future__ import annotations

import importlib
import sys
import types


def _reload_mem0_client(monkeypatch):
    from lib.memory.mem0 import client as client_module

    importlib.reload(client_module)
    return client_module


def test_register_memgraph_compat_provider_patches_upstream_payload_normalization(monkeypatch):
    """Defensively normalize single-entity graph payloads even on upstream MemoryGraph."""
    client_module = _reload_mem0_client(monkeypatch)
    fake_factory_module = types.ModuleType("mem0.utils.factory")
    fake_memgraph_module = types.ModuleType("mem0.memory.memgraph_memory")

    class FakeGraphStoreFactory:
        provider_to_class = {"memgraph": "mem0.memory.memgraph_memory.MemoryGraph"}

    class FakeMemoryGraph:
        def _retrieve_nodes_from_data(self, search_results, *args, **kwargs):
            entities = search_results["tool_calls"][0]["arguments"]["entities"]
            return [item["entity"] for item in entities]

    fake_factory_module.GraphStoreFactory = FakeGraphStoreFactory
    fake_memgraph_module.MemoryGraph = FakeMemoryGraph
    monkeypatch.setitem(sys.modules, "mem0.utils.factory", fake_factory_module)
    monkeypatch.setitem(sys.modules, "mem0.memory.memgraph_memory", fake_memgraph_module)

    client_module._register_memgraph_compat_provider()

    graph = FakeMemoryGraph()
    entities = graph._retrieve_nodes_from_data(
        {
            "content": None,
            "tool_calls": [
                {
                    "name": "extract_entities",
                    "arguments": {
                        "entities": {
                            "entity": "figure 1",
                            "entity_type": "Figure",
                        }
                    },
                }
            ],
        }
    )

    assert entities == ["figure 1"]
