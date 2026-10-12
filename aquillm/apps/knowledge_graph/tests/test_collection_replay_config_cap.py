"""Committed collection manifest limits survive real replay configuration."""
from types import SimpleNamespace

import pytest

from apps.knowledge_graph.resolution.collection import CollectionResolutionConfig
from apps.knowledge_graph.resolution import vocabulary_builds, vocabulary_capture


def replay(monkeypatch, cap):
    original = CollectionResolutionConfig()
    marker = {name: getattr(original, name) for name in (
        "max_entities", "max_memberships", "max_relations", "max_links")}
    marker["max_document_inputs"] = cap
    artifact = SimpleNamespace(collection_scope_id=17, _state=SimpleNamespace(db="default"))
    calls = []

    def current_policy(got_artifact, *, collection_id, using):
        assert got_artifact is artifact
        calls.append((collection_id, using))
        return None

    monkeypatch.setattr(vocabulary_capture, "current_artifact_policy", current_policy)
    result = vocabulary_builds.collection_replay_config(artifact, marker)
    assert calls == [(17, "default")]
    return result, marker


def test_replay_retains_exact_nondefault_committed_manifest_cap(monkeypatch):
    assert CollectionResolutionConfig().max_document_inputs != 3
    config, marker = replay(monkeypatch, 3)
    assert type(config) is CollectionResolutionConfig
    assert config.max_document_inputs == marker["max_document_inputs"] == 3
    for name in ("max_entities", "max_memberships", "max_relations", "max_links"):
        assert getattr(config, name) == marker[name]
    assert config.vocabulary is None


@pytest.mark.parametrize("cap", [0, -1, True, "3", 1.5])
def test_replay_rejects_invalid_committed_manifest_cap(monkeypatch, cap):
    with pytest.raises(ValueError, match="max_document_inputs"):
        replay(monkeypatch, cap)


def test_replay_rejects_manifest_cap_above_existing_maximum(monkeypatch):
    maximum = CollectionResolutionConfig().max_document_inputs
    with pytest.raises(ValueError, match="max_document_inputs"):
        replay(monkeypatch, maximum + 1)
