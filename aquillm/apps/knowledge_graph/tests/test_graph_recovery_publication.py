from __future__ import annotations

from importlib import import_module

import pytest
from test_graph_recovery import (
    _recovery,
)


@pytest.fixture(autouse=True)
def no_persisted_capacity_failure(monkeypatch):
    monkeypatch.setattr(_recovery(), "_exact_capacity_failure", lambda **_kwargs: False)


def test_missing_document_is_retried_after_broker_publication_failure(monkeypatch):
    recovery = _recovery()
    builds = import_module("apps.knowledge_graph.services.builds")
    attempts = []
    monkeypatch.setattr(
        builds,
        "derive_current_document_build_key",
        lambda *_args: "b" * 64,
    )
    monkeypatch.setattr(recovery, "_exact_artifact_exists", lambda **_kwargs: False)

    def publish(document_id, source_hash):
        attempts.append((document_id, source_hash))
        if len(attempts) == 1:
            raise ConnectionError("private broker detail")

    monkeypatch.setattr(builds, "enqueue_document_build", publish)
    document_id = __import__("uuid").UUID("11111111-1111-4111-8111-111111111111")

    first = recovery._recover_document(document_id, "a" * 64)
    second = recovery._recover_document(document_id, "a" * 64)

    assert first == recovery.RecoveryOutcome.PUBLISH_FAILED
    assert second == recovery.RecoveryOutcome.PUBLISHED
    assert attempts == [(document_id, "a" * 64), (document_id, "a" * 64)]
