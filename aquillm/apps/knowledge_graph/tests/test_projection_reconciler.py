# ruff: noqa: F401
from __future__ import annotations

from contextlib import nullcontext
from datetime import timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest
from django.utils import timezone

from apps.knowledge_graph.projection import reconciler, worker
from apps.knowledge_graph.projection.memgraph_driver import MemgraphDriverError
from apps.knowledge_graph.projection.state_repository import StateLeaseV1


def _lease(projection_id):
    return StateLeaseV1(
        projection_id, "worker-a", timezone.now() + timedelta(seconds=300), 1
    )


@pytest.fixture(autouse=True)
def _stub_fenced_renewal(monkeypatch):
    monkeypatch.setattr(
        worker.FunctionProjectionStateRepository,
        "renew",
        lambda self, **kwargs: StateLeaseV1(
            kwargs["projection_id"],
            kwargs["owner"],
            kwargs["now"] + timedelta(seconds=kwargs["lease_seconds"]),
            1,
        ),
    )


def test_project_generation_replays_partial_staging_and_ready_cas(monkeypatch):
    projection_id = uuid4()
    calls = []
    lease = _lease(projection_id)
    generation_key = "a" * 64
    private_checksum = "c" * 64
    bundle = SimpleNamespace(generation=SimpleNamespace(generation_key=generation_key))
    private_rows = (SimpleNamespace(projection_chunk_key="d" * 64),)
    validation = SimpleNamespace(
        valid=True,
        generation_key=generation_key,
        validation_checksum="b" * 64,
    )

    class Source:
        def load_projection_bundle(self, **_kwargs):
            calls.append("load_bundle")
            return bundle

        def load_private_chunk_references(self, **_kwargs):
            calls.append("load_private")
            return private_rows

        def persist_chunk_references(self, **kwargs):
            assert kwargs["rows"] is private_rows
            calls.append("persist_private")
            return private_checksum

    source = Source()

    def stage(**kwargs):
        assert kwargs["private_mapping_checksum"] == private_checksum
        calls.append("stage")

    graph = SimpleNamespace(
        write_staging_generation=stage,
        validate_generation=lambda **_kwargs: validation,
        mark_generation_ready=lambda **_kwargs: calls.append("graph_ready"),
    )
    monkeypatch.setattr(worker, "claim_projection_lease", lambda **_kwargs: lease)
    monkeypatch.setattr(worker, "_postgres_repository", lambda: source)
    monkeypatch.setattr(worker, "_memgraph_repository", lambda: graph)
    monkeypatch.setattr(
        worker,
        "_projection_settings",
        lambda: SimpleNamespace(
            projection_batch_size=37,
            projection_lease_seconds=41,
            graph_overall_timeout_ms=250,
        ),
    )
    monkeypatch.setattr(
        worker,
        "record_projection_private_mapping_checksum",
        lambda **kwargs: calls.append(("private_fence", kwargs["checksum"])),
        raising=False,
    )
    monkeypatch.setattr(
        worker,
        "_expected_manifest",
        lambda _bundle, checksum: SimpleNamespace(
            graph_checksum="b" * 64,
            private_mapping_checksum=checksum,
        ),
    )

    def publish(**kwargs):
        assert kwargs["expected_generation_key"] == generation_key
        assert kwargs["expected_graph_checksum"] == "b" * 64
        assert kwargs["expected_private_mapping_checksum"] == private_checksum
        return SimpleNamespace(published=True, failure_code=None)

    monkeypatch.setattr(
        worker,
        "publish_projection_ready_compare_and_set",
        publish,
    )

    outcome = worker.project_generation(
        projection_id=projection_id, lease_owner="worker-a"
    )

    assert outcome.ready is True
    assert calls == [
        "load_bundle",
        "load_private",
        "persist_private",
        ("private_fence", private_checksum),
        "stage",
        "graph_ready",
    ]


def test_project_generation_redacts_partial_write_failures(monkeypatch):
    projection_id = uuid4()
    monkeypatch.setattr(
        worker,
        "claim_projection_lease",
        lambda **_kwargs: _lease(projection_id),
    )
    monkeypatch.setattr(
        worker,
        "_postgres_repository",
        lambda: SimpleNamespace(
            load_projection_bundle=lambda **_kwargs: (_ for _ in ()).throw(
                RuntimeError("secret payload")
            )
        ),
    )
    failed = []
    monkeypatch.setattr(
        worker,
        "mark_projection_failed",
        lambda **kwargs: failed.append(kwargs["failure_code"]),
    )

    outcome = worker.project_generation(
        projection_id=projection_id, lease_owner="worker-a"
    )

    assert outcome.ready is False and outcome.failure_code == "write_failed"
    assert "secret" not in repr(outcome)
    assert len(failed) == 1


def test_project_generation_propagates_redacted_transient_for_celery_retry(
    monkeypatch,
):
    projection_id = uuid4()
    monkeypatch.setattr(
        worker,
        "_projection_settings",
        lambda: SimpleNamespace(
            projection_batch_size=10,
            projection_lease_seconds=30,
            graph_overall_timeout_ms=250,
        ),
    )
    monkeypatch.setattr(
        worker,
        "claim_projection_lease",
        lambda **_kwargs: _lease(projection_id),
    )
    monkeypatch.setattr(
        worker,
        "_postgres_repository",
        lambda: SimpleNamespace(
            load_projection_bundle=lambda **_kwargs: (_ for _ in ()).throw(
                TimeoutError("credential-bearing backend detail")
            )
        ),
    )

    with pytest.raises(TimeoutError, match="projection_backend_transient") as captured:
        worker.project_generation(
            projection_id=projection_id,
            lease_owner="worker-a",
        )

    assert "credential" not in repr(captured.value)


def test_project_generation_retries_redacted_memgraph_driver_failures(monkeypatch):
    projection_id = uuid4()
    monkeypatch.setattr(
        worker,
        "_projection_settings",
        lambda: SimpleNamespace(
            projection_batch_size=10,
            projection_lease_seconds=30,
            graph_overall_timeout_ms=250,
        ),
    )
    monkeypatch.setattr(
        worker,
        "claim_projection_lease",
        lambda **_kwargs: _lease(projection_id),
    )
    monkeypatch.setattr(
        worker,
        "_postgres_repository",
        lambda: SimpleNamespace(
            load_projection_bundle=lambda **_kwargs: object(),
            load_private_chunk_references=lambda **_kwargs: (),
            persist_chunk_references=lambda **_kwargs: "c" * 64,
        ),
    )
    monkeypatch.setattr(
        worker,
        "record_projection_private_mapping_checksum",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        worker,
        "_memgraph_repository",
        lambda: (_ for _ in ()).throw(MemgraphDriverError("memgraph_write_failed")),
    )

    with pytest.raises(TimeoutError, match="projection_backend_transient"):
        worker.project_generation(
            projection_id=projection_id,
            lease_owner="worker-a",
        )


def test_project_generation_does_not_swallow_transient_failure_recording(monkeypatch):
    projection_id = uuid4()
    monkeypatch.setattr(
        worker,
        "claim_projection_lease",
        lambda **_kwargs: _lease(projection_id),
    )
    monkeypatch.setattr(
        worker,
        "_postgres_repository",
        lambda: SimpleNamespace(
            load_projection_bundle=lambda **_kwargs: (_ for _ in ()).throw(
                ValueError("invalid source")
            )
        ),
    )
    monkeypatch.setattr(
        worker,
        "mark_projection_failed",
        lambda **_kwargs: (_ for _ in ()).throw(
            ConnectionError("database credential detail")
        ),
    )

    with pytest.raises(TimeoutError, match="projection_backend_transient") as captured:
        worker.project_generation(
            projection_id=projection_id,
            lease_owner="worker-a",
        )

    assert "credential" not in repr(captured.value)
