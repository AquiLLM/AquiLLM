"""Embedding outages fence dense deliveries while published lexical source survives."""
from unittest.mock import Mock, patch
from types import SimpleNamespace

import httpx
import pytest
from celery.exceptions import Retry

from apps.documents.models import TextChunk
from apps.documents.models.embedding_index import EmbeddingGeneration
from apps.documents.services import embedding_index_store as store
from apps.documents.tasks import embedding_index as task
from lib.embeddings import document_provider
from lib.embeddings.errors import EmbeddingUpstreamUnavailableError
from lib.embeddings.utils import EmbeddingContractError
from ._chunk_graph_lifecycle_support import configure_chunking_runtime
from .test_chunk_publication_recovery import document_factory
from .embedding_outage_support import (
    dense_source, legacy, transport, schedule, due, transport_failure, source_state,
    assert_dense_failed, assert_dense_ready, current_lease,
)

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize("failure", [EmbeddingUpstreamUnavailableError("offline"), EmbeddingContractError("malformed"), TypeError("bug")])
def test_document_batch_error_does_not_fan_out(document_factory, monkeypatch, failure):
    """The retained callable legacy helper still sends one batch and never fans out."""
    from apps.documents.tasks.chunking import create_chunks
    chunking = configure_chunking_runtime(monkeypatch)
    with patch.object(create_chunks, "delay"):
        document = document_factory("Synthetic source " * 200)
    batch, single = Mock(side_effect=failure), Mock()
    monkeypatch.setattr(chunking, "get_embeddings", batch)
    monkeypatch.setattr(TextChunk, "get_chunk_embedding", single)
    with pytest.raises(type(failure)):
        chunks, image = chunking._prepared_chunks(document)
        chunking._embed_chunks(document, chunks, image)
    assert batch.call_count == 1
    single.assert_not_called()


@pytest.mark.parametrize("failure,expected,kind", [
    (httpx.ConnectError("offline"), Retry, ""),
    (httpx.UnsupportedProtocol("bad configuration"), EmbeddingContractError, "contract"),
    (TypeError("provider bug"), TypeError, "unexpected"),
])
def test_actual_dense_batch_errors_never_fan_out(dense_source, monkeypatch, failure, expected, kind):
    key = schedule(dense_source)
    before = source_state(dense_source)
    dispatch, _ = transport_failure(monkeypatch, failure)
    with patch.object(task.build_embedding_generation, "retry", side_effect=Retry()) as retry:
        with pytest.raises(expected):
            task.build_embedding_generation.run(key)
    assert dispatch.call_count == 1
    assert retry.call_count == (1 if expected is Retry else 0)
    row = EmbeddingGeneration.objects.get(pk=key)
    assert row.failure_kind == kind and row.deliveries == 1
    assert row.state == ("building" if expected is Retry else "failed")
    assert not row.embeddings.exists()
    assert source_state(dense_source) == before


def test_contract_terminal_generation_explicit_rebuild_and_stale_delivery(dense_source, monkeypatch):
    key = schedule(dense_source)
    before = source_state(dense_source)
    dispatch, healthy = transport_failure(monkeypatch,
        lambda **kwargs: SimpleNamespace(model=dense_source[2].model, data=[]))
    with patch.object(task.build_embedding_generation, "retry") as retry:
        with pytest.raises(EmbeddingContractError):
            task.build_embedding_generation.run(key)
    retry.assert_not_called()
    assert dispatch.call_count == 1
    assert_dense_failed(key, "contract")
    due(key)
    assert task.build_embedding_generation.run(key) == "not_claimed"
    assert dispatch.call_count == 1
    replacement = schedule(dense_source)
    assert replacement != key
    assert task.build_embedding_generation.run(key) == "not_claimed"
    assert dispatch.call_count == 1
    monkeypatch.setattr(document_provider, "_client", healthy)
    assert task.build_embedding_generation.run(replacement) == "activated"
    assert_dense_ready(replacement, dense_source)
    assert EmbeddingGeneration.objects.get(pk=key).failure_kind == "contract"
    assert source_state(dense_source) == before


def test_transient_exhaustion_retains_history_and_explicitly_recovers(dense_source, monkeypatch):
    key = schedule(dense_source)
    before = source_state(dense_source)
    dispatch, healthy = transport_failure(monkeypatch, httpx.ConnectError("offline"))
    with patch.object(task.build_embedding_generation, "retry", side_effect=Retry()) as retry:
        for number in range(3):
            if number:
                due(key)
            if number < 2:
                with pytest.raises(Retry):
                    task.build_embedding_generation.run(key)
                assert retry.call_args.kwargs["countdown"] == 30 * 2 ** number
            else:
                assert task.build_embedding_generation.run(key) == "failed"
    assert dispatch.call_count == 3 and retry.call_count == 2
    assert_dense_failed(key, "upstream", deliveries=3)
    assert task.build_embedding_generation.run(key) == "not_claimed"
    assert dispatch.call_count == 3
    replacement = schedule(dense_source)
    assert replacement != key
    monkeypatch.setattr(document_provider, "_client", healthy)
    assert task.build_embedding_generation.run(replacement) == "activated"
    assert_dense_ready(replacement, dense_source)
    assert source_state(dense_source) == before


def test_old_failure_lease_cannot_terminal_block_current_delivery(dense_source):
    key = schedule(dense_source)
    stale = store.claim_embedding_delivery(key, using="default")
    due(key)
    current = store.claim_embedding_delivery(key, using="default")
    assert current.lease_token != stale.lease_token
    assert not store.fail_embedding_delivery(stale, kind="contract", using="default")
    row = EmbeddingGeneration.objects.get(pk=key)
    assert (row.state, row.failure_kind, row.lease_token) == ("building", "", current.lease_token)


def test_same_source_save_does_not_clear_terminal_dense_state(dense_source, monkeypatch):
    from apps.documents.tasks.chunking import create_chunks
    key = schedule(dense_source)
    dispatch, _ = transport_failure(monkeypatch, httpx.UnsupportedProtocol("synthetic configuration"))
    with pytest.raises(EmbeddingContractError):
        task.build_embedding_generation.run(key)
    before = source_state(dense_source)
    with patch.object(create_chunks, "delay") as structural_queue:
        dense_source[0].save()
    structural_queue.assert_not_called()
    assert_dense_failed(key, "contract")
    assert dispatch.call_count == 1 and source_state(dense_source) == before


def test_document_worker_transient_retry_chain_is_three_durable_deliveries(dense_source, monkeypatch):
    key = schedule(dense_source)
    dispatch, _ = transport_failure(monkeypatch, httpx.ConnectError("offline"))
    with patch.object(task.build_embedding_generation, "retry", side_effect=Retry()) as retry:
        for number in range(3):
            due(key)
            # Every delivery may be a fresh explicit-resume Celery message.
            task.build_embedding_generation.push_request(retries=0)
            try:
                if number < 2:
                    with pytest.raises(Retry):
                        task.build_embedding_generation.run(key)
                    assert retry.call_args.kwargs["countdown"] == 30 * 2 ** number
                else:
                    assert task.build_embedding_generation.run(key) == "failed"
            finally:
                task.build_embedding_generation.pop_request()
    assert dispatch.call_count == 3 and retry.call_count == 2
    assert_dense_failed(key, "upstream", deliveries=3)


@pytest.mark.parametrize("change", ["reset", "lease", "source"])
def test_inflight_old_failure_cannot_block_new_generation(dense_source, monkeypatch, change):
    from apps.documents.services.representation_history import reference
    from apps.documents.services.representation_snapshot_build import build_canonical_snapshot_representation
    from apps.documents.services.representation_store import freeze_active_representations
    from apps.documents.tasks.chunking import create_chunks
    key = schedule(dense_source)
    replacement = []

    def fail_after_change(**kwargs):
        lease = current_lease(key)
        if change == "lease":
            due(key)
            newer = store.claim_embedding_delivery(key, using="default")
            assert newer.lease_token != lease.lease_token
            replacement.append((key, dense_source))
        elif change == "reset":
            assert not store.fail_embedding_delivery(lease, kind="contract", using="default")
            replacement.append((schedule(dense_source), dense_source))
            assert replacement[0][0] != key
        else:
            document = dense_source[0]
            document.full_text = "New source while embedding"
            with patch.object(create_chunks, "delay"):
                document.save()
            assert build_canonical_snapshot_representation(reference(document), using="default") == "activated"
            document.refresh_from_db()
            selected = freeze_active_representations((reference(document),), using="default").selections[0]
            assert selected.generation_key != dense_source[1].generation_key
            source = document, selected, dense_source[2]
            replacement.append((schedule(source), source))
        raise httpx.UnsupportedProtocol("old delivery failure")

    dispatch, healthy = transport_failure(monkeypatch, fail_after_change)
    with pytest.raises(EmbeddingContractError):
        task.build_embedding_generation.run(key)
    assert dispatch.call_count == 1
    new_key, source = replacement[0]
    row = EmbeddingGeneration.objects.get(pk=new_key)
    assert (row.state, row.failure_kind) == ("building", "")
    monkeypatch.setattr(document_provider, "_client", healthy)
    due(new_key)
    assert task.build_embedding_generation.run(new_key) == "activated"
    assert_dense_ready(new_key, source)


def test_transient_peer_does_not_clear_terminal_failure(dense_source, monkeypatch):
    key = schedule(dense_source)
    def transient_after_peer(**kwargs):
        assert not store.fail_embedding_delivery(current_lease(key), kind="contract", using="default")
        raise httpx.ConnectError("offline")
    dispatch, _ = transport_failure(monkeypatch, transient_after_peer)
    with patch.object(task.build_embedding_generation, "retry") as retry:
        assert task.build_embedding_generation.run(key) == "failed"
    retry.assert_not_called()
    assert dispatch.call_count == 1
    assert_dense_failed(key, "contract")


def test_sdk_wrapped_permanent_protocol_failure_never_retries_worker(dense_source, monkeypatch):
    import openai
    key = schedule(dense_source)
    failure = openai.APIConnectionError(request=httpx.Request("POST", "http://fake-models:8000/v1/embeddings"))
    failure.__cause__ = httpx.UnsupportedProtocol("synthetic configuration")
    dispatch, _ = transport_failure(monkeypatch, failure)
    with patch.object(task.build_embedding_generation, "retry") as retry:
        with pytest.raises(EmbeddingContractError):
            task.build_embedding_generation.run(key)
    retry.assert_not_called()
    assert dispatch.call_count == 1
    assert_dense_failed(key, "contract")
