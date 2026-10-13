"""Actual SDK and final sinks retain the original query binding after sealing."""
import pytest
from channels.db import database_sync_to_async

from apps.chat.services import rag_pipeline, rag_immutable_images
from apps.chat.services.retrieval_query_publication import query_binding_digest, validate_packet_query_binding
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.ingestion.tests.output_support import upload as upload
from lib.embeddings.tests.test_document_provider import transport as transport
from lib.llm.answer_publication import LIMITED_ANSWER
from lib.llm.turn_context import current_turn
from .rag_query_treatment_support import (
    QUESTION, REWRITE, controls, mutate, query_corpus as query_corpus,
    close_query_worker as close_query_worker,
)
from .rag_query_treatment_transport import QueryProvider, score_transport as score_transport
from .rag_query_treatment_observation import install_observers, new_consumer, run

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("source", [False, True])
async def test_same_original_query_binding_survives_real_sealed_sdk_and_sinks(
    query_corpus, score_transport, monkeypatch, source,
):
    controls(monkeypatch, "vocabulary", source)
    provider, observed = QueryProvider(REWRITE), []
    original = provider.create
    async def create(**kwargs):
        turn = current_turn()
        context, lease = turn.answer_publication, turn.budget._synthesis
        assert lease is not None and context.lease is lease
        assert turn.budget.remaining_ms() == 0
        binding = context.packet._publication_query_binding
        witness = context.packet._publication_query_admission
        assert witness.budget is turn.budget and binding.runtime is witness.runtime
        await database_sync_to_async(validate_packet_query_binding)(context.packet)
        observed.append((binding, witness, lease, query_binding_digest(context.packet)))
        return await original(**kwargs)
    provider.create = create
    result = await run(query_corpus, provider, new_consumer(query_corpus))
    assert len(provider.answers) == len(observed) == 1
    assert result.stored != LIMITED_ANSWER and result.answer._validated_answer.outcome == "partial"
    binding, witness, lease, digest_at_dispatch = observed[0]
    assert current_turn() is None
    assert query_binding_digest(result.trace.delivered_packet) == digest_at_dispatch
    assert result.trace.final["support_packet_identity"]["scope_digest"] == result.answer._validated_answer.scope_digest
    assert result.runtime._query_original_binding is binding
    assert result.runtime._query_admission is witness and result.turn.budget._synthesis is lease


@pytest.mark.asyncio
@pytest.mark.parametrize("source", [False, True])
@pytest.mark.parametrize("boundary", ["last_action", "dispatch", "after_sdk"])
async def test_real_pin_change_at_late_boundary_cannot_publish_old_query_evidence(
    query_corpus, score_transport, monkeypatch, source, boundary,
):
    controls(monkeypatch, "vocabulary", source)
    provider, mutations = QueryProvider(REWRITE), []
    async def change():
        await database_sync_to_async(mutate)(query_corpus, "detach")
        mutations.append(True)
    if boundary == "last_action":
        real = rag_pipeline.synthesize_from_evidence
        async def synthesize(*args, **kwargs):
            await change()
            return await real(*args, **kwargs)
        monkeypatch.setattr(rag_pipeline, "synthesize_from_evidence", synthesize)
    elif boundary == "dispatch":
        real = rag_immutable_images.authorize_media_dispatch
        async def authorize(packet, **kwargs):
            if not kwargs.get("pre_seal", False):
                await change()
            return await real(packet, **kwargs)
        monkeypatch.setattr(rag_immutable_images, "authorize_media_dispatch", authorize)
    else:
        real = provider.create
        async def create(**kwargs):
            response = await real(**kwargs)
            await change()
            return response
        provider.create = create
    result = await run(query_corpus, provider, new_consumer(query_corpus))
    assert mutations == [True]
    assert len(provider.requests) == (1 if boundary == "after_sdk" else 0)
    assert result.stored == LIMITED_ANSWER


@pytest.mark.asyncio
@pytest.mark.parametrize("source", [False, True])
@pytest.mark.parametrize("change", ["packet_binding", "runtime_binding", "scope", "lease"])
async def test_inner_dispatch_rejects_original_binding_or_lease_substitution(
    query_corpus, score_transport, monkeypatch, source, change,
):
    controls(monkeypatch, "vocabulary", source)
    original, mutations = rag_immutable_images.authorize_media_dispatch, []
    async def authorize(packet, **kwargs):
        restore = lambda: None
        if not kwargs.get("pre_seal", False):
            context = current_turn().answer_publication
            if change == "packet_binding":
                retained = packet._publication_query_binding
                del packet._publication_query_binding
                restore = lambda: setattr(packet, "_publication_query_binding", retained)
            elif change == "runtime_binding":
                runtime = packet._publication_query_binding.runtime
                retained = runtime._query_original_binding
                del runtime._query_original_binding
                restore = lambda: setattr(runtime, "_query_original_binding", retained)
            elif change == "scope":
                from dataclasses import replace
                runtime = packet._publication_query_binding.runtime
                retained = runtime._frozen_retrieval_scope
                runtime._frozen_retrieval_scope = replace(retained)
                restore = lambda: setattr(runtime, "_frozen_retrieval_scope", retained)
            else:
                from dataclasses import replace
                retained = context.lease
                context.lease = replace(context.lease)
                restore = lambda: setattr(context, "lease", retained)
            mutations.append(True)
        try:
            return await original(packet, **kwargs)
        finally:
            # Restore only after the actual guard observed the substitution, so
            # the normal limited receipt can traverse the real final sinks.
            restore()
    monkeypatch.setattr(rag_immutable_images, "authorize_media_dispatch", authorize)
    provider = QueryProvider(REWRITE)
    result = await run(query_corpus, provider, new_consumer(query_corpus))
    assert mutations == [True] and not provider.requests
    assert result.stored == LIMITED_ANSWER


@pytest.mark.asyncio
@pytest.mark.parametrize("source", [False, True])
async def test_optional_transport_failure_keeps_current_original_partial_evidence(
    query_corpus, score_transport, monkeypatch, source,
):
    controls(monkeypatch, "vocabulary", source)
    real, attempts = rag_pipeline._run_vector_search, []
    def search(consumer, question, top_k):
        attempts.append(question)
        if question != QUESTION:
            raise OSError("synthetic optional transport outage")
        return real(consumer, question, top_k)
    monkeypatch.setattr(rag_pipeline, "_run_vector_search", search)
    provider = QueryProvider(REWRITE)
    result = await run(query_corpus, provider, new_consumer(query_corpus))
    assert len(attempts) == 2 and len(provider.answers) == 1
    assert result.answer._validated_answer.outcome == "partial" and result.stored != LIMITED_ANSWER


@pytest.mark.asyncio
@pytest.mark.parametrize("source", [False, True])
async def test_original_three_second_optional_timeout_preserves_current_partial_evidence(
    query_corpus, score_transport, monkeypatch, source,
):
    import asyncio
    from apps.chat.services import rag_preservation_turn
    controls(monkeypatch, "vocabulary", source)
    original, cancelled = rag_preservation_turn.execute_action, []
    async def execute(consumer, action, **kwargs):
        if action.query == QUESTION:
            return await original(consumer, action, **kwargs)
        try:
            await asyncio.sleep(3.1)
            raise AssertionError("original three-second action timeout did not cancel")
        except asyncio.CancelledError:
            cancelled.append(True)
            raise
    monkeypatch.setattr(rag_preservation_turn, "execute_action", execute)
    provider = QueryProvider(REWRITE)
    result = await run(query_corpus, provider, new_consumer(query_corpus))
    assert cancelled == [True] and len(provider.answers) == 1
    assert result.answer._validated_answer.outcome == "partial" and result.stored != LIMITED_ANSWER
