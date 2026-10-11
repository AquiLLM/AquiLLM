"""Real optional-query validation cannot refresh original pins or source scope."""
import json

import pytest
from channels.db import database_sync_to_async

from apps.collections.models import CollectionPermission
from apps.documents.models import RawTextDocument
from apps.ingestion.tests.output_support import upload as upload
from lib.embeddings.tests.test_document_provider import transport as transport
from lib.llm.answer_publication import LIMITED_ANSWER

from .rag_query_treatment_support import (
    EXPANSION, QUESTION, REWRITE, controls, mutate, inherit_from_parent,
    query_corpus as query_corpus, close_query_worker as close_query_worker,
)
from .rag_query_treatment_transport import QueryProvider, score_transport as score_transport
from .rag_query_treatment_observation import install_observers, new_consumer, run

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("followup", [False, True])
@pytest.mark.parametrize("question,rewrite", [
    (QUESTION, "Describe BRX-8 response"),
    ("Explain BRX-9 response without delay", "Describe BRX-9 response with delay"),
    ("Explain BRX-9 response at 5 ms", "Describe BRX-9 response at 6 ms"),
    ("Explain BRX-9 response at 5 ms", "Describe BRX-9 response at 5 kg"),
    ('Explain BRX-9 response in "Red Giants"', 'Describe BRX-9 response in "Blue Dwarfs"'),
])
async def test_real_rewrite_rejects_changed_protected_constraints(
    query_corpus, score_transport, monkeypatch, followup, question, rewrite,
):
    controls(monkeypatch, "rewrite", followup)
    audit = install_observers(monkeypatch)
    provider = QueryProvider(rewrite)
    result = await run(query_corpus, provider, new_consumer(query_corpus, question))
    assert [call[0] for call in audit.searches] == [question]
    assert [row.kind for row in audit.prepared[0][2]] == ["original"]
    assert result.runtime.budget.actions_used == 1
    assert result.runtime.budget.planner_calls == len(provider.plans) == 1
    assert len(provider.answers) == 1
    assert all(packet.query == question for packet in audit.packets)
    assert result.answer._validated_answer.outcome == "partial"


@pytest.mark.asyncio
@pytest.mark.parametrize("followup", [False, True])
@pytest.mark.parametrize("change", ["detach", "repin", "revoke", "generation"])
async def test_mutation_after_first_real_search_cannot_refresh_optional_authority(
    query_corpus, score_transport, monkeypatch, followup, change,
):
    controls(monkeypatch, "vocabulary", followup)
    audit = install_observers(monkeypatch, after_first=lambda: mutate(query_corpus, change))
    provider = QueryProvider(REWRITE)
    result = await run(query_corpus, provider, new_consumer(query_corpus))
    assert audit.prepared and len(audit.prepared[0][2]) == 2
    assert audit.searches[0][0] == QUESTION and audit.searches[0][4]["result"]
    assert audit.mutations == [(True, True)], "original optional budget must remain available after mutation"
    assert result.runtime._query_vocabulary.version_keys == (query_corpus.version.version_key,)
    assert all(call[1] is result.runtime and call[2] is result.turn.budget
               and call[3] is result.runtime.authorization for call in audit.searches)
    assert not provider.requests
    assert result.stored == LIMITED_ANSWER
    assert all(not call[4].get("result") for call in audit.searches[1:])
    assert result.runtime.budget.actions_used <= 2
    assert result.runtime.budget.planner_calls == 0
    if change != "generation":
        assert audit.validation_errors
        assert all(bound.runtime is result.runtime and budget is result.turn.budget
                   for bound, budget, _ in audit.validation_errors)
    events = result.trace.events
    assert len([e for e in events if e["event"] == "query_proposals"]) == 1
    assert not [e for e in events if e["event"] == "sdk_start"]
    # Retain the genuine first result/event; do not rewrite history after mutation.
    assert any(e["event"] == "action_start" for e in events)
    assert "BRX-9" in json.dumps(audit.searches[0][4])


@pytest.mark.asyncio
@pytest.mark.parametrize("followup", [False, True])
@pytest.mark.parametrize("historical_title,requested,rewrite,admitted", [
    ("Red", "Red", "Explain Red", True),
    ("Red", "Red: Giants", "Explain Red: Giants", True),
    ("Red: Giants", "Red: Giants", "Explain Red: Giants", True),
    ("Red: Giants", '"Red: Giants"', 'Explain "Red: Giants"', True),
    ("Red: Giants", "Red: Giants", "Explain Red", False),
])
async def test_actual_history_name_boundary_reaches_query_dispatch(
    query_corpus, score_transport, monkeypatch, followup, historical_title, requested, rewrite, admitted,
):
    await database_sync_to_async(lambda: RawTextDocument.objects.filter(
        id__in=[doc.id for doc in query_corpus.docs]).update(title=historical_title))()
    controls(monkeypatch, "original", False)
    consumer = new_consumer(query_corpus)
    await run(query_corpus, QueryProvider(REWRITE), consumer)
    prior_tool = next(message for message in reversed(consumer.convo.messages)
                      if message.role == "tool")
    historical = prior_tool.result_dict["result"][0]
    assert historical["title"] == historical_title
    controls(monkeypatch, "rewrite", followup)
    question = "Summarize " + requested
    audit = install_observers(monkeypatch)
    result = await run(query_corpus, QueryProvider(rewrite),
        new_consumer(query_corpus, question, history=consumer.convo.messages))
    assert [call[0] for call in audit.searches] == [question] + ([rewrite] if admitted else [])
    anchors = audit.prepared[0][1]["anchors"]
    expected = (str(historical["doc_id"]),) if requested.strip('"') == historical_title else ()
    assert anchors.document_ids == expected
    assert not anchors.unresolved_references
    assert all(packet.query == question for packet in audit.packets)
    assert result.runtime.budget.actions_used == 1 + int(admitted)


@pytest.mark.asyncio
@pytest.mark.parametrize("followup", [False, True])
@pytest.mark.parametrize("revoke_parent", [False, True])
async def test_explicit_ancestor_vocabulary_requires_current_parent_permission(
    query_corpus, score_transport, monkeypatch, followup, revoke_parent,
):
    corpus = query_corpus
    parent = await database_sync_to_async(inherit_from_parent)(corpus)
    controls(monkeypatch, "vocabulary", followup)
    def revoke():
        CollectionPermission.objects.filter(user=corpus.user, collection=parent).delete()
        assert CollectionPermission.objects.filter(user=corpus.user, collection=corpus.collection).exists()
    audit = install_observers(monkeypatch, after_first=revoke if revoke_parent else None)
    provider = QueryProvider(REWRITE)
    result = await run(corpus, provider, new_consumer(corpus))
    snapshot = result.runtime._query_vocabulary
    assert snapshot.version_keys == (corpus.version.version_key,)
    assert snapshot.binding.paths == ((corpus.collection.pk, (corpus.collection.pk, parent.pk)),)
    if revoke_parent:
        assert audit.mutations == [(True, True)] and audit.validation_errors
        assert [call[0] for call in audit.searches] == [QUESTION]
        assert result.stored == LIMITED_ANSWER and not provider.requests
    else:
        assert [call[0] for call in audit.searches] == [QUESTION, QUESTION + ' "' + EXPANSION + '"']
        assert result.answer._validated_answer.outcome == "partial" and len(provider.answers) == 1
