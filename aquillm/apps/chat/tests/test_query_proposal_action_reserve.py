"""Only generate optional queries that can fit after mandatory acquisitions."""

import pytest
from channels.db import database_sync_to_async

from apps.chat.services import rag_pipeline
from apps.chat.services.rag_preservation_turn import normal_source_handoff
from apps.chat.services.rag_turn import preservation_turn
from apps.chat.services.tool_wiring.documents import vector_search_tool
from apps.documents.services import chunk_search
from apps.documents.services.source_loading import current_source_runtime
from apps.ingestion.tests.output_support import upload as upload
from lib.embeddings.tests.test_document_provider import transport as transport
from lib.llm.types.messages import ToolMessage

from .rag_query_treatment_observation import install_observers, new_consumer
from .rag_query_treatment_support import (
    EXPANSION,
    QUESTION,
    REWRITE,
    controls,
)
from .rag_query_treatment_support import (
    close_query_worker as close_query_worker,
)
from .rag_query_treatment_support import (
    query_corpus as query_corpus,
)
from .rag_query_treatment_transport import (
    QueryProvider,
)
from .rag_query_treatment_transport import (
    score_transport as score_transport,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.django_db(transaction=True)]


@pytest.mark.parametrize(
    "route,query_limit,tool_limit,prior,arm,alias_match,extras",
    [
        ("direct", 1, 3, False, "rewrite", True, ()),
        ("direct", 3, 1, False, "rewrite", True, ()),
        ("direct", 2, 3, True, "rewrite", True, ()),
        ("direct", 2, 3, False, "rewrite", True, ("rewrite",)),
        ("normal", 2, 3, True, "rewrite", True, ("rewrite",)),
        ("normal", 1, 3, True, "rewrite", True, ()),
        ("direct", 2, 3, False, "interaction", True, ("vocabulary",)),
        ("direct", 2, 3, False, "interaction", False, ("rewrite",)),
        ("normal", 2, 3, True, "interaction", True, ("vocabulary",)),
        ("normal", 2, 3, True, "interaction", False, ("rewrite",)),
        (
            "direct",
            3,
            3,
            False,
            "interaction",
            True,
            ("vocabulary", "rewrite"),
        ),
    ],
    ids=(
        "initial_owns_single_query_slot",
        "initial_owns_single_tool_slot",
        "earlier_action_does_not_mean_initial_acquired",
        "direct_remaining_rewrite_slot",
        "normal_initial_already_acquired",
        "normal_action_limit_exhausted",
        "direct_vocabulary_owns_remaining_slot",
        "direct_no_vocabulary_match_leaves_rewrite_slot",
        "normal_vocabulary_owns_remaining_slot",
        "normal_no_vocabulary_match_leaves_rewrite_slot",
        "all_three_queries_fit",
    ),
)
async def test_rewrite_generation_requires_an_executable_optional_slot(
    query_corpus,
    score_transport,
    monkeypatch,
    route,
    query_limit,
    tool_limit,
    prior,
    arm,
    alias_match,
    extras,
):
    controls(monkeypatch, arm, True)
    monkeypatch.setenv("RAG_DIRECT_MAX_QUERIES", str(query_limit))
    question = QUESTION if alias_match else "Explain the response"
    rewrite = REWRITE if alias_match else "Describe the response"
    consumer = new_consumer(query_corpus, question)
    provider = QueryProvider(rewrite)
    audit = install_observers(monkeypatch)
    searches = []
    real_search = chunk_search.text_chunk_search

    def searched(model, query, *args, **kwargs):
        runtime = current_source_runtime()
        searches.append((query, runtime, runtime.budget.planner_calls))
        return real_search(model, query, *args, **kwargs)

    monkeypatch.setattr(chunk_search, "text_chunk_search", searched)
    async with preservation_turn(consumer, max_func_calls=tool_limit) as runtime:
        budget = runtime.budget
        deadline = budget._deadline
        authority = runtime.authorization
        representation = runtime._frozen_retrieval_scope.representation
        expected_queries = []
        if prior:
            prior_query = question if route == "normal" else "Find baseline response"
            raw = await database_sync_to_async(
                vector_search_tool(consumer.user, consumer.col_ref),
                thread_sensitive=False,
            )(search_string=prior_query, top_k=1)
            assert raw["result"]
            assert budget.actions_used == 1
            expected_queries.append(prior_query)

        if route == "normal":
            working = consumer.convo + [
                ToolMessage(
                    tool_name="vector_search",
                    for_whom="assistant",
                    content="{}",
                    arguments={},
                    result_dict=raw,
                )
            ]
            consumer.convo, outcome = await normal_source_handoff(
                consumer,
                provider.llm,
                working,
                2048,
                None,
            )
            assert outcome == "changed"
        else:
            assert (
                await rag_pipeline.run_direct_rag_turn(
                    consumer,
                    provider.llm,
                    consumer.convo,
                )
                == "handled"
            )
            expected_queries.append(question)

        additions = {
            "vocabulary": question + ' "' + EXPANSION + '"',
            "rewrite": rewrite,
        }
        expected_queries.extend(additions[kind] for kind in extras)
        assert [query for query, _, _ in searches] == expected_queries
        assert all(actual is runtime for _, actual, _ in searches)
        assert budget.actions_used == len(expected_queries)
        assert budget.limits.actions == min(query_limit, tool_limit, 3)
        assert budget._deadline == deadline
        assert runtime.authorization is authority
        assert runtime._frozen_retrieval_scope.representation is representation
        assert len(provider.answers) == 1
        assert consumer.convo[-1]._validated_answer is not None
        assert audit.prepared[0][0] == question
        assert audit.prepared[0][2][0].text == question
        expected_plans = int("rewrite" in extras)
        assert budget.planner_calls == expected_plans
        assert len(provider.plans) == expected_plans
        if not expected_plans:
            assert all(planner_calls == 0 for _, _, planner_calls in searches)
