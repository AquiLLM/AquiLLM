"""Real validated coordinates for auxiliary followups; no source authority mocks."""
from apps.chat.services.rag_retrieval import _verified_row_coordinates
from apps.chat.tests.test_rag_source_continuity import _history
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import AssistantMessage, ToolMessage, UserMessage

FOLLOWUPS = ("Does it support the claim?", "Can it explain the result?")
PAIRED_LOCAL = (
    "Does the instrument support the claim? Does it support the claim?",
    "Can cooling the sample explain the result? Can it explain the result?",
)
EXISTING_LOCAL = (
    "Explain Lumen's mechanism and its limitations.",
    "How did activating the Vela coating change thermal drift? How did it change spectral jitter?",
    "Explain what each paper is about? What overlaps between them?",
)


def single_history(*, compact=False):
    history = _history(compact=compact)
    history.messages[-1].content = "Displayed result [doc:paper-z chunk:7]."
    identities = {_verified_row_coordinates(row) for row in history.messages[1].result_dict["result"]}
    assert identities == {
        (7, "paper-z", 0, "[doc:paper-z chunk:7]"),
        (8, "paper-a", 0, "[doc:paper-a chunk:8]"),
    }
    assert history.messages[-1].content == "Displayed result [doc:paper-z chunk:7]."
    return history


async def persisted_history(question):
    from apps.chat.tests.persisted_direct_test_support import persisted_direct_case

    declared = [
        dict(doc_id="prior-a", chunk_id=1, title="Displayed source",
             text="The original source states its measured result.",
             citation="[doc:prior-a chunk:1]"),
        dict(doc_id="prior-b", chunk_id=2, title="Other selected source",
             text="An unrelated source states a different result.",
             citation="[doc:prior-b chunk:2]"),
    ]
    case = await persisted_direct_case(declared)
    rows = case.rows({"result": declared})["result"]
    history = Conversation(system="Use current selected evidence.", messages=[
        UserMessage(content="Read the selected sources."),
        ToolMessage(tool_name="vector_search", for_whom="assistant",
                    content="{}", result_dict={"result": rows}),
        AssistantMessage(content="Displayed result " + rows[0]["citation"] + ".",
                         stop_reason="end_turn"),
        UserMessage(content=question),
    ])
    assert _verified_row_coordinates(rows[0]) == (
        case.chunks[("prior-a", 1)].pk, str(case.documents["prior-a"].id), 0, rows[0]["citation"])
    assert _verified_row_coordinates(rows[1]) is not None
    return case, history, rows[0]


def controls(monkeypatch, settings):
    for name, value in {
        "RAG_DIRECT_ENABLED": "1", "RAG_EVIDENCE_TEXT_MODE": "source",
        "RAG_RERANK_TEXT_MODE": "legacy", "RAG_DOCUMENT_CAPACITY_MODE": "legacy",
        "RAG_FOLLOWUP_EVIDENCE_ENABLED": "1", "RAG_ITERATIVE_RETRIEVAL_ENABLED": "0",
        "RAG_QUERY_REWRITE_ENABLED": "0", "RAG_VOCABULARY_QUERY_ENABLED": "0",
        "RAG_EVIDENCE_SELECTION_MODE": "legacy",
    }.items():
        monkeypatch.setenv(name, value)
    settings.KG_OVERLAY_ENABLED = False
    settings.RAG_CACHE_ENABLED = False
    settings.DOCUMENT_EMBEDDING_SPACE_KEY = None
