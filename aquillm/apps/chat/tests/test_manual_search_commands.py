"""Manual search syntax and deterministic document selection."""

from types import SimpleNamespace

import pytest

from apps.chat.services.manual_search_commands import (
    ManualSearchError,
    parse_manual_search,
    resolve_search_document,
    resolve_search_query,
)
from lib.llm.types.messages import ToolMessage, UserMessage

DOC_A = "11111111-1111-4111-8111-111111111111"
DOC_B = "22222222-2222-4222-8222-222222222222"


@pytest.mark.parametrize(
    "text,command,document,query",
    [
        (
            "/collection flat field calibration",
            "collection",
            None,
            "flat field calibration",
        ),
        (
            " /SEARCH [Paper A] what is the method?",
            "search",
            "Paper A",
            "what is the method?",
        ),
        ('/search "Paper A" calibration', "search", "Paper A", "calibration"),
        (f"/search {DOC_A} calibration", "search", DOC_A, "calibration"),
        ("/search calibration", "search", None, "calibration"),
        ("/search", "search", None, ""),
        ("/collection", "collection", None, ""),
    ],
)
def test_parse_manual_search(text, command, document, query):
    result = parse_manual_search(text)
    assert (result.command, result.document, result.query) == (command, document, query)


@pytest.mark.parametrize(
    "text", ["/searching foo", "please /search foo", "hello", "/collections"]
)
def test_normal_messages_are_not_commands(text):
    assert parse_manual_search(text) is None


@pytest.mark.parametrize(
    "text", ["/search [Paper A", '/search "Paper A', "/search [] query"]
)
def test_invalid_selector_prompts_for_correct_syntax(text):
    with pytest.raises(ManualSearchError, match="/search"):
        parse_manual_search(text)


def test_bare_command_reuses_previous_question_without_command_markup():
    prior = [
        UserMessage(content="older question"),
        UserMessage(content="/collection calibration"),
    ]
    assert (
        resolve_search_query(parse_manual_search("/search [Paper A]"), prior)
        == "calibration"
    )
    assert (
        resolve_search_query(parse_manual_search("/collection new topic"), prior)
        == "new topic"
    )


def test_command_without_query_or_history_prompts_instead_of_searching_empty_text():
    with pytest.raises(ManualSearchError, match="question"):
        resolve_search_query(parse_manual_search("/collection"), [])


def _docs():
    return [
        SimpleNamespace(id=DOC_A, title="Paper A"),
        SimpleNamespace(id=DOC_B, title="Paper B"),
    ]


def test_document_title_and_prefix_resolve_to_full_id():
    assert resolve_search_document("paper a", [], _docs()) == DOC_A
    assert resolve_search_document("11111111", [], _docs()) == DOC_A


def test_full_id_can_be_passed_to_existing_tool_for_access_validation():
    assert resolve_search_document(DOC_A, [], []) == DOC_A


def test_omitted_document_reuses_recent_target_in_selected_scope():
    prior = [
        ToolMessage(
            tool_name="search_single_document",
            for_whom="assistant",
            content="evidence",
            arguments={"doc_id": DOC_B},
            result_dict={},
        )
    ]
    assert resolve_search_document(None, prior, _docs()) == DOC_B
    assert resolve_search_document(None, prior, _docs()[:1]) == DOC_A


@pytest.mark.parametrize("selector", [None, "Paper", "Missing paper"])
def test_ambiguous_or_missing_document_never_broadens_to_collection(selector):
    with pytest.raises(ManualSearchError, match="document"):
        resolve_search_document(selector, [], _docs())


def test_duplicate_titles_require_an_id():
    docs = [
        SimpleNamespace(id=DOC_A, title="Paper"),
        SimpleNamespace(id=DOC_B, title="Paper"),
    ]
    with pytest.raises(ManualSearchError, match="document"):
        resolve_search_document("Paper", [], docs)


@pytest.mark.parametrize(
    "argument", ["abcdabcd", "ABCDABCD-1234-4123-8123-123456789ABC"]
)
def test_recent_legacy_tool_arguments_resolve_using_canonical_evidence(argument):
    doc_id = "abcdabcd-1234-4123-8123-123456789abc"
    prior = [
        ToolMessage(
            tool_name="search_single_document",
            for_whom="assistant",
            content="evidence",
            arguments={"doc_id": argument},
            result_dict={"result": [{"doc_id": doc_id}]},
        )
    ]
    docs = [*_docs(), SimpleNamespace(id=doc_id, title="Legacy paper")]
    assert resolve_search_document(None, prior, docs) == doc_id


def test_old_prefix_cannot_select_a_different_document_after_scope_changes():
    original = "abcdabcd-1234-4123-8123-123456789abc"
    other = "abcdabcd-5678-4567-8567-123456789abc"
    prior = [
        ToolMessage(
            tool_name="search_single_document",
            for_whom="assistant",
            content="evidence",
            arguments={"doc_id": "abcdabcd"},
            result_dict={"result": [{"doc_id": original}]},
        )
    ]
    docs = [*_docs(), SimpleNamespace(id=other, title="Different paper")]
    with pytest.raises(ManualSearchError, match="document"):
        resolve_search_document(None, prior, docs)
