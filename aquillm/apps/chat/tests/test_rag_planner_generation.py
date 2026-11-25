"""Planner requests exercise the real provider up to its external SDK call."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

from apps.chat.services.rag_coverage import assess_coverage, validate_assessment
from lib.llm.providers.openai import OpenAIInterface
from lib.retrieval.evidence import SourceEvidence
from lib.retrieval.turn_budget import TurnBudget, TurnLimits

SOURCES = (
    SourceEvidence(1, "doc-a", 0, "revision-a", "Yield measurement methods."),
    SourceEvidence(2, "doc-b", 0, "revision-b", "Other measurements."),
)


def payload(action=None):
    return {
        "requested_aspects": ["yield"],
        "support": [],
        "unresolved_aspects": ["yield"],
        "next_action": action,
    }


def action(kind="vector", *, document_id=None, chunk_id=None):
    return {
        "kind": kind,
        "query": "" if kind == "adjacent" else "measured yield",
        "document_id": document_id,
        "chunk_id": chunk_id,
        "aspect": "yield",
    }


class Completions:
    def __init__(self, text, *, finish_reason="stop", error=None):
        self.text, self.finish_reason, self.error = text, finish_reason, error
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.text, tool_calls=[]),
                    finish_reason=self.finish_reason,
                )
            ],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50),
        )


async def assess(
    completions,
    *,
    base_url="http://vllm:8000/v1",
    sources=SOURCES,
    allowed_documents=("doc-a", "doc-b"),
    budget=None,
):
    llm = OpenAIInterface(
        SimpleNamespace(
            base_url=base_url, chat=SimpleNamespace(completions=completions)
        ),
        model="qwen3.6:27b",
    )
    return await assess_coverage(
        "What was the measured yield?",
        sources,
        None,
        llm=llm,
        budget=budget or TurnBudget(TurnLimits()),
        allowed_documents=allowed_documents,
    )


@pytest.fixture(autouse=True)
def bounded_context(monkeypatch):
    monkeypatch.setenv("OPENAI_CONTEXT_LIMIT", "131072")
    monkeypatch.setenv("OPENAI_COMPAT_ENABLE_THINKING", "1")


@pytest.mark.asyncio
async def test_planner_disables_inherited_thinking_at_transport_boundary():
    completions = Completions(json.dumps(payload(action())))
    result = await assess(completions)
    assert result.next_action.query == "measured yield"
    assert completions.calls[0]["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": False
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "candidate,valid",
    [
        (action(), True),
        (action("document", document_id="doc-a"), True),
        (action("adjacent", document_id="doc-b", chunk_id=2), True),
        (action("document"), False),
        (action("document", document_id="outside"), False),
        (action("vector", document_id="doc-a"), False),
        (action("adjacent", document_id="doc-a", chunk_id=2), False),
    ],
)
async def test_schema_enforces_kind_specific_source_coordinates(candidate, valid):
    completions = Completions(json.dumps(payload(action())))
    await assess(completions)
    schema = (
        completions.calls[0]
        .get("response_format", {})
        .get("json_schema", {})
        .get("schema")
    )
    assert schema is not None, "planner schema must reach the provider transport"
    Draft202012Validator.check_schema(schema)
    assert Draft202012Validator(schema).is_valid(payload(candidate)) is valid


@pytest.mark.asyncio
async def test_unknown_compatible_endpoint_uses_prompt_contract_without_schema_kwarg():
    completions = Completions(json.dumps(payload(action())))
    await assess(completions, base_url="https://compatible.example/v1")
    request = completions.calls[0]
    assert "response_format" not in request
    prompt = json.loads(request["messages"][1]["content"])
    assert prompt.get("action_coordinates") == {
        "document_ids": ["doc-a", "doc-b"],
        "adjacent_sources": [
            {"document_id": "doc-a", "chunk_id": 1},
            {"document_id": "doc-b", "chunk_id": 2},
        ],
    }


@pytest.mark.asyncio
async def test_schema_only_lists_visible_authorized_document_options():
    completions = Completions(json.dumps(payload(action())))
    await assess(completions, allowed_documents={"doc-a", "hidden-document"})
    request = completions.calls[0]
    prompt = json.loads(request["messages"][1]["content"])
    assert prompt["action_coordinates"] == {
        "document_ids": ["doc-a"],
        "adjacent_sources": [{"document_id": "doc-a", "chunk_id": 1}],
    }
    schema = request["response_format"]["json_schema"]["schema"]
    assert not Draft202012Validator(schema).is_valid(
        payload(action("document", document_id="hidden-document"))
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text,finish_reason,reason",
    [
        (None, "length", "output_limit"),
        (json.dumps(payload(action())), "length", "output_limit"),
        (None, "stop", "empty_response"),
        (" ", "stop", "empty_response"),
        ("sensitive invalid response", "stop", "invalid_json"),
        ('{"support": []}', "stop", "invalid_schema"),
        (json.dumps(payload(action("document"))), "stop", "invalid_action"),
    ],
)
async def test_planner_failure_has_safe_specific_category(text, finish_reason, reason):
    completions = Completions(text, finish_reason=finish_reason)
    with pytest.raises(ValueError) as caught:
        await assess(completions)
    assert getattr(caught.value, "reason", None) == reason
    assert str(caught.value) == reason


@pytest.mark.asyncio
async def test_provider_failure_does_not_expose_exception_payload():
    with pytest.raises(ValueError) as caught:
        await assess(Completions(None, error=RuntimeError("secret provider body")))
    assert getattr(caught.value, "reason", None) == "provider_error"
    assert str(caught.value) == "provider_error"


@pytest.mark.asyncio
async def test_cancellation_is_never_converted_to_planner_failure():
    with pytest.raises(asyncio.CancelledError):
        await assess(Completions(None, error=asyncio.CancelledError()))


@pytest.mark.parametrize("raw_action", [{}, False, [], "vector"])
def test_invalid_empty_action_is_not_coerced_to_no_action(raw_action):
    with pytest.raises(ValueError):
        validate_assessment(payload(raw_action), SOURCES, {"doc-a", "doc-b"})


@pytest.mark.asyncio
async def test_empty_scope_offers_vector_without_guessing_document_ids():
    completions = Completions(json.dumps(payload(action())))
    await assess(completions, allowed_documents=())
    schema = completions.calls[0]["response_format"]["json_schema"]["schema"]
    validator = Draft202012Validator(schema)
    assert validator.is_valid(payload(action()))
    assert not validator.is_valid(payload(action("document", document_id="doc-a")))
    assert not validator.is_valid(
        payload(action("adjacent", document_id="doc-a", chunk_id=1))
    )


@pytest.mark.asyncio
async def test_whole_view_omission_excludes_its_adjacent_and_document_options():
    completions = Completions(json.dumps(payload(action())))
    large = SourceEvidence(3, "omitted", 0, "revision-c", "x" * 16001)
    await assess(
        completions, sources=(*SOURCES, large), allowed_documents={"doc-a", "omitted"}
    )
    prompt = json.loads(completions.calls[0]["messages"][1]["content"])
    assert prompt["omitted_sources"] == 1
    assert prompt["action_coordinates"]["document_ids"] == ["doc-a"]
    assert prompt["evidence"][0]["text"] == SOURCES[0].text


@pytest.mark.asyncio
async def test_timeout_is_a_safe_planner_category():
    class PendingCompletions:
        async def create(self, **kwargs):
            await asyncio.Future()

    with pytest.raises(ValueError) as caught:
        await assess(
            PendingCompletions(), budget=TurnBudget(TurnLimits(planner_call_ms=1))
        )
    assert getattr(caught.value, "reason", None) == "timeout"


@pytest.mark.asyncio
async def test_full_schema_context_overflow_stops_before_transport(monkeypatch):
    monkeypatch.setenv("OPENAI_CONTEXT_LIMIT", "128")
    completions = Completions(json.dumps(payload(action())))
    with pytest.raises(ValueError) as caught:
        await assess(completions)
    assert getattr(caught.value, "reason", None) == "context_budget"
    assert completions.calls == []


@pytest.mark.parametrize(
    "override", [{"source_fingerprint": "stale"}, {"end": 1000}, {"chunk_id": True}]
)
def test_generated_support_still_needs_exact_revision_and_source_span(override):
    response = payload()
    response["support"] = [
        {
            "aspect": "yield",
            "chunk_id": 1,
            "source_fingerprint": "revision-a",
            "start": 0,
            "end": 5,
            **override,
        }
    ]
    with pytest.raises(ValueError):
        validate_assessment(response, SOURCES, {"doc-a", "doc-b"})
