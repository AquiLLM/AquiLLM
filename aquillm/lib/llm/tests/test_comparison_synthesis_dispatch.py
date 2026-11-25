import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from lib.llm.providers.rag_comparison_dispatch import decode_comparison, decode_note, structured_request, write_comparison
from lib.llm.providers.rag_comparison_writer import ComparisonAnswerPlan
from lib.llm.answer_publication import AnswerPublicationRejected, FinalCitationPolicy
from lib.llm.evidence_guard import EvidenceProtection, protect_evidence
from lib.llm.providers.rag_citation_evidence import bind_citation_sources
from lib.llm.types.messages import UserMessage
from lib.llm.types.response import LLMResponse


def study():
    refs = tuple(f"[doc:{owner} chunk:{i}]" for i, owner in enumerate(("a", "b", "c"), 1))
    texts = ("ARVO field set (Moss et al. 2020)", "ARVO field set (Moss et al. 2020)",
             "ARVO K3 (Moss et al. 2020)")
    rows = [dict(doc_id=owner, chunk_id=i, citation=ref, text=text)
            for i, (owner, ref, text) in enumerate(zip(("a", "b", "c"), refs, texts), 1)]
    policy = FinalCitationPolicy(frozenset(refs), True, "evidence", "scope",
                                 bind_citation_sources(rows, set(refs)))
    plan = ComparisonAnswerPlan("Read the actual sources", tuple((k, (ref,)) for k, ref in zip("ABC", refs)),
                                ("ARVO field set", "ARVO K3"),
                                references=tuple((ref, 0,len(text),text) for ref,text in zip(refs,texts)))
    return plan, policy, refs


def test_invalid_optional_decision_keeps_independently_valid_core():
    plan, policy, refs = study()
    core = {"relation":"retained","statement":"Both editions retained the ARVO field set.","sources":["A","B"]}
    wrong = {"decision":"explain","statement":"The binding changed.","sources":["A","B"]}
    draft, needs_note = decode_comparison(json.dumps({"claims":[core], "naming_decision":wrong}), plan, policy)
    assert draft == f"Both editions retained the ARVO field set {refs[0]}{refs[1]}."
    assert needs_note
    assert decode_comparison('{"claims":[],"naming_decision":{}}',plan,policy) == (None,False)


def test_model_already_provided_named_explanation_needs_no_second_call():
    plan, policy, refs = study()
    core = {"relation":"retained","statement":"The ARVO field set was retained.","sources":["A","B"]}
    note = {"decision":"explain","statement":"Both inventories call it the ARVO field set and the register lists it as ARVO K3.","sources":["A","B","C"]}
    draft, needs_note = decode_comparison(json.dumps({"claims":[core],"naming_decision":note}),plan,policy)
    assert draft and refs[2] in draft and "ARVO K3" in draft
    assert not needs_note
    args = structured_request(plan)
    assert args["thinking_budget"] == 0
    assert "naming_decision" in args["response_format"]["json_schema"]["schema"]["required"]


@pytest.mark.parametrize(("statement", "needs_note"), [
    ("The inventories cover different areas.", False),
    ("Each inventory retained the ARVO field set.", True),
])
def test_undecided_naming_only_triggers_clarification_for_relevant_core(statement, needs_note):
    plan, policy, _ = study()
    claim = {"relation": "other", "statement": statement, "sources": ["A", "B"]}
    undecided = {"decision": "insufficient_or_immaterial", "statement": "", "sources": []}
    answer, clarify = decode_comparison(json.dumps({"claims": [claim], "naming_decision": undecided}), plan, policy)
    assert answer and statement[:-1] in answer
    assert clarify is needs_note


def dispatch_fixture():
    plan, policy, refs = study()
    lease = SimpleNamespace(calls=1, max_calls=4, max_output_tokens=2000, output_tokens=100,
                            remaining_seconds=lambda: 45, check_active=lambda: None)
    protection = EvidenceProtection("selected sources", 8192, 100, 10, synthesis_lease=lease)
    publication = SimpleNamespace(policy=policy, lease=lease, _authorized=AsyncMock(),
                                  accepts_comparison_text=lambda text: True)
    args = dict(messages=[{"role":"user","content":"selected sources"}],
                messages_pydantic=[UserMessage(content="selected sources")], max_tokens=512)
    return plan, refs, publication, protection, args


def completion(text, stop="stop"):
    return LLMResponse(text=text, stop_reason=stop, tool_call=None, input_usage=10, output_usage=12)


@pytest.mark.asyncio
async def test_optional_bad_or_cutoff_note_cannot_replace_valid_core():
    plan, refs, pub, guard, args = dispatch_fixture()
    claim = dict(relation="retained",statement="Both releases retained the ARVO field set.",
                 sources=["A","B"],reporting_quote="")
    structured = completion(json.dumps(dict(claims=[claim],naming_decision={})))
    llm = SimpleNamespace(supports_json_schema=True,
        get_message=AsyncMock(side_effect=[structured, completion("unfinished", "length")]))
    with protect_evidence(guard):
        answer = await write_comparison(llm, args, plan, pub)
    assert answer.text == f"Both releases retained the ARVO field set {refs[0]}{refs[1]}."
    assert (answer.input_usage, answer.output_usage) == (20,24)
    first, second = llm.get_message.await_args_list
    assert first.kwargs["messages"][-1]["content"] == plan.brief
    assert len(first.kwargs["messages"]) == len(first.kwargs["messages_pydantic"]) == 2
    assert second.kwargs["max_tokens"] == 384
    schema = second.kwargs["response_format"]["json_schema"]
    assert schema["name"] == "comparison_note"
    assert schema["schema"]["properties"]["decision"]["enum"] == ["explain", "qualified", "skip"]
    note_prompt = second.kwargs["messages"][-1]["content"]
    assert second.kwargs["messages_pydantic"][-1].content == note_prompt
    # Fixed input gives an exact wire comparison against the tested clarification prompt.
    from hashlib import sha256
    assert sha256(note_prompt.encode()).hexdigest() == "f0b2c85ac949f453405511ae96b3499b6c6c6f0e65055096d5ec5383ce3911f9"
    assert json.loads(note_prompt.splitlines()[-1]) == [
        {"id": key, "actual_citation": ref, "exact_quote": quote}
        for key, (ref, _, _, quote) in zip("ABC", plan.references, strict=True)
    ]
    assert len(args["messages"]) == 1


@pytest.mark.asyncio
async def test_revocation_propagates_and_changed_source_plan_is_never_sent():
    plan, _, pub, guard, args = dispatch_fixture()
    llm = SimpleNamespace(supports_json_schema=True, get_message=AsyncMock(return_value=completion("ordinary")))
    pub._authorized.side_effect = AnswerPublicationRejected("authority_unavailable")
    with protect_evidence(guard), pytest.raises(AnswerPublicationRejected):
        await write_comparison(llm, args, plan, pub)
    llm.get_message.assert_not_awaited()
    pub._authorized.side_effect = None
    pub.policy = replace(pub.policy, sources=pub.policy.sources[:-1])
    with protect_evidence(guard):
        assert (await write_comparison(llm, args, plan, pub)).text == "ordinary"
    assert llm.get_message.await_args.kwargs == args


def test_note_publishes_only_model_chosen_jointly_supported_sources():
    plan, policy, refs = study()
    sentence = "The inventories use ARVO field set while the independent register calls it ARVO K3."
    obj = dict(decision="explain", statement=sentence, sources=["B","A","C"])
    assert decode_note(json.dumps(obj),plan,policy) == sentence[:-1] + f" {refs[1]}{refs[0]}{refs[2]}."
    assert decode_note(json.dumps(dict(obj,sources=["A","B"])),plan,policy) is None
    assert decode_note(json.dumps(dict(obj,statement=sentence+" [doc:other chunk:99]")),plan,policy) is None
    assert decode_note(json.dumps(dict(decision="skip",statement="",sources=[])),plan,policy) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["comparison_clarification", "comparison_paired_support"])
async def test_optional_comparison_does_not_retry_the_local_sdk_on_timeout(monkeypatch, phase):
    from lib.llm.providers.openai import OpenAIInterface
    from lib.llm.providers import openai_runtime_config

    transport = AsyncMock(side_effect=TimeoutError("timed out"))
    client = SimpleNamespace(base_url="http://vllm:8000/v1",
                             chat=SimpleNamespace(completions=SimpleNamespace(create=transport)))
    client.with_options = lambda **kwargs: client
    provider = OpenAIInterface(client, "local-test")
    monkeypatch.setattr(openai_runtime_config, "request_limits", lambda: (1.0, 10.0, 2, 3))
    with pytest.raises(TimeoutError):
        await provider.get_message(model="local-test",system="Use the evidence.",
                                   messages=[{"role":"user","content":"Compare the records."}],
                                   max_tokens=100,thinking_budget=0,_synthesis_phase=phase)
    assert transport.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["comparison_clarification", "comparison_paired_support"])
async def test_sdk_http_retries_are_disabled_only_for_the_optional_request(monkeypatch, phase):
    import httpx
    from openai import AsyncOpenAI, RateLimitError
    from lib.llm.providers import openai_runtime_config
    from lib.llm.providers.openai import OpenAIInterface

    attempts = []
    async def transport(request):
        attempts.append(request)
        if len(attempts) == 1:
            return httpx.Response(429, headers={"retry-after-ms":"1"}, json={"error":{"message":"busy"}})
        return httpx.Response(200, json={
            "id":"test-reply", "object":"chat.completion", "created":0, "model":"local",
            "choices":[{"index":0, "message":{"role":"assistant", "content":"Answer from evidence."},
                        "finish_reason":"stop"}],
            "usage":{"prompt_tokens":5, "completion_tokens":4, "total_tokens":9},
        })

    monkeypatch.setattr(openai_runtime_config, "request_limits", lambda: (1.0, 10.0, 2, 3))
    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as http:
        client = AsyncOpenAI(base_url="http://vllm:8000/v1", api_key="unused-fixture", http_client=http)
        provider = OpenAIInterface(client, "local")
        args = dict(model="local", system="Use the evidence.", max_tokens=100, thinking_budget=0,
                    messages=[{"role":"user","content":"Compare the records."}])
        with pytest.raises(RateLimitError):
            await provider.get_message(**args, _synthesis_phase=phase)
        assert len(attempts) == 1
        attempts.clear()
        response = await provider.get_message(**args)
        assert len(attempts) == 2
        assert response.text == "Answer from evidence."

def test_qualified_note_requires_both_names_and_all_three_model_chosen_sources():
    plan, policy, refs = study()
    statement = "ARVO field set and ARVO K3 may describe related observations."
    note = {"decision":"qualified", "statement":statement, "sources":["C","A","B"]}
    assert decode_note(json.dumps(note), plan, policy) == statement[:-1]+f" {refs[2]}{refs[0]}{refs[1]}."
    assert decode_note(json.dumps(dict(note, sources=["A","B"])), plan, policy) is None
    assert decode_note(json.dumps(dict(note, statement="These observations could be related.")), plan, policy) is None
    assert decode_note(json.dumps(dict(note, statement=statement+" There is a second claim.")), plan, policy) is None
