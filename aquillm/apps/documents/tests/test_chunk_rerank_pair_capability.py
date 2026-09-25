"""Warm protocol checks must prove identity and reject silent overflow scoring."""

from types import SimpleNamespace

import pytest

from lib.retrieval import TurnBudget, TurnLimits


class Tokenizer:
    def apply_chat_template(self, messages, *, chat_template, tokenize):
        assert [m["role"] for m in messages] == ["query", "document"]
        assert tokenize is False
        return "Q:" + messages[0]["content"] + " D:" + messages[1]["content"]

    def encode(self, prompt, *, add_special_tokens):
        assert add_special_tokens
        return [1] + list(prompt.encode("utf-8")) + [2]


@pytest.mark.parametrize("silent_overflow", [False, True])
def test_warm_validation_requires_token_ids_usage_and_recognized_overflow(
    silent_overflow,
):
    from apps.documents.services.chunk_rerank_pair_capability import (
        registered_pair_counter,
        verify_pair_capability,
    )

    provider = SimpleNamespace(
        endpoint="http://test/score",
        model_name="m",
        headers={},
        scorer_fingerprint="capability-fixture",
        shape="score_single_text_pair",
    )
    budget = TurnBudget(TurnLimits())

    def post(endpoint, *, json, headers, timeout):
        assert 0 < timeout <= 3
        if endpoint.endswith("/tokenize"):
            ids = Tokenizer().encode(json["prompt"], add_special_tokens=True)
            body = {"tokens": ids, "count": len(ids), "max_model_len": 1024}
            return SimpleNamespace(status_code=200, json=lambda: body)
        assert json["truncate_prompt_tokens"] is None
        assert json["max_tokens_per_query"] == json["max_tokens_per_doc"] == 0
        size = len(
            Tokenizer().encode(
                "Q:" + json["text_1"] + " D:" + json["text_2"], add_special_tokens=True
            )
        )
        if size > 1024 and not silent_overflow:
            body = {
                "error": {
                    "type": "BadRequestError",
                    "code": 400,
                    "param": "input_tokens",
                    "message": "maximum context length is 1024 tokens; input has 1100"
                    " tokens",
                }
            }
            return SimpleNamespace(status_code=400, json=lambda: body)
        return SimpleNamespace(
            status_code=200,
            json=lambda: {"score": 0.5, "usage": {"prompt_tokens": size}},
        )

    from hashlib import sha256

    identity = {
        "runtime_digest": "sha256:" + "a" * 64,
        "model_revision": "b" * 40,
        "tokenizer_revision": "c" * 40,
        "code_revision": "d" * 40,
        "template_sha256": sha256(b"template").hexdigest(),
    }
    result = verify_pair_capability(
        provider,
        tokenizer=Tokenizer(),
        template="template",
        identity=identity,
        budget=budget,
        post=post,
    )
    assert (result is not None) is (not silent_overflow)
    assert budget.pairs_used["acquisition"] == 3
    assert (registered_pair_counter(provider) is not None) is (not silent_overflow)
