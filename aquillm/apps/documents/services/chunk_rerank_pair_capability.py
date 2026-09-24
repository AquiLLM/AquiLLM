"""Explicit per-worker warm validation of the pinned vLLM score-pair protocol.

The caller must first verify the supplied immutable identity against the deployed
runtime. Configured names alone are not attestation. No implicit request-time
probe, tokenizer download, or remote code loading occurs here. Missing proof or
assets leaves the worker unknown. Reinitialize on runtime replacement.
"""

import json
import re
from dataclasses import dataclass, field
from threading import Event, RLock
from time import monotonic

import requests

from .chunk_rerank_results import fingerprint_text

_registered = {}
_lock = RLock()


@dataclass(frozen=True)
class VerifiedPairCounter:
    tokenizer: object
    template: str
    identity: str
    tokenizer_identity: str
    template_identity: str
    expires: float
    revoked: Event = field(default_factory=Event, compare=False)

    def rendered(self, query, document):
        return self.tokenizer.apply_chat_template(
            [
                {"role": "query", "content": query},
                {"role": "document", "content": document},
            ],
            chat_template=self.template,
            tokenize=False,
        )

    def token_ids(self, query, document):
        return self.tokenizer.encode(
            self.rendered(query, document), add_special_tokens=True
        )

    def input_codepoints(self, query, document):
        return len(self.rendered(query, document))

    def __call__(self, query, document):
        return (
            len(self.token_ids(query, document))
            if not self.revoked.is_set() and monotonic() < self.expires
            else None
        )


def registered_pair_counter(provider):
    if provider is None:
        return None
    with _lock:
        result = _registered.get(provider.scorer_fingerprint)
        return (
            result
            if result is not None
            and not result.revoked.is_set()
            and monotonic() < result.expires
            else None
        )


def invalidate_pair_capability(provider=None):
    """Revoke even counters held by an already-created request scorer."""
    with _lock:
        keys = [provider.scorer_fingerprint] if provider else list(_registered)
        for key in keys:
            counter = _registered.pop(key, None)
            if counter is not None:
                counter.revoked.set()


def _valid_identity(identity, template):
    if not isinstance(identity, dict) or set(identity) != {
        "runtime_digest",
        "model_revision",
        "tokenizer_revision",
        "code_revision",
        "template_sha256",
    }:
        return False
    return (
        all(
            isinstance(identity[k], str) and re.fullmatch(r"[a-f0-9]{40}", identity[k])
            for k in ("model_revision", "tokenizer_revision", "code_revision")
        )
        and bool(re.fullmatch(r"sha256:[a-f0-9]{64}", identity["runtime_digest"]))
        and fingerprint_text(template) == identity["template_sha256"]
    )


def verify_pair_capability(
    provider, *, tokenizer, template, identity, budget, post=requests.post
):
    """Three bounded inference probes, charged even for overflow rejection.

    This is an explicit worker/bootstrap operation. ``budget`` is mandatory, and
    must be the turn's existing ledger if invoked during a request. Identity must
    come from deployment revision verification, not unverified HTTP metadata.
    """
    if budget is None or not budget.can_publish():
        return None
    invalidate_pair_capability(provider)
    if (
        budget is None
        or provider.shape != "score_single_text_pair"
        or not _valid_identity(identity, template)
    ):
        return None
    counter = VerifiedPairCounter(
        tokenizer,
        template,
        fingerprint_text(json.dumps(identity, sort_keys=True)),
        identity["tokenizer_revision"],
        identity["template_sha256"],
        monotonic() + 300,
    )
    probes = (
        ("What changed?", "Treatment did not improve survival at 5 μg/kg. ΔE=mc²."),
        ("query " * 30, "tail evidence"),
        ("overflow", " overflow" * 1500),
    )
    tokenize_endpoint = provider.endpoint.rsplit("/", 1)[0] + "/tokenize"
    try:
        for index, (query, document) in enumerate(probes):
            if not budget.reserve_text(len(query) + len(document), kind="tokenized"):
                return None
            rendered = counter.rendered(query, document)
            local_ids = tokenizer.encode(rendered, add_special_tokens=True)
            if not budget.reserve_text(len(rendered), kind="tokenized"):
                return None
            remaining = min(3, budget.remaining_ms() / 1000)
            if remaining <= 0:
                return None
            response = post(
                tokenize_endpoint,
                headers=provider.headers,
                json={
                    "model": provider.model_name,
                    "prompt": rendered,
                    "add_special_tokens": True,
                },
                timeout=remaining,
            )
            body = response.json()
            if (
                response.status_code != 200
                or body.get("tokens") != local_ids
                or body.get("count") != len(local_ids)
                or body.get("max_model_len") != 1024
            ):
                return None
            oversized = index == 2
            if (len(local_ids) > 1024) != oversized or not budget.start_pair(
                phase="acquisition"
            ):
                return None
            try:
                remaining = min(3, budget.remaining_ms() / 1000)
                if remaining <= 0:
                    return None
                response = post(
                    provider.endpoint,
                    headers=provider.headers,
                    json={
                        "model": provider.model_name,
                        "text_1": query,
                        "text_2": document,
                        "truncate_prompt_tokens": None,
                        "max_tokens_per_query": 0,
                        "max_tokens_per_doc": 0,
                    },
                    timeout=remaining,
                )
            finally:
                budget.finish_pair()
            body = response.json()
            if oversized:
                error = body.get("error", {})
                message = error.get("message", "")
                if (
                    response.status_code != 400
                    or error.get("type") != "BadRequestError"
                    or error.get("code") != 400
                    or error.get("param") != "input_tokens"
                    or "1024" not in message
                    or "context" not in message
                    or "token" not in message
                ):
                    return None
            elif response.status_code != 200 or body.get("usage", {}).get(
                "prompt_tokens"
            ) != len(local_ids):
                return None

        def register():
            with _lock:
                _registered[provider.scorer_fingerprint] = counter

        return counter if budget.publish(register) else None
    except (requests.RequestException, ValueError, TypeError, AttributeError):
        return None


def initialize_worker_pair_capability(
    provider, *, tokenizer_name, template, identity, budget
):
    """Call explicitly inside each serving worker after deployment attestation.

    A separate management process cannot initialize serving workers. Local cache
    only; a missing pinned tokenizer keeps coverage unknown without downloading.
    """
    if not _valid_identity(identity, template):
        return None
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            revision=identity["tokenizer_revision"],
            local_files_only=True,
            trust_remote_code=False,
        )
    except (ImportError, OSError, ValueError):
        return None
    return verify_pair_capability(
        provider,
        tokenizer=tokenizer,
        template=template,
        identity=identity,
        budget=budget,
    )
