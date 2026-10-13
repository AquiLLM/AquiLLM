"""Request work never negotiates protocols or retries unavailable scoring."""

from dataclasses import dataclass
from math import isfinite
from threading import RLock
from time import time
from typing import Literal
from urllib.parse import urlsplit

_KINDS = {"score_single_text_pair", "score_batch_text_pairs", "rerank_documents"}
_BINDINGS = {}
_BINDING_LOCK = RLock()


@dataclass(frozen=True, slots=True)
class RerankProtocolV1:
    endpoint: str
    payload_kind: str
    model_key: str
    verified_at: float
    expires_at: float

    def __post_init__(self):
        if type(self.endpoint) is not str:
            raise ValueError("invalid endpoint")
        url = urlsplit(self.endpoint)
        if (
            url.scheme not in ("http", "https")
            or not url.hostname
            or url.username
            or url.password
        ):
            raise ValueError("invalid endpoint")
        if (
            self.payload_kind not in _KINDS
            or type(self.model_key) is not str
            or not self.model_key
        ):
            raise ValueError("invalid protocol identity")
        if any(
            type(x) not in (int, float) or not isfinite(x)
            for x in (self.verified_at, self.expires_at)
        ):
            raise ValueError("invalid verification expiration")
        if self.verified_at < 0 or self.expires_at <= self.verified_at:
            raise ValueError("invalid verification interval")

    def live(self, model_key, *, now=None):
        now = time() if now is None else now
        return self.model_key == model_key and self.verified_at <= now < self.expires_at

    def as_cache_entry(self):
        return dict(
            endpoint=self.endpoint,
            shape=self.payload_kind,
            model_key=self.model_key,
            verified_at=self.verified_at,
            expires_at=self.expires_at,
            protocol_version="v1",
        )


def finite_protocol_scores(
    body, *, payload_kind, count, model_key, require_identity=False
):
    """Validate the entire response, including duplicate/bool indices and model."""
    if payload_kind not in _KINDS or type(count) is not int or not 1 <= count <= 6:
        return None
    if isinstance(body, dict):
        if "model" in body and body["model"] != model_key:
            return None
    if require_identity and (
        not isinstance(body, dict) or body.get("model") != model_key
    ):
        return None

    def finite(value):
        return type(value) in (int, float) and isfinite(float(value))

    if payload_kind == "score_single_text_pair":
        if count != 1:
            return None
        value = body.get("score") if isinstance(body, dict) else body
        if not finite(value) and isinstance(body, dict):
            items = body.get("data", body.get("results"))
            if type(items) is list and len(items) == 1 and isinstance(items[0], dict):
                item = items[0]
                if "index" in item and (
                    type(item["index"]) is not int or item["index"] != 0
                ):
                    return None
                value = item.get("score", item.get("relevance_score"))
        return (float(value),) if finite(value) else None
    items = body.get("data", body.get("results")) if isinstance(body, dict) else None
    if type(items) is not list or len(items) != count:
        return None
    values = {}
    for item in items:
        if not isinstance(item, dict):
            return None
        index = item.get("index")
        value = item.get("score", item.get("relevance_score"))
        if (
            type(index) is not int
            or not 0 <= index < count
            or index in values
            or not finite(value)
        ):
            return None
        values[index] = float(value)
    return tuple(values[i] for i in range(count))


def rerank_retry_action(
    status: int | None, *, discovery: bool
) -> Literal["stop", "next_protocol"]:
    if type(discovery) is not bool:
        raise TypeError("discovery must be boolean")
    if status is not None:
        if type(status) is not int:
            raise TypeError("status must be an HTTP integer or None")
        if not 100 <= status <= 599:
            raise ValueError("invalid HTTP status")
    return "next_protocol" if discovery and status in (404, 405, 415, 422) else "stop"


def resolve_rerank_protocol(base_url, model_key, *, budget=None):
    """Old shape-only entries are unverified; serving never upgrades them."""
    from . import rag_cache

    with _BINDING_LOCK:
        binding = _BINDINGS.get((base_url.rstrip("/"), model_key))
        if binding is not None and binding.live(model_key):
            return binding
    entry = rag_cache.get_cached_rerank_capability(
        base_url, model_key, **({"budget": budget} if budget else {})
    )
    if not isinstance(entry, dict) or entry.get("protocol_version") != "v1":
        return None
    try:
        binding = RerankProtocolV1(
            entry["endpoint"],
            entry["shape"],
            entry["model_key"],
            entry["verified_at"],
            entry["expires_at"],
        )
    except (KeyError, TypeError, ValueError):
        return None
    root = base_url.rstrip("/").removesuffix("/v1")
    endpoints = {
        f"{root}/score",
        f"{root}/v1/score",
        f"{root}/rerank",
        f"{root}/v2/rerank",
        f"{root}/v1/rerank",
    }
    return (
        binding if binding.endpoint in endpoints and binding.live(model_key) else None
    )


def register_rerank_protocol(base_url, binding):
    """Called after an out-of-request finite-score/identity verification."""
    from . import rag_cache

    if not isinstance(binding, RerankProtocolV1) or not binding.live(binding.model_key):
        raise ValueError("verified live binding required")
    with _BINDING_LOCK:
        _BINDINGS[(base_url.rstrip("/"), binding.model_key)] = binding
    rag_cache.set_cached_rerank_capability(
        base_url, binding.model_key, binding.as_cache_entry()
    )


def revoke_rerank_protocol(base_url, binding):
    from . import rag_cache

    with _BINDING_LOCK:
        key = (base_url.rstrip("/"), binding.model_key)
        if _BINDINGS.get(key) == binding:
            _BINDINGS.pop(key)
    entry = rag_cache.get_cached_rerank_capability(base_url, binding.model_key)
    if entry == binding.as_cache_entry():
        rag_cache.delete_cached_rerank_capability(base_url, binding.model_key)


def discover_rerank_protocol(
    base_url, model_key, *, deadline, max_attempts=3, headers=None, budget=None,
    on_registered=None,
):
    """Explicit bounded worker/preflight discovery, never a serving fallback."""
    from time import monotonic

    import requests

    from lib.llm.turn_context import current_turn
    from lib.retrieval.turn_budget import TurnBudget, TurnLimits

    from .rerank_request_lifetime import TurnRerankState
    from .rerank_protocol_publication import publish_owned_protocol

    if current_turn() is not None:
        raise RuntimeError("protocol discovery is outside request work")
    if type(max_attempts) is not int or not 1 <= max_attempts <= 3:
        raise ValueError("discovery attempts must be within three")
    milliseconds = min(15_000, max(0, int((deadline - monotonic()) * 1000)))
    if milliseconds <= 0:
        return None
    budget = budget or TurnBudget(
        TurnLimits(
            acquisition_pairs=max_attempts,
            final_pairs=0,
            in_flight_pairs=1,
            retrieval_ms=milliseconds,
            final_scoring_ms=0,
        )
    )
    state = TurnRerankState(budget)
    root = base_url.rstrip("/").removesuffix("/v1")
    candidates = (
        (f"{root}/score", "score_batch_text_pairs"),
        (f"{root}/score", "score_single_text_pair"),
        (f"{root}/rerank", "rerank_documents"),
    )
    for endpoint, kind in candidates[:max_attempts]:

        def probe(timeout):
            payload = protocol_payload(
                kind, model_key, (("protocol verification", "synthetic evidence"),)
            )
            response = requests.post(
                endpoint,
                headers=headers or {},
                json=payload,
                timeout=min(3, timeout, max(0.001, deadline - monotonic())),
                allow_redirects=False,
            )
            if response.status_code != 200:
                return (
                    ("next",)
                    if rerank_retry_action(response.status_code, discovery=True)
                    == "next_protocol"
                    else None
                )
            # A redirect is not proof of the endpoint to which we bind.
            if getattr(response, "url", endpoint) != endpoint:
                return None
            values = finite_protocol_scores(
                response.json(),
                payload_kind=kind,
                count=1,
                model_key=model_key,
                require_identity=True,
            )
            if values is None:
                return None
            now = time()
            return RerankProtocolV1(endpoint, kind, model_key, now, now + 300)

        result = state.run(1, probe, timeout=min(3, max(0, deadline - monotonic())))
        if isinstance(result, RerankProtocolV1):
            if not state.publish(
                lambda: publish_owned_protocol(base_url, result, on_registered)
            ):
                return None
            return result
        if result != ("next",):
            return None
    return None


def protocol_payload(kind, model_key, pairs, *, pair_limit=None):
    if not pairs or len(pairs) > 6:
        raise ValueError("transport must contain one to six pairs")
    if kind == "score_single_text_pair":
        if len(pairs) != 1:
            raise ValueError("pointwise transport requires one pair")
        payload = {"model": model_key, "text_1": pairs[0][0], "text_2": pairs[0][1]}
    elif kind == "score_batch_text_pairs":
        payload = {
            "model": model_key,
            "text_1": [q for q, _ in pairs],
            "text_2": [d for _, d in pairs],
        }
        if len({q for q, _ in pairs}) == 1:
            payload["text_1"] = pairs[0][0]
    elif kind == "rerank_documents" and len({q for q, _ in pairs}) == 1:
        payload = {
            "model": model_key,
            "query": pairs[0][0],
            "documents": [d for _, d in pairs],
            "top_n": len(pairs),
        }
    else:
        raise ValueError("unsupported bound protocol or heterogeneous listwise query")
    if pair_limit is not None and kind != "rerank_documents":
        payload.update(truncate_prompt_tokens=pair_limit, truncation_side="right")
    return payload
