"""Serving never drops configured model identity to retry rejected scoring."""
from time import monotonic
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from apps.documents.services import rag_cache, rerank_transport_policy as policy
from apps.documents.services.chunk_rerank import rerank_chunks_scored


def _row(pk, content):
    return SimpleNamespace(
        pk=pk, content=content, doc_id=UUID(int=1), chunk_number=pk,
        modality="text", Modality=SimpleNamespace(IMAGE="image"),
    )


@pytest.fixture
def configured(monkeypatch, settings):
    base = f"http://identity-{uuid4().hex}.invalid/v1"
    monkeypatch.setenv("APP_RERANK_BASE_URL", base)
    monkeypatch.setenv("APP_RERANK_PROVIDER", "local")
    monkeypatch.setenv("APP_RERANK_MODEL", "synthetic-pinned")
    monkeypatch.setenv("APP_RERANK_MODEL_REVISION", "pinned")
    monkeypatch.setenv("RAG_RERANK_TEXT_MODE", "legacy")
    settings.RAG_CACHE_ENABLED = False
    writes = []
    monkeypatch.setattr(rag_cache, "cache_get", lambda key: None)
    monkeypatch.setattr(rag_cache, "cache_set", lambda *args: writes.append(args))
    try:
        yield base, writes
    finally:
        binding = policy.resolve_rerank_protocol(base, "synthetic-pinned")
        if binding is not None:
            policy.revoke_rerank_protocol(base, binding)


def _transport(monkeypatch, kind, *, reject_serving=False, identity=True):
    calls = []
    phase = {"value": "discovery"}

    def post(endpoint, *, json, **kwargs):
        calls.append((phase["value"], endpoint, json))
        assert kwargs["allow_redirects"] is False
        assert 0 < kwargs["timeout"] <= 3
        if kind == "rerank_documents" and endpoint.endswith("/score"):
            return SimpleNamespace(status_code=422)
        # A model-less fallback would succeed and reverse the input ranking.
        rejected = phase["value"] == "serving" and reject_serving and "model" in json
        count = len(json["documents"] if "documents" in json else json["text_2"])
        body = {"data": [{"index": i, "score": 0.1 + 0.8 * i} for i in range(count)]}
        if identity:
            body["model"] = "synthetic-pinned"
        return SimpleNamespace(
            status_code=400 if rejected else 200, url=endpoint, json=lambda: body,
        )

    monkeypatch.setattr("requests.post", post)
    return calls, phase


def _discover(base, kind):
    binding = policy.discover_rerank_protocol(
        base, "synthetic-pinned", deadline=monotonic() + 3,
    )
    assert binding is not None and binding.payload_kind == kind
    assert policy.resolve_rerank_protocol(base, "synthetic-pinned") is binding
    return binding


@pytest.mark.parametrize("kind", ["score_batch_text_pairs", "rerank_documents"])
@pytest.mark.parametrize("reject_serving", [False, True])
def test_verified_model_is_preserved_without_model_less_retry(
    monkeypatch, configured, kind, reject_serving,
):
    base, writes = configured
    calls, phase = _transport(monkeypatch, kind, reject_serving=reject_serving)
    binding = _discover(base, kind)
    phase["value"] = "serving"
    result = rerank_chunks_scored(object, "question", [_row(10, "a"), _row(20, "b")], 1)
    serving = [call for call in calls if call[0] == "serving"]
    assert len(serving) == 1 and serving[0][1] == binding.endpoint
    assert all(payload["model"] == "synthetic-pinned" for _, _, payload in calls)
    if reject_serving:
        assert result.ranked_ids == (10,)
        assert result.score_set.status == "unavailable"
        assert result.score_set.scorer_fingerprint == ""
        assert result.score_set.scores == ()
        assert writes == []
    else:
        assert result.ranked_ids == (20,)
        assert result.score_set.status == "complete"
        assert result.score_set.scorer_fingerprint
        assert tuple((s.chunk_pk, s.value) for s in result.score_set.scores) == ((10, 0.1), (20, 0.9))
        assert len(writes) == 1 and writes[0][0].startswith("rrscore:v2:")


@pytest.mark.parametrize("kind", ["score_batch_text_pairs", "rerank_documents"])
def test_missing_discovery_model_identity_never_admits_serving(monkeypatch, configured, kind):
    base, writes = configured
    calls, phase = _transport(monkeypatch, kind, identity=False)
    assert policy.discover_rerank_protocol(
        base, "synthetic-pinned", deadline=monotonic() + 3,
    ) is None
    assert policy.resolve_rerank_protocol(base, "synthetic-pinned") is None
    phase["value"] = "serving"
    result = rerank_chunks_scored(object, "question", [_row(10, "a"), _row(20, "b")], 1)
    assert all(phase_name == "discovery" for phase_name, _, _ in calls)
    assert len(calls) == (3 if kind == "rerank_documents" else 1)
    assert all(payload["model"] == "synthetic-pinned" for _, _, payload in calls)
    assert result.ranked_ids == (10,)
    assert result.score_set.status == "unavailable"
    assert result.score_set.scorer_fingerprint == "" and result.score_set.scores == ()
    assert writes == []
