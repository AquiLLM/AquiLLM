from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from lib.knowledge_graph.query_extractor import service
from lib.knowledge_graph.query_extractor.config import load_query_extractor_settings
from lib.knowledge_graph.query_extractor.contracts import (
    QUERY_EXTRACTION_RESPONSE_SCHEMA_CHECKSUM,
    QueryExtractionRequestV1,
    canonical_query_extraction_request_bytes,
    parse_query_extraction_response,
)
from lib.knowledge_graph.types import EntityCandidate, ExtractionBatchResult

DIGEST = "a" * 64
REVISION = "8437ba583a733d87f56ae902f3b197934eedd58e"
BUILD = "b" * 64
EMOJI = chr(0x1F600)


def _settings():
    return load_query_extractor_settings(
        {
            "KG_QUERY_EXTRACTOR_URL": "http://extractor:8080/v1/extract",
            "KG_QUERY_EXTRACTOR_BEARER_TOKEN": "private-token",
            "KG_QUERY_EXTRACTOR_MODEL": "fastino/gliner2-base-v1",
            "KG_QUERY_EXTRACTOR_MODEL_REVISION": REVISION,
            "KG_QUERY_EXTRACTOR_BUILD_HASH": BUILD,
            "KG_QUERY_EXTRACTOR_EXPECTED_SCHEMA_VERSION": "query-entities-v1",
            "KG_QUERY_EXTRACTOR_EXPECTED_SCHEMA_CHECKSUM": (
                QUERY_EXTRACTION_RESPONSE_SCHEMA_CHECKSUM
            ),
            "KG_QUERY_EXTRACTOR_ONTOLOGY_PATH": "ontology.yaml",
            "KG_QUERY_EXTRACTOR_ONTOLOGY_CHECKSUM": DIGEST,
            "KG_QUERY_EXTRACTOR_TIMEOUT_MS": "75",
            "KG_QUERY_MAX_BYTES": "64",
            "KG_QUERY_MAX_CODEPOINTS": "32",
            "KG_QUERY_MAX_SPANS": "4",
        }
    )


class Backend:
    def extract_batch(self, texts, *, ontology):
        assert texts == (f"A{EMOJI}B",)
        assert ontology.checksum == DIGEST
        return (
            ExtractionBatchResult(
                entities=(EntityCandidate("model", EMOJI, 1, 2, 0.75),),
                relations=(),
                diagnostics=(),
            ),
        )


async def _call(
    *,
    path: str,
    method: str = "POST",
    body: bytes = b"",
    authorization: bytes | None = None,
):
    headers = [] if authorization is None else [(b"authorization", authorization)]
    scope = {"type": "http", "path": path, "method": method, "headers": headers}
    messages = [{"type": "http.request", "body": body, "more_body": False}]
    sent: list[dict[str, object]] = []

    async def receive():
        return messages.pop(0)

    async def send(message):
        sent.append(message)

    await service.app(scope, receive, send)
    return sent


def _request(query: str | None = None) -> bytes:
    query = f"A{EMOJI}B" if query is None else query
    return canonical_query_extraction_request_bytes(
        QueryExtractionRequestV1(
            "query-request-v1",
            query,
            DIGEST,
            64,
            32,
            4,
        )
    )


@pytest.fixture(autouse=True)
def runtime(monkeypatch):
    value = service.QueryExtractorRuntime(
        settings=_settings(),
        ontology=SimpleNamespace(checksum=DIGEST, entity_types={"model": object()}),
        backend=Backend(),
    )
    monkeypatch.setattr(service, "_get_runtime", lambda: value)


@pytest.mark.asyncio
async def test_bearer_auth_is_constant_time_and_response_spans_are_text_free(
    monkeypatch,
) -> None:
    comparisons: list[tuple[str, str]] = []

    def compare(left: str, right: str) -> bool:
        comparisons.append((left, right))
        return left == right

    monkeypatch.setattr(service, "compare_digest", compare)
    unauthorized = await _call(path="/v1/extract", body=_request())
    assert unauthorized[0]["status"] == 401
    assert comparisons == [("", "Bearer private-token")]

    authorized = await _call(
        path="/v1/extract",
        body=_request(),
        authorization=b"Bearer private-token",
    )
    assert authorized[0]["status"] == 200
    payload = b"".join(message.get("body", b"") for message in authorized[1:])
    response = parse_query_extraction_response(payload)
    assert response.query_utf8_bytes == 6
    assert response.query_code_points == 3
    assert not hasattr(response.spans[0], "text")
    assert b"\xf0\x9f\x98\x80" not in payload


@pytest.mark.asyncio
async def test_service_enforces_body_and_contract_caps_without_echoing_payload() -> (
    None
):
    oversized = await _call(
        path="/v1/extract",
        body=b"secret-canary" * 3_000,
        authorization=b"Bearer private-token",
    )
    assert oversized[0]["status"] == 413
    encoded = repr(oversized).encode()
    assert b"secret-canary" not in encoded

    import json

    wrong_caps = json.loads(_request())
    wrong_caps["max_spans"] = 3
    noncanonical = json.dumps(
        wrong_caps, sort_keys=True, separators=(",", ":")
    ).encode()
    rejected = await _call(
        path="/v1/extract",
        body=noncanonical,
        authorization=b"Bearer private-token",
    )
    assert rejected[0]["status"] == 422


@pytest.mark.asyncio
async def test_health_and_route_dispatch_are_minimal() -> None:
    health = await _call(path="/healthz", method="GET")
    missing = await _call(path="/elsewhere", method="GET")
    assert health[0]["status"] == 200
    assert missing[0]["status"] == 404
    assert service.UVICORN_ACCESS_LOG is False


def test_importing_service_does_not_import_ml_runtime() -> None:
    root = Path(__file__).resolve().parents[3]
    script = (
        "import sys; import lib.knowledge_graph.query_extractor.service; "
        "assert not ({'gliner2','torch','huggingface_hub'} & set(sys.modules))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
