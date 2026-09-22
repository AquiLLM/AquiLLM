# ruff: noqa: F401,E501
"""Related graph infrastructure regression scenarios."""

from lib.knowledge_graph.tests.test_query_extractor_service import (
    DIGEST,
    EMOJI,
    EntityCandidate,
    ExtractionBatchResult,
    Path,
    SimpleNamespace,
    SlowBackend,
    _call,
    _request,
    _settings,
    _use_backend,
    _wait_for_inference_slot,
    asyncio,
    json,
    parse_query_extraction_response,
    pytest,
    replace,
    runtime,
    service,
    subprocess,
    sys,
    time,
    yaml,
)


@pytest.mark.asyncio
async def test_inference_runs_off_loop_and_rejects_concurrent_overload(monkeypatch) -> None:
    _use_backend(monkeypatch, SlowBackend(0.05))
    extraction = asyncio.create_task(_call(path="/v1/extract", body=_request(), authorization=b"Bearer private-token"))
    await asyncio.sleep(0.005)
    assert not extraction.done()
    health = await asyncio.wait_for(_call(path="/healthz", method="GET"), timeout=0.02)
    overloaded = await _call(path="/v1/extract", body=_request(), authorization=b"Bearer private-token")
    assert health[0]["status"] == 200
    assert overloaded[0]["status"] == 503
    assert (await extraction)[0]["status"] == 200



@pytest.mark.asyncio
async def test_timeout_and_cancellation_hold_slot_until_worker_finishes(monkeypatch) -> None:
    backend = SlowBackend(0.05)
    _use_backend(monkeypatch, backend, timeout_ms=10)
    timed_out = await _call(path="/v1/extract", body=_request(), authorization=b"Bearer private-token")
    assert timed_out[0]["status"] == 503
    assert timed_out[1]["body"] == b'{"reason":"extractor_timeout"}'
    assert (await _call(path="/v1/extract", body=_request(), authorization=b"Bearer private-token"))[0]["status"] == 503
    await _wait_for_inference_slot()
    backend.delay = 0.05
    cancelled = asyncio.create_task(_call(path="/v1/extract", body=_request(), authorization=b"Bearer private-token"))
    await asyncio.sleep(0.005)
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    assert (await _call(path="/v1/extract", body=_request(), authorization=b"Bearer private-token"))[0]["status"] == 503
    await _wait_for_inference_slot()
    backend.delay = 0.0
    _use_backend(monkeypatch, backend)
    recovered = await _call(path="/v1/extract", body=_request(), authorization=b"Bearer private-token")
    assert recovered[0]["status"] == 200, recovered



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



def _custom_ontology(version="0.0.1+collection.223"):
    from apps.knowledge_graph.services.ontology import load_ontology_yaml

    return load_ontology_yaml(
        yaml.safe_dump(
            {
                "version": version,
                "entity_types": [
                    {
                        "name": "audit_entity",
                        "description": "Audit entity.",
                        "aliases": [],
                        "default_retrieval_weight": 1.0,
                        "default_suppression_policy": "never",
                        "default_suppression_threshold": 0.0,
                    }
                ],
                "relations": [
                    {
                        "name": "audit_related_to",
                        "description": "Audit relation.",
                        "direction": "directed",
                        "allowed_head_types": ["audit_entity"],
                        "allowed_tail_types": ["audit_entity"],
                    }
                ],
            }
        )
    )



def _custom_request(ontology):
    payload = json.loads(_request())
    payload["ontology_checksum"] = ontology.checksum
    payload["ontology_definition"] = yaml.safe_load(ontology.canonical_yaml)
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()



def test_custom_ontology_is_request_local_and_response_binds_its_checksum(monkeypatch):
    seen = []

    class CustomBackend:
        def extract_entities_batch(self, texts, *, ontology):
            seen.append(ontology.checksum)
            return (
                ExtractionBatchResult(
                    entities=(EntityCandidate("audit_entity", EMOJI, 1, 2, 0.75),),
                    relations=(),
                    diagnostics=(),
                ),
            )

    pinned = service.QueryExtractorRuntime(
        settings=_settings(),
        ontology=SimpleNamespace(checksum=DIGEST),
        backend=CustomBackend(),
    )
    monkeypatch.setattr(service, "_get_runtime", lambda *_args: pinned)
    definitions = (_custom_ontology(), _custom_ontology("0.0.2+collection.224"))
    for definition in definitions:
        sent = asyncio.run(
            _call(
                path="/v1/extract",
                body=_custom_request(definition),
                authorization=b"Bearer private-token",
            )
        )
        assert sent[0]["status"] == 200
        response = parse_query_extraction_response(sent[1]["body"])
        assert response.provenance.ontology_checksum == definition.checksum
        assert response.spans[0].ontology_type == "audit_entity"
        assert pinned.ontology.checksum == DIGEST
    assert seen == [definition.checksum for definition in definitions]



@pytest.mark.parametrize(
    "mutation",
    [
        "checksum",
        "name",
        "endpoint",
        "number",
        "extra",
        "count",
        "description",
    ],
)
def test_invalid_dynamic_ontology_never_reaches_provider(monkeypatch, mutation):
    body = json.loads(_custom_request(_custom_ontology()))
    definition = body["ontology_definition"]
    if mutation == "checksum":
        body["ontology_checksum"] = "f" * 64
    elif mutation == "name":
        definition["entity_types"][0]["name"] = "entities"
    elif mutation == "endpoint":
        definition["relations"][0]["allowed_head_types"] = ["foreign"]
    elif mutation == "number":
        definition["entity_types"][0]["default_retrieval_weight"] = True
    elif mutation == "extra":
        definition["private_metadata"] = "not_allowed"
    elif mutation == "count":
        definition["entity_types"] *= 65
    else:
        definition["entity_types"][0]["description"] = "x" * 513
    calls = []

    class NeverBackend:
        def extract_entities_batch(self, *args, **kwargs):
            calls.append(True)
            raise AssertionError("invalid definition reached provider")

    pinned = service.QueryExtractorRuntime(
        settings=_settings(),
        ontology=SimpleNamespace(checksum=DIGEST),
        backend=NeverBackend(),
    )
    monkeypatch.setattr(service, "_get_runtime", lambda *_args: pinned)
    sent = asyncio.run(
        _call(
            path="/v1/extract",
            body=json.dumps(
                body, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode(),
            authorization=b"Bearer private-token",
        )
    )
    assert sent[0]["status"] == 422
    assert calls == []



def test_custom_ontology_requires_authentication():
    sent = asyncio.run(
        _call(path="/v1/extract", body=_custom_request(_custom_ontology()))
    )
    assert sent[0]["status"] == 401



@pytest.mark.parametrize("kind", ["checksum", "oversized", "caps"])
def test_invalid_authenticated_request_does_not_initialize_backend(monkeypatch, kind):
    calls = []

    def runtime_loader(*_args):
        calls.append(True)
        raise AssertionError("invalid request initialized backend")

    monkeypatch.setattr(service, "_get_runtime", runtime_loader)
    payload = json.loads(_custom_request(_custom_ontology()))
    if kind == "checksum":
        payload["ontology_checksum"] = "f" * 64
    else:
        payload["max_spans"] = 3
    body = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    if kind == "oversized":
        body = b"x" * (_settings().max_request_body_bytes + 1)
    sent = asyncio.run(
        _call(path="/v1/extract", body=body, authorization=b"Bearer private-token")
    )
    assert sent[0]["status"] == (413 if kind == "oversized" else 422)
    assert calls == []
