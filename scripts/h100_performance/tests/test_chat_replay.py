"""CPU contract tests; the local HTTP server is not application replay evidence."""

import asyncio
import importlib.util
import json
from pathlib import Path
from urllib.parse import parse_qs

import httpx
import pytest


def module():
    path = Path(__file__).resolve().parents[1] / "chat_replay.py"
    assert path.exists(), "live replay client is missing"
    spec = importlib.util.spec_from_file_location("chat_replay", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def fixture():
    return dict(
        collection_id=12,
        document_id="12345678-1234-5678-1234-567812345678",
        chunk_ids=[34],
        run_id="cpu-fixture",
        chat_code="CHAT-7391",
        rag_code="ORCHID-7391",
        expected_model="frozen-model",
    )


def stream(content=None, done=False, finish=None, uuid="answer"):
    value = dict(role="assistant", message_uuid=uuid, done=done)
    if content is not None:
        value["content"] = content
    if finish is not None:
        value["stop_reason"] = finish
    return {"stream": value}


def delta(content, uuid="answer", **fields):
    return {
        "delta": {
            "messages": [
                dict(role="assistant", message_uuid=uuid, content=content, **fields)
            ]
        }
    }


def test_cumulative_visible_content_requires_final_and_matching_persisted_delta():
    client = module()
    state = client.ReplayState("chat", client.validate_fixture(fixture()), 10.0)
    state.observe(
        {
            "stream": {
                "role": "assistant",
                "message_uuid": "answer",
                "reasoning_content": "hidden",
                "done": False,
            }
        },
        10.1,
    )
    state.observe({"stream": {"role": "tool", "content": "tool output"}}, 10.2)
    state.observe(
        {
            "stream": {
                "role": "assistant",
                "content": "tool output",
                "tool_calls": [{"name": "vector_search"}],
            }
        },
        10.2,
    )
    assert state.first_visible_at is None
    state.observe(stream("CHAT-"), 10.3)
    state.observe(stream("CHAT-7391"), 10.4)
    assert not state.finished
    state.observe(stream("CHAT-7391", True, "stop"), 10.5)
    assert not state.finished
    state.observe(delta("CHAT-7391"), 10.6)
    result = state.result(10.6)
    assert result["complete"] and result["error"] is None
    assert result["output_text"] == result["stream_output_text"] == "CHAT-7391"
    assert result["visible_ttft_seconds"] == pytest.approx(0.3)
    assert result["final_seconds"] == pytest.approx(0.5)
    assert result["total_seconds"] == pytest.approx(0.6)
    assert not result["model_verified"] and not result["route_verified"]


@pytest.mark.parametrize(
    "frames,error",
    [
        ([stream("CHAT-7391")], "missing_stream_done"),
        ([stream("CHAT-7391", True, "stop")], "missing_persisted_delta"),
        ([delta("CHAT-7391")], "missing_stream_done"),
        (
            [stream("CHAT-7391", True, "length"), delta("CHAT-7391")],
            "invalid_finish_reason",
        ),
        (
            [stream("CHAT-7391", True, "stop"), delta("CHAT-7391 NOT")],
            "answer_mismatch",
        ),
        (
            [stream("CHAT-7391", True, "stop"), delta("CHAT-7391", uuid="other")],
            "missing_persisted_delta",
        ),
        ([{"exception": "password=secret", "debug_html": "secret"}], "server_error"),
    ],
)
def test_incomplete_or_error_frames_never_pass_and_server_errors_are_not_retained(
    frames, error
):
    client = module()
    state = client.ReplayState("chat", client.validate_fixture(fixture()), 1.0)
    for index, frame in enumerate(frames):
        state.observe(frame, 1.1 + index / 10)
    result = state.result(2.0)
    assert not result["complete"] and result["error"] == error
    assert "secret" not in json.dumps(result)


def test_delta_before_final_is_supported_and_tool_assistant_delta_is_ignored():
    client = module()
    state = client.ReplayState("chat", client.validate_fixture(fixture()), 1.0)
    state.observe(delta("wrong", uuid="tool", tool_call_name="vector_search"), 1.1)
    state.observe(delta("CHAT-7391"), 1.2)
    state.observe(stream("CHAT-7391", True, "stop"), 1.3)
    assert state.finished and state.result(1.3)["complete"]


def test_rag_oracle_is_exact_and_sources_footer_is_strict():
    client = module()
    source = client.validate_fixture(fixture())
    citation = "[doc:" + source["document_id"] + " chunk:34]"
    answer = "ORCHID-7391 " + citation
    assert client.exact_answer("rag", source, answer)
    assert client.exact_answer("rag", source, answer + "\n\nSources:\n- " + citation)
    for wrong in (
        "Not " + answer,
        answer + " but ORCHID-9999",
        answer + " extra",
        answer.replace("chunk:34", "chunk:35"),
        answer + "\nSources:\n- [doc:other chunk:34]",
        "ORCHID-7391",
    ):
        assert not client.exact_answer("rag", source, wrong)
    state = client.ReplayState("rag", source, 1.0)
    state.observe(stream(answer, True, "stop"), 1.1)
    state.observe(delta(answer + "\n\nSources:\n- " + citation), 1.2)
    assert state.result(1.2)["complete"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("collection_id", True),
        ("chunk_ids", [34, 34]),
        ("document_id", "not-a-uuid"),
        ("rag_code", "code with spaces"),
        ("run_id", "a\nsecret"),
        ("password", "secret"),
    ],
)
def test_fixture_rejects_invalid_ids_or_unexpected_secret_fields(field, value):
    source = fixture()
    source[field] = value
    with pytest.raises(ValueError):
        module().validate_fixture(source)


def test_login_token_and_same_origin_redirect_contract():
    client = module()
    assert (
        client.csrf_from_html('<input value="safe-token" name="csrfmiddlewaretoken">')
        == "safe-token"
    )
    assert (
        client.created_conversation("http://localhost:8000", "/chat/ws_convo/42") == 42
    )
    assert (
        client.created_conversation(
            "http://localhost:8000", "http://localhost:8000/chat/ws_convo/43"
        )
        == 43
    )
    for location in (
        "https://evil.test/chat/ws_convo/42",
        "/accounts/login/?next=/chat/ws_convo/42",
        "/chat/ws_convo/42?password=secret",
        "/chat/ws_convo/0",
    ):
        with pytest.raises(ValueError):
            client.created_conversation("http://localhost:8000", location)


@pytest.mark.parametrize(
    "behavior,expected",
    [
        ("success", None),
        ("truncated", "transport_closed"),
        ("error", "server_error"),
        ("timeout", "timeout"),
        ("malformed", "invalid_frame"),
    ],
)
def test_protocol_session_csrf_cookie_transfer_failure_rows_and_scoped_cleanup(
    monkeypatch, behavior, expected
):
    """Simulated transport exercises client boundaries, not real model evidence."""
    client = module()
    paths, sent = [], []

    def respond(request):
        paths.append((request.method, request.url.path))
        if request.url.path == "/accounts/login/" and request.method == "GET":
            return httpx.Response(
                200,
                text='<input name="csrfmiddlewaretoken" value="form-token">',
                headers={"Set-Cookie": "csrftoken=old; Path=/"},
            )
        if request.url.path == "/accounts/login/":
            assert parse_qs(request.content.decode()) == dict(
                login=["principal"],
                password=["credential-secret"],
                csrfmiddlewaretoken=["form-token"],
            )
            return httpx.Response(
                302,
                headers=[
                    ("Location", "/"),
                    ("Set-Cookie", "csrftoken=rotated; Path=/"),
                    ("Set-Cookie", "sessionid=cookie-secret; Path=/"),
                ],
            )
        assert "sessionid=cookie-secret" in request.headers["Cookie"]
        if request.url.path == "/chat/new_ws_convo/":
            return httpx.Response(302, headers={"Location": "/chat/ws_convo/91"})
        if request.url.path == "/chat/delete_ws_convo/91":
            assert request.headers["X-CSRFToken"] == "rotated"
            return httpx.Response(200)
        raise AssertionError("Unexpected endpoint")

    class Socket:
        def __init__(self):
            self.frames = [
                {"conversation": {"messages": []}},
                stream("CHAT-"),
                stream("CHAT-7391", True, "stop"),
            ]
            if behavior == "success":
                self.frames.append(delta("CHAT-7391"))
            elif behavior == "error":
                self.frames.append(
                    {"exception": "credential-secret", "debug_html": "cookie-secret"}
                )
            elif behavior == "malformed":
                self.frames.append("not JSON")

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def recv(self):
            if self.frames:
                frame = self.frames.pop(0)
                return frame if isinstance(frame, str) else json.dumps(frame)
            if behavior == "timeout":
                await asyncio.Future()
            raise client.ReplayError("transport_closed")

        async def send(self, value):
            payload = json.loads(value)
            sent.append(payload)
            if payload["action"] == "select_collections":
                self.frames.append(
                    {
                        "context_selection": {
                            "request_id": payload["request_id"],
                            "selected_collections": payload["collections"],
                        }
                    }
                )

    def connect(uri, *, additional_headers, **options):
        assert uri == "ws://localhost:8080/ws/convo/91/"
        assert options["origin"] == "http://localhost:8080"
        assert "sessionid=cookie-secret" in additional_headers["Cookie"]
        return Socket()

    monkeypatch.setattr(client.websockets, "connect", connect)

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as session:
            runner = client.ReplayClient(
                session,
                "http://localhost:8080",
                client.validate_fixture(fixture()),
                timeout=0.1,
            )
            await runner.login("principal", "credential-secret")
            return await runner.replay("chat")

    row = asyncio.run(exercise())
    assert row["complete"] == (expected is None)
    assert row["error"] == expected
    assert row["cleanup_success"] and row["conversation_id"] == 91
    assert paths[-1] == ("DELETE", "/chat/delete_ws_convo/91")
    assert len([path for method, path in paths if method == "DELETE"]) == 1
    assert sent[0] == client.payload_for("chat", fixture())
    assert row["action_completion_acknowledged"] == (expected is None)
    assert row["background_completion_verified"] is False
    if expected is None:
        assert sent[1]["action"] == "select_collections"
        assert sent[1]["collections"] == []
        assert row["action_completion_seconds"] >= row["total_seconds"]
    else:
        assert len(sent) == 1
    assert all(
        secret not in json.dumps(row)
        for secret in ("credential-secret", "cookie-secret", "rotated")
    )


def test_rejected_creation_redirect_never_connects_or_deletes_any_fixture_id(
    monkeypatch,
):
    client = module()
    requests = []

    def respond(request):
        requests.append((request.method, request.url.path))
        return httpx.Response(
            302, headers={"Location": "https://evil.test/chat/ws_convo/12"}
        )

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as session:
            runner = client.ReplayClient(session, "http://localhost:8080", fixture())
            return await runner.replay("rag")

    row = asyncio.run(exercise())
    assert row["error"] == "chat_redirect_rejected"
    assert row["conversation_id"] is None and row["cleanup_success"] is None
    assert requests == [("GET", "/chat/new_ws_convo/")]


@pytest.mark.parametrize("truncate", [False, True])
def test_real_loopback_websocket_uses_session_cookie_and_records_rag_or_closed_stream(
    truncate,
):
    """Real CPU socket; HTTP setup is simulated and no live application is involved."""
    client = module()
    deleted = []
    source = fixture()
    answer = "ORCHID-7391 [doc:" + source["document_id"] + " chunk:34]"

    def respond(request):
        if request.url.path == "/chat/new_ws_convo/":
            return httpx.Response(302, headers={"Location": "/chat/ws_convo/92"})
        if request.url.path == "/chat/delete_ws_convo/92":
            deleted.append(request.url.path)
            assert request.headers["X-CSRFToken"] == "token"
            return httpx.Response(200)
        raise AssertionError("Unexpected endpoint")

    async def exercise():
        async def handler(ws):
            assert ws.request.path == "/ws/convo/92/"
            assert "sessionid=loopback-session" in ws.request.headers["Cookie"]
            await ws.send(json.dumps({"conversation": {"messages": []}}))
            payload = json.loads(await ws.recv())
            assert payload == client.payload_for("rag", source)
            assert source["rag_code"] not in payload["message"]["content"]
            await ws.send(json.dumps(stream(answer)))
            if not truncate:
                await ws.send(json.dumps(stream(answer, True, "stop")))
                await ws.send(json.dumps(delta(answer)))
                barrier = json.loads(await ws.recv())
                assert barrier["action"] == "select_collections"
                assert barrier["collections"] == [source["collection_id"]]
                await ws.send(
                    json.dumps(
                        {
                            "context_selection": {
                                "request_id": barrier["request_id"],
                                "selected_collections": barrier["collections"],
                            }
                        }
                    )
                )

        async with client.websockets.serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            base = f"http://127.0.0.1:{port}"
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(respond),
                cookies={"sessionid": "loopback-session", "csrftoken": "token"},
            ) as session:
                runner = client.ReplayClient(session, base, source, timeout=2)
                return await runner.replay("rag")

    row = asyncio.run(exercise())
    assert row["complete"] == (not truncate)
    assert row["error"] == ("transport_closed" if truncate else None)
    assert row["output_text"] == answer
    assert row["action_completion_acknowledged"] == (not truncate)
    assert row["cleanup_success"] and deleted == ["/chat/delete_ws_convo/92"]
