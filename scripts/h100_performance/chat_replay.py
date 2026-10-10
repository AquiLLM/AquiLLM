#!/usr/bin/env python
"""Live authenticated chat/RAG replay. Requires installed httpx and websockets.

Fixture JSON: collection_id, document_id (UUID), chunk_ids, run_id, chat_code,
rag_code, expected_model. The operator must ingest and verify these sources first.
Credentials: AQUILLM_REPLAY_USERNAME / AQUILLM_REPLAY_PASSWORD, or one stdin JSON
object with username/password when --credentials-stdin is set. Never save cookies.

Fresh chats are deleted by default; --keep-conversations retains them for operator
inspection. This client cannot verify stored Message.model, provider URL, source
entailment, or a real model dispatch: verify those separately before claiming an
application replay pass. CPU tests are protocol tests, not live model evidence.
The answer timer ends at its persisted delta. A subsequent context acknowledgement
barrier waits for append to finish enqueueing background work; job completion and
quiescence still require operator verification.
"""

import argparse
import asyncio
import hashlib
import inspect
import json
import os
import re
import sys
import uuid
from datetime import UTC, datetime
from html.parser import HTMLParser
from http.cookies import SimpleCookie
from pathlib import Path
from time import perf_counter
from urllib.parse import urljoin, urlsplit

import httpx
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException


class ReplayError(Exception):
    """Only fixed, content-free codes may leave the network boundary."""


def normalized(text):
    return " ".join(text.split())


def validate_fixture(value):
    keys = {
        "collection_id",
        "document_id",
        "chunk_ids",
        "run_id",
        "chat_code",
        "rag_code",
        "expected_model",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("invalid_fixture")
    if type(value["collection_id"]) is not int or value["collection_id"] <= 0:
        raise ValueError("invalid_fixture")
    chunks = value["chunk_ids"]
    if (
        not isinstance(chunks, list)
        or not 1 <= len(chunks) <= 32
        or any(type(x) is not int or x <= 0 for x in chunks)
        or len(set(chunks)) != len(chunks)
    ):
        raise ValueError("invalid_fixture")
    if not isinstance(value["document_id"], str):
        raise ValueError("invalid_fixture")
    document = str(uuid.UUID(value["document_id"]))
    if document != value["document_id"]:
        raise ValueError("invalid_fixture")
    for key in ("chat_code", "rag_code"):
        if not isinstance(value[key], str) or not re.fullmatch(
            r"[A-Z0-9-]{1,64}", value[key]
        ):
            raise ValueError("invalid_fixture")
    if not isinstance(value["run_id"], str) or not re.fullmatch(
        r"[A-Za-z0-9_-]{1,64}", value["run_id"]
    ):
        raise ValueError("invalid_fixture")
    if not isinstance(value["expected_model"], str) or not re.fullmatch(
        r"[A-Za-z0-9._:/-]{1,200}", value["expected_model"]
    ):
        raise ValueError("invalid_fixture")
    return dict(value)


def exact_answer(kind, fixture, text):
    """Whole-answer equality, allowing only the application's exact Sources footer."""
    if not isinstance(text, str):
        return False
    if kind == "chat":
        return normalized(text) == fixture["chat_code"]
    for chunk in fixture["chunk_ids"]:
        citation = f"[doc:{fixture['document_id']} chunk:{chunk}]"
        answer = fixture["rag_code"] + " " + citation
        if normalized(text) in (
            answer,
            normalized(answer + "\n\nSources:\n- " + citation),
        ):
            return True
    return False


class ReplayState:
    def __init__(self, kind, fixture, started):
        self.kind, self.fixture, self.started = kind, fixture, started
        self.first_visible_at = None
        self.streams, self.persisted = {}, {}
        self.error = None

    def observe(self, frame, now):
        if not isinstance(frame, dict):
            raise ReplayError("invalid_frame")
        if "exception" in frame or "error" in frame:
            self.error = "server_error"
            return
        stream = frame.get("stream")
        if (
            isinstance(stream, dict)
            and stream.get("role") == "assistant"
            and not stream.get("tool_call_name")
            and not stream.get("tool_calls")
            and not stream.get("tool_call_id")
        ):
            text = stream.get("content", "")
            if not isinstance(text, str):
                raise ReplayError("invalid_frame")
            if text.strip():
                identity = stream.get("message_uuid")
                if not isinstance(identity, str) or not identity:
                    raise ReplayError("invalid_frame")
                if self.first_visible_at is None:
                    self.first_visible_at = now
                record = self.streams.setdefault(identity, {})
                record["text"] = text  # The application sends cumulative text.
                if stream.get("done") is True:
                    record.update(done=True, finish=stream.get("stop_reason"), at=now)
                    if record["finish"] != "stop":
                        self.error = "invalid_finish_reason"
        delta = frame.get("delta")
        if isinstance(delta, dict):
            messages = delta.get("messages", [])
            if not isinstance(messages, list):
                raise ReplayError("invalid_frame")
            for message in messages:
                if not isinstance(message, dict):
                    raise ReplayError("invalid_frame")
                if message.get("role") != "assistant" or message.get("tool_call_name"):
                    continue
                identity, text = message.get("message_uuid"), message.get("content", "")
                if not isinstance(text, str):
                    raise ReplayError("invalid_frame")
                if text.strip() and isinstance(identity, str) and identity:
                    self.persisted[identity] = dict(text=text, at=now)

    @property
    def finished(self):
        return self.error is not None or any(
            row.get("done") and identity in self.persisted
            for identity, row in self.streams.items()
        )

    def result(self, ended, error=None):
        final = [
            (identity, row) for identity, row in self.streams.items() if row.get("done")
        ]
        identity, stream = (
            final[-1] if final else (None, next(reversed(self.streams.values()), {}))
        )
        persisted = self.persisted.get(identity, {})
        failure = self.error or error
        if failure is None:
            if not final:
                failure = "missing_stream_done"
            elif not persisted:
                failure = "missing_persisted_delta"
            elif (
                not exact_answer(self.kind, self.fixture, stream["text"])
                or not exact_answer(self.kind, self.fixture, persisted["text"])
                or normalized(stream["text"]).split(" Sources:")[0]
                != normalized(persisted["text"]).split(" Sources:")[0]
            ):
                failure = "answer_mismatch"
        output = persisted.get("text", stream.get("text", ""))
        return dict(
            complete=failure is None,
            error=failure,
            finish_reason=stream.get("finish"),
            stream_done=bool(final),
            persisted_delta=bool(persisted),
            message_uuid=identity,
            output_text=output,
            stream_output_text=stream.get("text", ""),
            output_sha256=hashlib.sha256(output.encode()).hexdigest(),
            visible_ttft_seconds=self.first_visible_at - self.started
            if self.first_visible_at is not None
            else None,
            final_seconds=stream["at"] - self.started if final else None,
            persisted_seconds=persisted["at"] - self.started if persisted else None,
            total_seconds=ended - self.started,
            expected_model=self.fixture["expected_model"],
            model_verified=False,
            route_verified=False,
            source_entailment_verified=False,
        )


class TokenParser(HTMLParser):
    token = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "input" and attrs.get("name") == "csrfmiddlewaretoken":
            self.token = attrs.get("value")


def csrf_from_html(html):
    parser = TokenParser()
    parser.feed(html)
    if not parser.token:
        raise ReplayError("missing_csrf")
    return parser.token


def checked_base(value):
    parsed = urlsplit(value)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in ("", "/")
    ):
        raise ValueError("invalid_base_url")
    return value.rstrip("/")


def created_conversation(base, location):
    target = urlsplit(urljoin(base + "/", location))
    origin = urlsplit(base)
    if (
        (target.scheme, target.netloc) != (origin.scheme, origin.netloc)
        or target.query
        or target.fragment
    ):
        raise ValueError("invalid_chat_redirect")
    match = re.fullmatch(r"/chat/ws_convo/([1-9][0-9]*)", target.path)
    if not match:
        raise ValueError("invalid_chat_redirect")
    return int(match[1])


def payload_for(kind, fixture):
    content = (
        f"Reply with exactly {fixture['chat_code']} and nothing else."
        if kind == "chat"
        else f"Search the selected documents. For replay {fixture['run_id']}, "
        "what is the amber calibration code? Return only the code followed by "
        "one exact chunk citation, with no other prose."
    )
    return dict(
        action="append",
        message=dict(role="user", content=content),
        collections=[] if kind == "chat" else [fixture["collection_id"]],
    )


class ReplayClient:
    def __init__(self, session, base, fixture, timeout=180, cleanup=True):
        self.session, self.base = session, checked_base(base)
        self.fixture, self.timeout, self.cleanup = fixture, timeout, cleanup

    async def login(self, username, password):
        endpoint = self.base + "/accounts/login/"
        response = await self.session.get(endpoint, follow_redirects=False)
        if response.status_code != 200:
            raise ReplayError("login_form_failed")
        token = csrf_from_html(response.text)
        response = await self.session.post(
            endpoint,
            data=dict(login=username, password=password, csrfmiddlewaretoken=token),
            headers={"Referer": endpoint},
            follow_redirects=False,
        )
        if response.status_code not in (302, 303):
            raise ReplayError("login_failed")
        target = urlsplit(urljoin(endpoint, response.headers.get("Location", "")))
        origin = urlsplit(self.base)
        if (target.scheme, target.netloc) != (origin.scheme, origin.netloc):
            raise ReplayError("login_redirect_rejected")

    async def frame(self, ws):
        try:
            message = await ws.recv()
        except ConnectionClosed:
            raise ReplayError("transport_closed") from None
        if isinstance(message, str):
            try:
                return json.loads(message)
            except (ValueError, TypeError):
                raise ReplayError("invalid_frame") from None
        raise ReplayError("invalid_frame")

    async def delete(self, identity):
        # Only IDs returned by this runner's successful creation request are used.
        cookies = SimpleCookie()
        cookies.load(
            self.session.build_request("GET", self.base + "/").headers.get("Cookie", "")
        )
        cookie = cookies.get("csrftoken")
        if cookie is None:
            raise ReplayError("cleanup_missing_csrf")
        response = await self.session.delete(
            self.base + f"/chat/delete_ws_convo/{identity}",
            headers={"X-CSRFToken": cookie.value, "Referer": self.base + "/"},
            follow_redirects=False,
        )
        if response.status_code != 200:
            raise ReplayError("cleanup_failed")

    async def replay(self, kind):
        identity, state, failure = None, None, None
        answer_ended, action_completed = None, None
        setup_started = perf_counter()
        payload = payload_for(kind, self.fixture)
        try:
            async with asyncio.timeout(self.timeout):
                response = await self.session.get(
                    self.base + "/chat/new_ws_convo/", follow_redirects=False
                )
                if response.status_code not in (302, 303):
                    raise ReplayError("chat_creation_failed")
                try:
                    identity = created_conversation(
                        self.base, response.headers.get("Location", "")
                    )
                except ValueError:
                    raise ReplayError("chat_redirect_rejected") from None
                websocket_url = self.base.replace("https://", "wss://", 1).replace(
                    "http://", "ws://", 1
                )
                headers = {
                    "Cookie": self.session.build_request(
                        "GET", self.base + f"/ws/convo/{identity}/"
                    ).headers.get("Cookie", "")
                }
                options = dict(
                    origin=self.base,
                    max_size=1024 * 1024,
                    open_timeout=self.timeout,
                    close_timeout=1,
                )
                parameters = inspect.signature(websockets.connect).parameters
                options[
                    "additional_headers"
                    if "additional_headers" in parameters
                    else "extra_headers"
                ] = headers
                if "proxy" in parameters:
                    options["proxy"] = None
                async with websockets.connect(
                    websocket_url + f"/ws/convo/{identity}/", **options
                ) as ws:
                    while True:
                        frame = await self.frame(ws)
                        if (
                            not isinstance(frame, dict)
                            or "exception" in frame
                            or "error" in frame
                        ):
                            raise ReplayError("connect_failed")
                        if isinstance(frame.get("conversation"), dict):
                            if frame["conversation"].get("messages") != []:
                                raise ReplayError("chat_not_fresh")
                            break
                    state = ReplayState(kind, self.fixture, perf_counter())
                    await ws.send(json.dumps(payload))
                    while not state.finished:
                        state.observe(await self.frame(ws), perf_counter())
                    answer_ended = perf_counter()
                    if state.result(answer_ended)["complete"]:
                        request_id = uuid.uuid4().hex
                        await ws.send(
                            json.dumps(
                                dict(
                                    action="select_collections",
                                    collections=payload["collections"],
                                    request_id=request_id,
                                )
                            )
                        )
                        while True:
                            frame = await self.frame(ws)
                            if (
                                not isinstance(frame, dict)
                                or "exception" in frame
                                or "error" in frame
                            ):
                                raise ReplayError("action_completion_failed")
                            if "context_selection_error" in frame:
                                raise ReplayError("action_completion_failed")
                            ack = frame.get("context_selection")
                            if (
                                isinstance(ack, dict)
                                and ack.get("request_id") == request_id
                            ):
                                if (
                                    ack.get("selected_collections")
                                    != payload["collections"]
                                ):
                                    raise ReplayError("action_completion_failed")
                                action_completed = perf_counter()
                                break
        except TimeoutError:
            failure = "timeout"
        except ReplayError as exc:
            failure = str(exc)
        except (httpx.HTTPError, WebSocketException, OSError):
            failure = "transport_error"
        except (ValueError, TypeError, KeyError):
            failure = "invalid_frame"
        ended = answer_ended if answer_ended is not None else perf_counter()
        if state is None:
            state = ReplayState(kind, self.fixture, setup_started)
        result = state.result(ended, failure)
        result.update(
            kind=kind,
            conversation_id=identity,
            cleanup_requested=self.cleanup,
            cleanup_success=None,
            cleanup_error=None,
            action_completion_acknowledged=action_completed is not None,
            action_completion_seconds=action_completed - state.started
            if action_completed is not None
            else None,
            background_completion_verified=False,
            input_sha256=hashlib.sha256(
                json.dumps(payload, sort_keys=True).encode()
            ).hexdigest(),
        )
        if self.cleanup and identity is not None:
            try:
                async with asyncio.timeout(min(self.timeout, 30)):
                    await self.delete(identity)
                result["cleanup_success"] = True
            except (TimeoutError, ReplayError, httpx.HTTPError, OSError):
                result.update(cleanup_success=False, cleanup_error="cleanup_failed")
        return result


async def execute(args, fixture, credentials):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    failed = False
    fixture_digest = hashlib.sha256(
        json.dumps(fixture, sort_keys=True).encode()
    ).hexdigest()
    async with httpx.AsyncClient(timeout=args.timeout, trust_env=False) as session:
        client = ReplayClient(
            session, args.base_url, fixture, args.timeout, not args.keep_conversations
        )
        with args.output.open("a", encoding="utf-8") as output:
            try:
                async with asyncio.timeout(args.timeout):
                    await client.login(*credentials)
            except (ReplayError, httpx.HTTPError, TimeoutError, OSError):
                row = dict(
                    kind="setup",
                    label=args.label,
                    complete=False,
                    error="authentication_failed",
                )
                output.write(json.dumps(row) + "\n")
                return 1
            for repeat in range(-args.warmup, args.repeats):
                for kind in ("chat", "rag"):
                    row = await client.replay(kind)
                    row.update(
                        label=args.label,
                        repeat=repeat,
                        warmup=repeat < 0,
                        fixture_sha256=fixture_digest,
                        oracle="exact-chat-rag-v1",
                        captured_at=datetime.now(UTC).isoformat(),
                    )
                    output.write(json.dumps(row) + "\n")
                    output.flush()
                    print(json.dumps(row), flush=True)
                    failed |= not row["complete"] or row["cleanup_success"] is False
    return int(failed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--credentials-stdin", action="store_true")
    parser.add_argument("--keep-conversations", action="store_true")
    args = parser.parse_args()
    try:
        checked_base(args.base_url)
        if (
            not 1 <= args.repeats <= 100
            or not 0 <= args.warmup <= 10
            or not 0 < args.timeout <= 1800
        ):
            raise ValueError("invalid_arguments")
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", args.label):
            raise ValueError("invalid_arguments")
        if args.output.resolve() == args.fixture.resolve():
            raise ValueError("invalid_arguments")
        fixture = validate_fixture(json.loads(args.fixture.read_text(encoding="utf-8")))
        if args.credentials_stdin:
            value = json.loads(sys.stdin.readline())
            if not isinstance(value, dict) or set(value) != {"username", "password"}:
                raise ValueError("invalid_credentials")
            credentials = value["username"], value["password"]
        else:
            credentials = (
                os.environ.get("AQUILLM_REPLAY_USERNAME"),
                os.environ.get("AQUILLM_REPLAY_PASSWORD"),
            )
        if any(not isinstance(value, str) or not value for value in credentials):
            raise ValueError("invalid_credentials")
    except (ValueError, TypeError, OSError):
        print(
            '{"complete":false,"error":"invalid_arguments_or_credentials"}',
            file=sys.stderr,
        )
        return 2
    return asyncio.run(execute(args, fixture, credentials))


if __name__ == "__main__":
    raise SystemExit(main())
