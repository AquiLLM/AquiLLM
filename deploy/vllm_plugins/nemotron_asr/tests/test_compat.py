"""Pinned vLLM 0.21 tests for the narrow Nemotron HTTP compatibility hook."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from importlib import reload
from types import SimpleNamespace

import pytest
from starlette.background import BackgroundTask
from starlette.responses import JSONResponse as StarletteJSONResponse

vllm = pytest.importorskip("vllm", reason="compatibility hook requires vLLM")
if getattr(vllm, "__version__", None) != "0.21.0":
    pytest.skip(
        "compatibility tests are pinned to vLLM 0.21.0", allow_module_level=True
    )

from aquillm_vllm_nemotron_asr import compat  # noqa: E402


@dataclass
class Request:
    language: str | None = None
    response_format: str = "json"
    temperature: float | None = None
    use_beam_search: bool = False
    n: int = 1
    stream: bool = False
    to_language: str | None = None
    prompt: str | None = None
    hotwords: str | None = None
    vllm_xargs: dict[str, object] | None = None
    max_completion_tokens: int | None = None
    include_stop_str_in_output: bool = False
    stream_include_usage: bool | None = None
    stream_continuous_usage_stats: bool | None = None
    timestamp_granularities: list[str] | None = None


class Handler:
    task_type = "transcribe"
    model_cls = object

    def __init__(self) -> None:
        self.errors: list[tuple[str, str, int]] = []

    def create_error_response(
        self, message: str, *, err_type: str, status_code: int, param: str | None = None
    ) -> SimpleNamespace:
        self.errors.append((message, err_type, status_code))
        return SimpleNamespace(
            message=message, err_type=err_type, status_code=status_code, param=param
        )


def run(coroutine):
    return asyncio.run(coroutine)


@pytest.fixture(autouse=True)
def installed_hook(monkeypatch: pytest.MonkeyPatch):
    """Install once, while making the preserved private originals replaceable."""
    compat.install_compatibility_hook()
    state = compat._PATCH_STATE
    assert state is not None
    create, preprocess = state.original_create, state.original_preprocess
    yield state
    state.original_create = create
    state.original_preprocess = preprocess
    monkeypatch.undo()


def test_version_gate_accepts_only_vllm_0210(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vllm, "__version__", "0.21.0")
    compat.verify_vllm_compatibility()

    for version in ("0.20.0", "0.21.1", None):
        monkeypatch.setattr(vllm, "__version__", version, raising=False)
        with pytest.raises(RuntimeError, match="0.21.0"):
            compat.verify_vllm_compatibility()


def test_version_gate_rejects_v2_model_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm import envs

    monkeypatch.setattr(envs, "VLLM_USE_V2_MODEL_RUNNER", True)

    with pytest.raises(RuntimeError, match="VLLM_USE_V2_MODEL_RUNNER=0"):
        compat.verify_vllm_compatibility()


def test_install_is_reentrant_and_preserves_original_descriptors() -> None:
    from vllm.entrypoints.openai.speech_to_text import api_router
    from vllm.entrypoints.openai.speech_to_text.speech_to_text import (
        OpenAISpeechToText,
    )

    state = compat._PATCH_STATE
    assert state is not None
    create_wrapper = state.wrapped_create
    preprocess_wrapper = state.wrapped_preprocess
    json_response_wrapper = state.wrapped_json_response
    marked_response_type = state.marked_response_type

    compat.install_compatibility_hook()

    assert compat._PATCH_STATE is state
    assert state.wrapped_create is create_wrapper
    assert state.wrapped_preprocess is preprocess_wrapper
    assert state.wrapped_json_response is json_response_wrapper
    assert state.marked_response_type is marked_response_type
    assert state.original_create is not create_wrapper
    assert state.original_preprocess is not preprocess_wrapper
    assert OpenAISpeechToText._create_speech_to_text is create_wrapper
    assert OpenAISpeechToText._preprocess_speech_to_text is preprocess_wrapper
    assert api_router.JSONResponse is json_response_wrapper
    assert create_wrapper.__wrapped__ is state.original_create
    assert preprocess_wrapper.__wrapped__ is state.original_preprocess
    assert issubclass(json_response_wrapper, state.original_json_response)


def test_install_rejects_a_partially_replaced_existing_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.entrypoints.openai.speech_to_text.speech_to_text import (
        OpenAISpeechToText,
    )

    state = compat._PATCH_STATE
    assert state is not None
    monkeypatch.setattr(
        OpenAISpeechToText, "_create_speech_to_text", state.original_create
    )

    with pytest.raises(RuntimeError, match="partially replaced"):
        compat.install_compatibility_hook()


def test_install_rejects_a_partially_replaced_json_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.entrypoints.openai.speech_to_text import api_router

    state = compat._PATCH_STATE
    assert state is not None
    monkeypatch.setattr(api_router, "JSONResponse", state.original_json_response)

    with pytest.raises(RuntimeError, match="partially replaced"):
        compat.install_compatibility_hook()


def test_reload_reuses_the_existing_wrapper_state_without_stacking() -> None:
    import aquillm_vllm_nemotron_asr as plugin
    from vllm.entrypoints.openai.speech_to_text import api_router
    from vllm.entrypoints.openai.speech_to_text.speech_to_text import (
        OpenAISpeechToText,
    )

    original_state = compat._PATCH_STATE
    assert original_state is not None
    original_create = original_state.original_create
    original_preprocess = original_state.original_preprocess
    original_create_wrapper = original_state.wrapped_create
    original_preprocess_wrapper = original_state.wrapped_preprocess
    original_json_response = original_state.original_json_response
    original_json_response_wrapper = original_state.wrapped_json_response
    original_marked_response_type = original_state.marked_response_type

    reloaded_compat = reload(compat)
    reloaded_plugin = reload(plugin)
    reloaded_plugin.register()

    assert reloaded_compat._PATCH_STATE is original_state
    assert OpenAISpeechToText._create_speech_to_text is original_create_wrapper
    assert OpenAISpeechToText._preprocess_speech_to_text is original_preprocess_wrapper
    assert api_router.JSONResponse is original_json_response_wrapper
    assert original_state.original_create is original_create
    assert original_state.original_preprocess is original_preprocess
    assert original_state.original_json_response is original_json_response
    assert original_state.marked_response_type is original_marked_response_type


def test_non_nemotron_create_is_an_exact_original_passthrough(installed_hook) -> None:
    called: list[tuple[tuple[object, ...], dict[str, object]]] = []
    result = object()

    async def original(handler, *args, **kwargs):
        called.append((args, kwargs))
        return result

    installed_hook.original_create = original
    handler = Handler()
    args = (b"audio", Request(), object(), object(), object())

    actual = run(installed_hook.wrapped_create(handler, *args, marker="unchanged"))

    assert actual is result
    assert called == [(args, {"marker": "unchanged"})]


def test_non_nemotron_create_preserves_original_exception(installed_hook) -> None:
    expected = RuntimeError("original create failure")

    async def original(*args, **kwargs):
        raise expected

    installed_hook.original_create = original

    with pytest.raises(RuntimeError) as raised:
        run(
            installed_hook.wrapped_create(
                Handler(), b"audio", Request(), object(), object(), object()
            )
        )

    assert raised.value is expected


def test_non_nemotron_preprocess_preserves_original_exception(installed_hook) -> None:
    expected = RuntimeError("original preprocess failure")

    async def original(*args, **kwargs):
        raise expected

    installed_hook.original_preprocess = original

    with pytest.raises(RuntimeError) as raised:
        run(installed_hook.wrapped_preprocess(Handler(), Request(), b"audio", "id"))

    assert raised.value is expected


def test_non_nemotron_preprocess_is_an_exact_original_passthrough(
    installed_hook,
) -> None:
    called: list[tuple[tuple[object, ...], dict[str, object]]] = []
    result = object()

    async def original(handler, *args, **kwargs):
        called.append((args, kwargs))
        return result

    installed_hook.original_preprocess = original
    args = (Request(), b"audio", "request-id")

    actual = run(installed_hook.wrapped_preprocess(Handler(), *args, marker="same"))

    assert actual is result
    assert called == [(args, {"marker": "same"})]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature", 0.5),
        ("use_beam_search", True),
        ("n", 2),
        ("prompt", "words"),
        ("hotwords", "words"),
        ("response_format", "verbose_json"),
        ("stream", True),
        ("vllm_xargs", {"x": 1}),
        ("max_completion_tokens", 2),
        ("include_stop_str_in_output", True),
        ("timestamp_granularities", ["word"]),
    ],
)
def test_nemotron_option_errors_are_stable_400s_before_generation(
    monkeypatch: pytest.MonkeyPatch, installed_hook, field: str, value: object
) -> None:
    monkeypatch.setattr(compat, "_is_nemotron_handler", lambda handler: True)
    called = False

    async def original(*args, **kwargs):
        nonlocal called
        called = True
        return object()

    installed_hook.original_create = original
    request = Request()
    setattr(request, field, value)
    handler = Handler()

    response = run(
        installed_hook.wrapped_create(
            handler, b"audio", request, object(), object(), object()
        )
    )

    assert response.status_code == 400
    assert response.err_type == "BadRequestError"
    assert field in response.message
    assert response.param == field
    assert called is False


def test_translation_handler_returns_stable_400_before_generation(
    monkeypatch: pytest.MonkeyPatch, installed_hook
) -> None:
    from vllm.entrypoints.openai.speech_to_text import api_router

    monkeypatch.setattr(compat, "_is_nemotron_handler", lambda handler: True)
    handler = Handler()
    handler.task_type = "translate"
    called = False

    async def original(*args, **kwargs):
        nonlocal called
        called = True

    installed_hook.original_create = original
    response = run(
        installed_hook.wrapped_create(
            handler, b"audio", Request(), object(), object(), object()
        )
    )

    assert response.status_code == 400
    assert "task_type" in response.message
    assert called is False
    rendered = api_router.JSONResponse(
        content=response.__dict__, status_code=response.status_code
    )
    assert rendered.media_type == "application/json"
    assert b'"err_type":"BadRequestError"' in rendered.body


def test_nemotron_validation_uses_vllms_real_error_response_surface(
    monkeypatch: pytest.MonkeyPatch, installed_hook
) -> None:
    from vllm.entrypoints.openai.engine.serving import create_error_response
    from vllm.entrypoints.openai.speech_to_text import api_router

    monkeypatch.setattr(compat, "_is_nemotron_handler", lambda handler: True)
    handler = Handler()
    handler.create_error_response = create_error_response

    response = run(
        installed_hook.wrapped_create(
            handler,
            b"audio",
            Request(temperature=0.5),
            object(),
            object(),
            object(),
        )
    )

    assert response.error.code == 400
    assert response.error.type == "BadRequestError"
    assert "temperature" in response.error.message
    assert response.error.param == "temperature"
    rendered = api_router.JSONResponse(
        content=response.model_dump(), status_code=response.error.code
    )
    assert rendered.media_type == "application/json"
    assert b'"type":"BadRequestError"' in rendered.body


def test_outer_wrapper_converts_preprocess_duration_error_before_generation(
    monkeypatch: pytest.MonkeyPatch, installed_hook
) -> None:
    """The stock create path dynamically reaches the patched preprocess method."""
    from vllm.entrypoints.openai.engine.serving import create_error_response

    monkeypatch.setattr(compat, "_is_nemotron_handler", lambda handler: True)
    generation_reached = False

    async def original_preprocess(*args, **kwargs):
        return ([{"resampled": True}], 391.0)

    async def stock_like_original_create(
        handler, audio_data, request, raw_request, response_class, stream_method
    ):
        nonlocal generation_reached
        await handler._preprocess_speech_to_text(request, audio_data, "request-id")
        generation_reached = True
        return object()

    installed_hook.original_preprocess = original_preprocess
    installed_hook.original_create = stock_like_original_create
    handler = Handler()
    handler.create_error_response = create_error_response

    async def call_patched_preprocess(*args, **kwargs):
        return await installed_hook.wrapped_preprocess(handler, *args, **kwargs)

    handler._preprocess_speech_to_text = call_patched_preprocess
    response = run(
        installed_hook.wrapped_create(
            handler, b"audio", Request(), object(), object(), object()
        )
    )

    assert response.error.code == 400
    assert response.error.type == "BadRequestError"
    assert response.error.param == "duration_s"
    assert "duration_s" in response.error.message
    assert generation_reached is False


@pytest.mark.parametrize("duration_s", [389.0, 390.0])
def test_preprocess_keeps_valid_resampled_results_unchanged(
    monkeypatch: pytest.MonkeyPatch, installed_hook, duration_s: float
) -> None:
    monkeypatch.setattr(compat, "_is_nemotron_handler", lambda handler: True)
    result = ([{"resampled": True}], duration_s)
    calls = 0

    async def original(*args, **kwargs):
        nonlocal calls
        calls += 1
        return result

    installed_hook.original_preprocess = original

    assert (
        run(installed_hook.wrapped_preprocess(Handler(), Request(), b"x", "id"))
        is result
    )
    assert calls == 1


def test_outer_validation_does_not_replace_the_language_seen_by_vllm(
    monkeypatch: pytest.MonkeyPatch, installed_hook
) -> None:
    monkeypatch.setattr(compat, "_is_nemotron_handler", lambda handler: True)
    request = Request(language="en")
    seen: list[str | None] = []

    async def original(handler, *args, **kwargs):
        seen.append(args[1].language)
        return object()

    installed_hook.original_create = original

    run(
        installed_hook.wrapped_create(
            Handler(), b"audio", request, object(), object(), object()
        )
    )

    assert seen == ["en"]
    assert request.language == "en"


def test_preprocess_rejects_391s_after_the_original_resampled_duration(
    monkeypatch: pytest.MonkeyPatch, installed_hook
) -> None:
    monkeypatch.setattr(compat, "_is_nemotron_handler", lambda handler: True)
    calls = 0

    async def original(*args, **kwargs):
        nonlocal calls
        calls += 1
        return ([{"resampled": True}], 391.0)

    installed_hook.original_preprocess = original

    with pytest.raises(compat.RequestValidationError, match="duration_s"):
        run(installed_hook.wrapped_preprocess(Handler(), Request(), b"x", "id"))

    assert calls == 1


def _transcription_response(text: str):
    from vllm.entrypoints.openai.speech_to_text.protocol import (
        TranscriptionResponse,
        TranscriptionUsageAudio,
    )

    return TranscriptionResponse(text=text, usage=TranscriptionUsageAudio(seconds=7))


def _render_vllm_response(result, **kwargs):
    """Exercise the same dynamic JSONResponse boundary as vLLM's route."""
    from vllm.entrypoints.openai.speech_to_text import api_router

    return api_router.JSONResponse(content=result.model_dump(), **kwargs)


@pytest.mark.parametrize("text", ["", "café ☕", 'say "hello"', "line 1\nline 2\n"])
def test_nemotron_text_response_is_byte_exact_plain_text(
    monkeypatch: pytest.MonkeyPatch, installed_hook, text: str
) -> None:
    monkeypatch.setattr(compat, "_is_nemotron_handler", lambda handler: True)
    original_result = _transcription_response(text)

    async def original(*args, **kwargs):
        return original_result

    installed_hook.original_create = original
    result = run(
        installed_hook.wrapped_create(
            Handler(),
            b"audio",
            Request(response_format="text"),
            object(),
            object(),
            object(),
        )
    )
    background = BackgroundTask(lambda: None)
    response = _render_vllm_response(
        result,
        status_code=201,
        headers={"x-test": "preserved"},
        background=background,
    )

    assert isinstance(result, type(original_result))
    assert result.text == text
    assert result.usage == original_result.usage
    assert isinstance(result.model_dump(), dict)
    assert response.body == text.encode("utf-8")
    assert response.media_type == "text/plain"
    assert response.headers["content-type"] == "text/plain; charset=utf-8"
    assert response.headers["x-test"] == "preserved"
    assert response.status_code == 201
    assert response.background is background
    assert isinstance(response, installed_hook.original_json_response)


def test_marked_model_dump_remains_an_ordinary_json_serializable_dict(
    monkeypatch: pytest.MonkeyPatch, installed_hook
) -> None:
    import json

    monkeypatch.setattr(compat, "_is_nemotron_handler", lambda handler: True)

    async def original(*args, **kwargs):
        return _transcription_response("plain")

    installed_hook.original_create = original
    result = run(
        installed_hook.wrapped_create(
            Handler(),
            b"audio",
            Request(response_format="text"),
            object(),
            object(),
            object(),
        )
    )
    payload = result.model_dump()

    assert json.loads(json.dumps(payload)) == {
        "text": "plain",
        "usage": {"type": "duration", "seconds": 7},
    }


def test_nemotron_json_response_remains_stock_json_with_usage(
    monkeypatch: pytest.MonkeyPatch, installed_hook
) -> None:
    monkeypatch.setattr(compat, "_is_nemotron_handler", lambda handler: True)
    original_result = _transcription_response("hello")

    async def original(*args, **kwargs):
        return original_result

    installed_hook.original_create = original
    result = run(
        installed_hook.wrapped_create(
            Handler(),
            b"audio",
            Request(response_format="json"),
            object(),
            object(),
            object(),
        )
    )
    response = _render_vllm_response(result)

    assert result is original_result
    assert response.body == b'{"text":"hello","usage":{"type":"duration","seconds":7}}'
    assert response.media_type == "application/json"


@pytest.mark.parametrize("response_format", ["text", "json"])
def test_non_nemotron_responses_remain_stock_json(
    installed_hook, response_format: str
) -> None:
    original_result = _transcription_response("unchanged")

    async def original(*args, **kwargs):
        return original_result

    installed_hook.original_create = original
    result = run(
        installed_hook.wrapped_create(
            Handler(),
            b"audio",
            Request(response_format=response_format),
            object(),
            object(),
            object(),
        )
    )
    response = _render_vllm_response(result)

    assert result is original_result
    assert response.body.startswith(b'{"text":"unchanged"')
    assert response.media_type == "application/json"


def test_nemotron_text_preserves_original_exception_and_cancellation(
    monkeypatch: pytest.MonkeyPatch, installed_hook
) -> None:
    monkeypatch.setattr(compat, "_is_nemotron_handler", lambda handler: True)

    async def raises_runtime(*args, **kwargs):
        raise RuntimeError("generation failed")

    installed_hook.original_create = raises_runtime
    with pytest.raises(RuntimeError, match="generation failed"):
        run(
            installed_hook.wrapped_create(
                Handler(),
                b"audio",
                Request(response_format="text"),
                object(),
                object(),
                object(),
            )
        )

    async def raises_cancelled(*args, **kwargs):
        raise asyncio.CancelledError

    installed_hook.original_create = raises_cancelled
    with pytest.raises(asyncio.CancelledError):
        run(
            installed_hook.wrapped_create(
                Handler(),
                b"audio",
                Request(response_format="text"),
                object(),
                object(),
                object(),
            )
        )


def test_error_payloads_remain_json(installed_hook) -> None:
    from vllm.entrypoints.openai.speech_to_text import api_router

    payload = {"error": {"message": "bad request", "code": 400}}
    response = api_router.JSONResponse(content=payload, status_code=400)

    assert response.body == b'{"error":{"message":"bad request","code":400}}'
    assert response.media_type == "application/json"
    assert response.status_code == 400


def test_concurrent_requests_do_not_cross_contaminate_plain_text_marker(
    monkeypatch: pytest.MonkeyPatch, installed_hook
) -> None:
    class NemotronHandler(Handler):
        pass

    nemotron_handler = NemotronHandler()
    other_handler = Handler()
    monkeypatch.setattr(
        compat,
        "_is_nemotron_handler",
        lambda handler: isinstance(handler, NemotronHandler),
    )

    async def original(handler, *args, **kwargs):
        await asyncio.sleep(0)
        return _transcription_response(args[1].response_format)

    installed_hook.original_create = original

    async def exercise():
        calls = [
            installed_hook.wrapped_create(
                nemotron_handler,
                b"a",
                Request(response_format="text"),
                object(),
                object(),
                object(),
            ),
            installed_hook.wrapped_create(
                nemotron_handler,
                b"a",
                Request(response_format="json"),
                object(),
                object(),
                object(),
            ),
            installed_hook.wrapped_create(
                other_handler,
                b"a",
                Request(response_format="text"),
                object(),
                object(),
                object(),
            ),
            installed_hook.wrapped_create(
                other_handler,
                b"a",
                Request(response_format="json"),
                object(),
                object(),
                object(),
            ),
        ]
        return await asyncio.gather(*calls)

    results = run(exercise())
    responses = [_render_vllm_response(result) for result in results]

    assert [response.media_type for response in responses] == [
        "text/plain",
        "application/json",
        "application/json",
        "application/json",
    ]
    assert responses[0].body == b"text"
    assert all(isinstance(response, StarletteJSONResponse) for response in responses)
