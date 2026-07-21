"""Pinned vLLM 0.21 tests for the narrow Nemotron HTTP compatibility hook."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from importlib import reload
from types import SimpleNamespace

import pytest

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
    from vllm.entrypoints.openai.speech_to_text.speech_to_text import (
        OpenAISpeechToText,
    )

    state = compat._PATCH_STATE
    assert state is not None
    create_wrapper = state.wrapped_create
    preprocess_wrapper = state.wrapped_preprocess

    compat.install_compatibility_hook()

    assert compat._PATCH_STATE is state
    assert state.wrapped_create is create_wrapper
    assert state.wrapped_preprocess is preprocess_wrapper
    assert state.original_create is not create_wrapper
    assert state.original_preprocess is not preprocess_wrapper
    assert OpenAISpeechToText._create_speech_to_text is create_wrapper
    assert OpenAISpeechToText._preprocess_speech_to_text is preprocess_wrapper
    assert create_wrapper.__wrapped__ is state.original_create
    assert preprocess_wrapper.__wrapped__ is state.original_preprocess


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


def test_reload_reuses_the_existing_wrapper_state_without_stacking() -> None:
    import aquillm_vllm_nemotron_asr as plugin
    from vllm.entrypoints.openai.speech_to_text.speech_to_text import (
        OpenAISpeechToText,
    )

    original_state = compat._PATCH_STATE
    assert original_state is not None
    original_create = original_state.original_create
    original_preprocess = original_state.original_preprocess
    original_create_wrapper = original_state.wrapped_create
    original_preprocess_wrapper = original_state.wrapped_preprocess

    reloaded_compat = reload(compat)
    reloaded_plugin = reload(plugin)
    reloaded_plugin.register()

    assert reloaded_compat._PATCH_STATE is original_state
    assert OpenAISpeechToText._create_speech_to_text is original_create_wrapper
    assert OpenAISpeechToText._preprocess_speech_to_text is original_preprocess_wrapper
    assert original_state.original_create is original_create
    assert original_state.original_preprocess is original_preprocess


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


def test_nemotron_validation_uses_vllms_real_error_response_surface(
    monkeypatch: pytest.MonkeyPatch, installed_hook
) -> None:
    from vllm.entrypoints.openai.engine.serving import create_error_response

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
