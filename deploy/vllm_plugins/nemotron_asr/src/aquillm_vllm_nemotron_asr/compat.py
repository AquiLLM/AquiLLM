"""Narrow, version-pinned vLLM HTTP validation for Nemotron ASR.

This module intentionally reaches into two private vLLM 0.21 methods.  Keep
the patch small and fail fast rather than attempting to support a nearby vLLM
release with potentially different request lifecycles.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import wraps
from http import HTTPStatus
from typing import Any, cast

from .languages import RequestValidationError, adaptation_languages_enabled
from .validation import validate_request

_SUPPORTED_VLLM_VERSION = "0.21.0"
_NEMOTRON_MODEL_MODULE = "aquillm_vllm_nemotron_asr.model"
_NEMOTRON_MODEL_NAME = "Nemotron3_5AsrForRNNT"
_PATCH_SENTINEL = "__aquillm_nemotron_asr_compat_patch__"
_PATCH_OWNER = "aquillm_vllm_nemotron_asr.compat"
_PATCH_VERSION = 1


@dataclass
class _PatchState:
    """The original descriptors and installed wrappers for one vLLM class."""

    original_create: Callable[..., Awaitable[Any]]
    original_preprocess: Callable[..., Awaitable[Any]]
    wrapped_create: Callable[..., Awaitable[Any]]
    wrapped_preprocess: Callable[..., Awaitable[Any]]
    owner: str = _PATCH_OWNER
    version: int = _PATCH_VERSION


_PATCH_STATE: _PatchState | None = None


def _validate_existing_patch_state(
    handler_class: type[object], existing: object
) -> _PatchState:
    """Validate a reload-stable sentinel without depending on class identity."""
    if (
        getattr(existing, "owner", None) != _PATCH_OWNER
        or getattr(existing, "version", None) != _PATCH_VERSION
    ):
        raise RuntimeError("Nemotron ASR compatibility hook has a foreign sentinel.")

    original_create = getattr(existing, "original_create", None)
    original_preprocess = getattr(existing, "original_preprocess", None)
    wrapped_create = getattr(existing, "wrapped_create", None)
    wrapped_preprocess = getattr(existing, "wrapped_preprocess", None)
    if not all(
        callable(value)
        for value in (
            original_create,
            original_preprocess,
            wrapped_create,
            wrapped_preprocess,
        )
    ):
        raise RuntimeError("Nemotron ASR compatibility hook has a malformed sentinel.")
    if (
        handler_class._create_speech_to_text is not wrapped_create
        or handler_class._preprocess_speech_to_text is not wrapped_preprocess
    ):
        raise RuntimeError(
            "Nemotron ASR compatibility hook was partially replaced; "
            "refusing to stack wrappers."
        )
    return cast(_PatchState, existing)


def verify_vllm_compatibility() -> None:
    """Reject every vLLM runtime except the private API we source-checked."""
    try:
        import vllm
    except ImportError as exc:
        raise RuntimeError(
            "Nemotron ASR requires vLLM 0.21.0; vLLM is not installed."
        ) from exc

    version = getattr(vllm, "__version__", None)
    if version != _SUPPORTED_VLLM_VERSION:
        raise RuntimeError(
            "Nemotron ASR compatibility hooks require vLLM "
            f"{_SUPPORTED_VLLM_VERSION} exactly; found {version!r}."
        )

    from vllm import envs

    if envs.VLLM_USE_V2_MODEL_RUNNER:
        raise RuntimeError("Nemotron ASR requires VLLM_USE_V2_MODEL_RUNNER=0.")


def _is_nemotron_handler(handler: object) -> bool:
    """Identify only this plugin model without importing it during discovery."""
    model_cls = getattr(handler, "model_cls", None)
    return (
        getattr(model_cls, "__module__", None) == _NEMOTRON_MODEL_MODULE
        and getattr(model_cls, "__qualname__", None) == _NEMOTRON_MODEL_NAME
    )


def _request_from_create_call(
    args: tuple[object, ...], kwargs: dict[str, object]
) -> object:
    if "request" in kwargs:
        return kwargs["request"]
    if len(args) < 2:
        raise TypeError("vLLM 0.21 _create_speech_to_text request argument is missing")
    return args[1]


def _request_from_preprocess_call(
    args: tuple[object, ...], kwargs: dict[str, object]
) -> object:
    if "request" in kwargs:
        return kwargs["request"]
    if not args:
        raise TypeError(
            "vLLM 0.21 _preprocess_speech_to_text request argument is missing"
        )
    return args[0]


def _validation_error_response(handler: object, error: RequestValidationError) -> Any:
    return handler.create_error_response(
        error.message,
        err_type="BadRequestError",
        status_code=HTTPStatus.BAD_REQUEST,
        param=error.parameter,
    )


def install_compatibility_hook() -> None:
    """Install re-entrant wrappers on vLLM 0.21 speech-to-text handlers."""
    global _PATCH_STATE

    verify_vllm_compatibility()
    from vllm.entrypoints.openai.speech_to_text.speech_to_text import (
        OpenAISpeechToText,
    )

    existing = getattr(OpenAISpeechToText, _PATCH_SENTINEL, None)
    if existing is not None:
        _PATCH_STATE = _validate_existing_patch_state(OpenAISpeechToText, existing)
        return

    original_create = cast(
        Callable[..., Awaitable[Any]], OpenAISpeechToText._create_speech_to_text
    )
    original_preprocess = cast(
        Callable[..., Awaitable[Any]], OpenAISpeechToText._preprocess_speech_to_text
    )

    @wraps(original_create)
    async def wrapped_create(handler: object, *args: object, **kwargs: object) -> Any:
        if not _is_nemotron_handler(handler):
            return await state.original_create(handler, *args, **kwargs)
        try:
            request = _request_from_create_call(args, kwargs)
            # The original preprocessing path supplies the actual resampled
            # duration later; zero here makes option validation happen before
            # model checking, rendering, and generation.
            validate_request(
                request,
                task_type=getattr(handler, "task_type", None),
                duration_s=0.0,
                allow_adaptation=adaptation_languages_enabled(),
            )
            return await state.original_create(handler, *args, **kwargs)
        except RequestValidationError as error:
            return _validation_error_response(handler, error)

    @wraps(original_preprocess)
    async def wrapped_preprocess(
        handler: object, *args: object, **kwargs: object
    ) -> Any:
        if not _is_nemotron_handler(handler):
            return await state.original_preprocess(handler, *args, **kwargs)

        result = await state.original_preprocess(handler, *args, **kwargs)
        # vLLM 0.21's exact original method decodes and resamples audio before
        # returning ``(engine_inputs, duration_s)``.  Use that duration, never
        # multipart headers or compressed input size.
        try:
            _engine_inputs, duration_s = result
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "vLLM 0.21 _preprocess_speech_to_text returned an unexpected value"
            ) from exc
        request = _request_from_preprocess_call(args, kwargs)
        validate_request(
            request,
            task_type=getattr(handler, "task_type", None),
            duration_s=float(duration_s),
            allow_adaptation=adaptation_languages_enabled(),
        )
        return result

    state = _PatchState(
        original_create=original_create,
        original_preprocess=original_preprocess,
        wrapped_create=wrapped_create,
        wrapped_preprocess=wrapped_preprocess,
    )
    OpenAISpeechToText._create_speech_to_text = wrapped_create
    OpenAISpeechToText._preprocess_speech_to_text = wrapped_preprocess
    setattr(OpenAISpeechToText, _PATCH_SENTINEL, state)
    _PATCH_STATE = state
