"""Request-policy validation for Nemotron ASR's deterministic batch mode."""

from __future__ import annotations

from typing import Any

from .languages import RequestValidationError, normalize_language


def _unsupported(parameter: str, value: Any, policy: str) -> RequestValidationError:
    return RequestValidationError(
        parameter,
        value,
        f"Unsupported {parameter}={value!r}; {policy}.",
    )


def _request_value(request: object, parameter: str, default: Any) -> Any:
    return getattr(request, parameter, default)


def validate_request(
    request: object,
    *,
    task_type: str,
    duration_s: float,
    allow_adaptation: bool = False,
) -> str:
    """Validate a vLLM transcription request and return its model locale."""
    if task_type != "transcribe":
        raise _unsupported("task_type", task_type, "only 'transcribe' is supported")

    to_language = _request_value(request, "to_language", None)
    if to_language not in (None, ""):
        raise _unsupported("to_language", to_language, "translation is not supported")

    vllm_xargs = _request_value(request, "vllm_xargs", None)
    if vllm_xargs is not None and (not isinstance(vllm_xargs, dict) or vllm_xargs):
        raise _unsupported(
            "vllm_xargs",
            vllm_xargs,
            "custom vLLM arguments are not supported",
        )

    max_completion_tokens = _request_value(request, "max_completion_tokens", None)
    if max_completion_tokens is not None:
        raise _unsupported(
            "max_completion_tokens",
            max_completion_tokens,
            "completion length is controlled by the audio",
        )

    include_stop_str_in_output = _request_value(
        request, "include_stop_str_in_output", False
    )
    if include_stop_str_in_output is not False:
        raise _unsupported(
            "include_stop_str_in_output",
            include_stop_str_in_output,
            "stop strings are not used for transcription",
        )

    temperature = _request_value(request, "temperature", None)
    if temperature not in (None, 0):
        raise _unsupported("temperature", temperature, "only None or 0 is supported")

    use_beam_search = _request_value(request, "use_beam_search", False)
    if use_beam_search is not False:
        raise _unsupported(
            "use_beam_search",
            use_beam_search,
            "beam search is not supported",
        )

    n = _request_value(request, "n", 1)
    if n != 1:
        raise _unsupported("n", n, "only n=1 is supported")

    prompt = _request_value(request, "prompt", None)
    if prompt not in (None, ""):
        raise _unsupported("prompt", prompt, "prompts are not supported")

    hotwords = _request_value(request, "hotwords", None)
    if hotwords not in (None, ""):
        raise _unsupported("hotwords", hotwords, "hotwords are not supported")

    stream = _request_value(request, "stream", False)
    if stream is not False and stream is not None:
        raise _unsupported("stream", stream, "streaming is not supported")

    stream_include_usage = _request_value(request, "stream_include_usage", False)
    if stream_include_usage is not False and stream_include_usage is not None:
        raise _unsupported(
            "stream_include_usage",
            stream_include_usage,
            "streaming is not supported",
        )

    stream_continuous_usage_stats = _request_value(
        request, "stream_continuous_usage_stats", False
    )
    if (
        stream_continuous_usage_stats is not False
        and stream_continuous_usage_stats is not None
    ):
        raise _unsupported(
            "stream_continuous_usage_stats",
            stream_continuous_usage_stats,
            "streaming is not supported",
        )

    response_format = _request_value(request, "response_format", "json")
    if not isinstance(response_format, str) or response_format not in {"json", "text"}:
        raise _unsupported(
            "response_format",
            response_format,
            "only 'json' or 'text' is supported",
        )

    timestamp_granularities = _request_value(request, "timestamp_granularities", None)
    if timestamp_granularities not in (None, []):
        raise _unsupported(
            "timestamp_granularities",
            timestamp_granularities,
            "timestamps are not supported",
        )

    if duration_s > 390.0:
        raise _unsupported(
            "duration_s",
            duration_s,
            "audio must be 390 seconds or shorter",
        )

    return normalize_language(
        _request_value(request, "language", None), allow_adaptation=allow_adaptation
    )
