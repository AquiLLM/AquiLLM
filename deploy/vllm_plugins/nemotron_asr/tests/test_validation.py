from __future__ import annotations

import sys
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aquillm_vllm_nemotron_asr.languages import RequestValidationError
from aquillm_vllm_nemotron_asr.validation import validate_request


@dataclass
class FakeTranscriptionRequest:
    """The vLLM 0.21 request fields that affect the ASR plugin policy."""

    file: Any = None
    model: str | None = None
    language: str | None = None
    prompt: str | None = None
    response_format: str = "json"
    temperature: float | None = None
    timestamp_granularities: list[str] | None = None
    stream: bool | None = False
    stream_include_usage: bool | None = False
    stream_continuous_usage_stats: bool | None = False
    include_stop_str_in_output: bool = False
    max_completion_tokens: int | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    seed: int | None = None
    frequency_penalty: float | None = None
    repetition_penalty: float | None = None
    presence_penalty: float | None = None
    length_penalty: float | None = None
    n: int = 1
    use_beam_search: bool = False
    hotwords: str | None = None
    to_language: str | None = None
    vllm_xargs: dict[str, Any] | None = None


class PartialRequest:
    """A compat-shaped object that leaves all optional request fields absent."""


@pytest.mark.parametrize(
    "asr_request",
    [
        FakeTranscriptionRequest(),
        FakeTranscriptionRequest(response_format="text", language="en"),
        FakeTranscriptionRequest(
            prompt="",
            hotwords="",
            temperature=0,
            to_language="",
            vllm_xargs={},
            timestamp_granularities=[],
            stream=False,
            stream_include_usage=False,
            stream_continuous_usage_stats=False,
        ),
        FakeTranscriptionRequest(stream=None),
        FakeTranscriptionRequest(stream_include_usage=None),
        FakeTranscriptionRequest(stream_continuous_usage_stats=None),
        FakeTranscriptionRequest(
            language="de",
            top_p=0.3,
            top_k=7,
            min_p=0.2,
            seed=42,
            frequency_penalty=1.5,
            repetition_penalty=1.2,
            presence_penalty=-1.0,
            length_penalty=0.8,
        ),
    ],
)
@pytest.mark.parametrize("duration_s", [389.0, 390.0])
def test_validate_request_accepts_the_supported_policy(
    asr_request: FakeTranscriptionRequest, duration_s: float
) -> None:
    original = deepcopy(asr_request)

    locale = validate_request(
        asr_request,
        task_type="transcribe",
        duration_s=duration_s,
    )

    expected_locale = {"de": "de-DE", "en": "en-US"}.get(
        asr_request.language,
        "auto",
    )
    assert locale == expected_locale
    assert asr_request == original


def test_validate_request_defaults_missing_optionals() -> None:
    request = PartialRequest()

    assert validate_request(request, task_type="transcribe", duration_s=1.0) == "auto"


@pytest.mark.parametrize(
    ("task_type", "changes", "duration_s", "parameter"),
    [
        ("translate", {}, 1.0, "task_type"),
        ("transcribe", {"to_language": "fr"}, 1.0, "to_language"),
        ("transcribe", {"vllm_xargs": {"foo": "bar"}}, 1.0, "vllm_xargs"),
        ("transcribe", {"max_completion_tokens": 1}, 1.0, "max_completion_tokens"),
        (
            "transcribe",
            {"include_stop_str_in_output": True},
            1.0,
            "include_stop_str_in_output",
        ),
        (
            "transcribe",
            {"include_stop_str_in_output": None},
            1.0,
            "include_stop_str_in_output",
        ),
        ("transcribe", {"temperature": 0.5}, 1.0, "temperature"),
        ("transcribe", {"use_beam_search": True}, 1.0, "use_beam_search"),
        ("transcribe", {"use_beam_search": None}, 1.0, "use_beam_search"),
        ("transcribe", {"n": 2}, 1.0, "n"),
        ("transcribe", {"prompt": "context"}, 1.0, "prompt"),
        ("transcribe", {"hotwords": "AquiLLM"}, 1.0, "hotwords"),
        ("transcribe", {"hotwords": []}, 1.0, "hotwords"),
        ("transcribe", {"vllm_xargs": []}, 1.0, "vllm_xargs"),
        ("transcribe", {"stream": True}, 1.0, "stream"),
        ("transcribe", {"stream_include_usage": True}, 1.0, "stream_include_usage"),
        (
            "transcribe",
            {"stream_continuous_usage_stats": True},
            1.0,
            "stream_continuous_usage_stats",
        ),
        ("transcribe", {"response_format": "verbose_json"}, 1.0, "response_format"),
        ("transcribe", {"response_format": "srt"}, 1.0, "response_format"),
        ("transcribe", {"response_format": "vtt"}, 1.0, "response_format"),
        ("transcribe", {"response_format": "other"}, 1.0, "response_format"),
        ("transcribe", {"response_format": []}, 1.0, "response_format"),
        (
            "transcribe",
            {"timestamp_granularities": ["word"]},
            1.0,
            "timestamp_granularities",
        ),
        ("transcribe", {}, 391.0, "duration_s"),
    ],
)
def test_validate_request_rejects_unsupported_options_with_stable_parameter(
    task_type: str, changes: dict[str, Any], duration_s: float, parameter: str
) -> None:
    request = replace(FakeTranscriptionRequest(), **changes)

    with pytest.raises(RequestValidationError) as error:
        validate_request(request, task_type=task_type, duration_s=duration_s)

    assert error.value.parameter == parameter
    expected_value = (
        task_type
        if parameter == "task_type"
        else duration_s
        if parameter == "duration_s"
        else getattr(request, parameter)
    )
    assert error.value.value == expected_value
    assert parameter in error.value.message


def test_validate_request_propagates_language_policy_and_adaptation_switch() -> None:
    request = FakeTranscriptionRequest(language="el")

    with pytest.raises(RequestValidationError) as error:
        validate_request(request, task_type="transcribe", duration_s=1.0)

    assert error.value.parameter == "language"
    assert (
        validate_request(
            request, task_type="transcribe", duration_s=1.0, allow_adaptation=True
        )
        == "el-GR"
    )
