from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aquillm_vllm_nemotron_asr.languages import (
    ADAPTATION_LOCALES,
    PRODUCTION_LOCALES,
    RequestValidationError,
    adaptation_languages_enabled,
    normalize_language,
)

EXPECTED_PRODUCTION_LOCALES = (
    "en-US",
    "en-GB",
    "es-US",
    "es-ES",
    "fr-FR",
    "fr-CA",
    "it-IT",
    "pt-BR",
    "pt-PT",
    "nl-NL",
    "de-DE",
    "tr-TR",
    "ru-RU",
    "ar-AR",
    "hi-IN",
    "ja-JP",
    "ko-KR",
    "vi-VN",
    "uk-UA",
    "pl-PL",
    "sv-SE",
    "cs-CZ",
    "nb-NO",
    "da-DK",
    "bg-BG",
    "fi-FI",
    "hr-HR",
    "sk-SK",
    "zh-CN",
    "hu-HU",
    "ro-RO",
    "et-EE",
)
EXPECTED_ADAPTATION_LOCALES = (
    "el-GR",
    "lt-LT",
    "lv-LV",
    "mt-MT",
    "sl-SI",
    "he-IL",
    "th-TH",
    "nn-NO",
)


def test_language_locale_sets_match_the_pinned_model_readme() -> None:
    assert PRODUCTION_LOCALES == EXPECTED_PRODUCTION_LOCALES
    assert ADAPTATION_LOCALES == EXPECTED_ADAPTATION_LOCALES


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "auto"),
        ("", "auto"),
        ("  ", "auto"),
        ("auto", "auto"),
        (" AUTO ", "auto"),
        ("en", "en-US"),
        ("es", "es-US"),
        ("fr", "fr-FR"),
        ("pt", "pt-BR"),
        ("zh", "zh-CN"),
        (" eN-uS ", "en-US"),
        ("FR_ca", "fr-CA"),
    ],
)
def test_normalize_language_canonicalizes_default_and_ambiguous_values(
    value: str | None, expected: str
) -> None:
    assert normalize_language(value) == expected


@pytest.mark.parametrize(
    "locale",
    [
        "it-IT",
        "nl-NL",
        "de-DE",
        "tr-TR",
        "ru-RU",
        "ar-AR",
        "hi-IN",
        "ja-JP",
        "ko-KR",
        "vi-VN",
        "uk-UA",
        "pl-PL",
        "sv-SE",
        "cs-CZ",
        "nb-NO",
        "da-DK",
        "bg-BG",
        "fi-FI",
        "hr-HR",
        "sk-SK",
        "hu-HU",
        "ro-RO",
        "et-EE",
    ],
)
def test_normalize_language_maps_each_unambiguous_production_bare_code(
    locale: str,
) -> None:
    assert normalize_language(locale.split("-", maxsplit=1)[0].upper()) == locale


@pytest.mark.parametrize("locale", EXPECTED_PRODUCTION_LOCALES)
def test_normalize_language_accepts_every_explicit_production_locale(
    locale: str,
) -> None:
    assert normalize_language(locale) == locale


@pytest.mark.parametrize(
    "value",
    [
        *EXPECTED_ADAPTATION_LOCALES,
        *(locale.split("-", maxsplit=1)[0] for locale in EXPECTED_ADAPTATION_LOCALES),
    ],
)
def test_normalize_language_rejects_adaptation_languages_by_default(value: str) -> None:
    with pytest.raises(RequestValidationError) as error:
        normalize_language(value)

    assert error.value.parameter == "language"
    assert error.value.value == value
    assert value in error.value.message
    assert "supported choices" in error.value.message


@pytest.mark.parametrize(
    ("value", "expected"),
    [(locale, locale) for locale in EXPECTED_ADAPTATION_LOCALES]
    + [
        (locale.split("-", maxsplit=1)[0], locale)
        for locale in EXPECTED_ADAPTATION_LOCALES
    ],
)
def test_normalize_language_allows_adaptation_locales_when_enabled(
    value: str, expected: str
) -> None:
    assert normalize_language(value, allow_adaptation=True) == expected


@pytest.mark.parametrize("value", ["en-AU", "klingon", "123", object()])
def test_normalize_language_rejects_unknown_values_with_actionable_error(
    value: object,
) -> None:
    with pytest.raises(RequestValidationError) as error:
        normalize_language(value)  # type: ignore[arg-type]

    assert error.value.parameter == "language"
    assert error.value.value is value
    assert repr(value) in error.value.message
    assert "supported choices" in error.value.message
    assert "en-US" in error.value.message
    assert all(
        locale not in error.value.message for locale in EXPECTED_ADAPTATION_LOCALES
    )


def test_unknown_language_with_adaptation_enabled_lists_adaptation_choices() -> None:
    with pytest.raises(RequestValidationError) as error:
        normalize_language("klingon", allow_adaptation=True)

    assert error.value.parameter == "language"
    assert all(locale in error.value.message for locale in EXPECTED_ADAPTATION_LOCALES)


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        ({}, False),
        ({"NEMOTRON_ASR_ALLOW_ADAPTATION_LANGUAGES": "0"}, False),
        ({"NEMOTRON_ASR_ALLOW_ADAPTATION_LANGUAGES": "false"}, False),
        ({"NEMOTRON_ASR_ALLOW_ADAPTATION_LANGUAGES": "1"}, True),
        ({"NEMOTRON_ASR_ALLOW_ADAPTATION_LANGUAGES": "true"}, True),
        ({"NEMOTRON_ASR_ALLOW_ADAPTATION_LANGUAGES": "YES"}, True),
        ({"NEMOTRON_ASR_ALLOW_ADAPTATION_LANGUAGES": "on"}, True),
    ],
)
def test_adaptation_languages_enabled_parses_environment_at_call_time(
    environment: Mapping[str, str], expected: bool
) -> None:
    assert adaptation_languages_enabled(environment) is expected


def test_adaptation_languages_enabled_reads_os_environment_at_each_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NEMOTRON_ASR_ALLOW_ADAPTATION_LANGUAGES", "0")
    assert adaptation_languages_enabled() is False

    monkeypatch.setenv("NEMOTRON_ASR_ALLOW_ADAPTATION_LANGUAGES", "yes")
    assert adaptation_languages_enabled() is True
