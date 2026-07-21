"""Language selection helpers for the Nemotron ASR plugin."""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Mapping
from typing import Any


# https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/blob/f3d333391852ba876df169dcc9ba902d25b6ab0b/README.md
PRODUCTION_LOCALES = (
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

ADAPTATION_LOCALES = (
    "el-GR",
    "lt-LT",
    "lv-LV",
    "mt-MT",
    "sl-SI",
    "he-IL",
    "th-TH",
    "nn-NO",
)


class RequestValidationError(ValueError):
    """A framework-independent validation error for the ASR compatibility layer."""

    def __init__(self, parameter: str, value: Any, message: str) -> None:
        self.parameter = parameter
        self.value = value
        self.message = message
        super().__init__(message)


def _locale_key(locale: str) -> str:
    return locale.replace("_", "-").lower()


def _locales_by_language(locales: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    grouped: defaultdict[str, list[str]] = defaultdict(list)
    for locale in locales:
        language, _region = locale.split("-", maxsplit=1)
        grouped[language].append(locale)
    return {language: tuple(values) for language, values in grouped.items()}


_PRODUCTION_BY_LANGUAGE = _locales_by_language(PRODUCTION_LOCALES)
_ADAPTATION_BY_LANGUAGE = _locales_by_language(ADAPTATION_LOCALES)
_PRODUCTION_BY_KEY = {_locale_key(locale): locale for locale in PRODUCTION_LOCALES}
_ADAPTATION_BY_KEY = {_locale_key(locale): locale for locale in ADAPTATION_LOCALES}
_DEFAULT_LOCALES = {
    "en": "en-US",
    "es": "es-US",
    "fr": "fr-FR",
    "pt": "pt-BR",
    "zh": "zh-CN",
}
_PRODUCTION_CHOICES = ("auto", *PRODUCTION_LOCALES)


def adaptation_languages_enabled(
    environment: Mapping[str, str] | None = None,
) -> bool:
    """Return whether the process environment opts into adaptation-only locales."""
    if environment is None:
        environment = os.environ
    value = environment.get("NEMOTRON_ASR_ALLOW_ADAPTATION_LANGUAGES", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _language_error(
    value: Any,
    *,
    adaptation: bool = False,
    allow_adaptation: bool = False,
) -> RequestValidationError:
    supported_choices = ", ".join(
        (*_PRODUCTION_CHOICES, *(ADAPTATION_LOCALES if allow_adaptation else ()))
    )
    detail = (
        " This locale is adaptation-ready; set "
        "NEMOTRON_ASR_ALLOW_ADAPTATION_LANGUAGES=1 to allow it."
        if adaptation
        else ""
    )
    return RequestValidationError(
        "language",
        value,
        f"Unsupported language {value!r}; supported choices: {supported_choices}.{detail}",
    )


def normalize_language(value: str | None, *, allow_adaptation: bool = False) -> str:
    """Return the canonical model locale selected by an API language value."""
    if value is None:
        return "auto"
    if not isinstance(value, str):
        raise _language_error(value, allow_adaptation=allow_adaptation)

    candidate = value.strip()
    if not candidate or candidate.lower() == "auto":
        return "auto"

    key = _locale_key(candidate)
    production_locale = _PRODUCTION_BY_KEY.get(key)
    if production_locale is not None:
        return production_locale

    adaptation_locale = _ADAPTATION_BY_KEY.get(key)
    if adaptation_locale is not None:
        if allow_adaptation:
            return adaptation_locale
        raise _language_error(value, adaptation=True, allow_adaptation=allow_adaptation)

    if "-" not in key:
        default_locale = _DEFAULT_LOCALES.get(key)
        if default_locale is not None:
            return default_locale

        production_locales = _PRODUCTION_BY_LANGUAGE.get(key, ())
        if len(production_locales) == 1:
            return production_locales[0]

        adaptation_locales = _ADAPTATION_BY_LANGUAGE.get(key, ())
        if len(adaptation_locales) == 1:
            if allow_adaptation:
                return adaptation_locales[0]
            raise _language_error(value, adaptation=True, allow_adaptation=allow_adaptation)

    raise _language_error(value, allow_adaptation=allow_adaptation)
