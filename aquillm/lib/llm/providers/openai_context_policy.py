"""OpenAI context-estimation and retry adapters; legacy policy remains unchanged."""

from tiktoken import encoding_for_model

from .openai_overflow import (
    retry_args_for_context_overflow,
    retry_args_for_timeout,
    strip_images_from_messages,
)
from .openai_request import is_timeout_error
from .openai_tokens import (
    context_reserve_tokens,
    env_float,
    env_int,
    estimate_prompt_tokens,
    preflight_trim_for_context,
    trim_messages_for_overflow,
)

gpt_enc = encoding_for_model("gpt-4o")


class OpenAIContextPolicy:
    @staticmethod
    def _trim_messages_for_overflow(arguments: dict, overflow_tokens: int) -> bool:
        return trim_messages_for_overflow(arguments, overflow_tokens)

    @classmethod
    def _estimate_prompt_tokens(cls, messages: list[dict]) -> int:
        return estimate_prompt_tokens(messages, gpt_enc)

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        return env_int(name, default)

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        return env_float(name, default)

    @classmethod
    def _context_reserve_tokens(cls, context_limit: int) -> tuple[int, int]:
        return context_reserve_tokens(context_limit)

    @classmethod
    def _preflight_trim_for_context(
        cls, arguments: dict, context_limit: int, extra_prompt_slack: int = 0
    ) -> None:
        preflight_trim_for_context(cls, arguments, context_limit, extra_prompt_slack)

    @staticmethod
    def _strip_images_from_messages(arguments: dict) -> bool:
        return strip_images_from_messages(arguments)

    @staticmethod
    def _retry_args_for_context_overflow(
        arguments: dict, exc: Exception
    ) -> dict | None:
        return retry_args_for_context_overflow(arguments, exc)

    @staticmethod
    def _is_timeout_error(exc: Exception) -> bool:
        return is_timeout_error(exc)

    @staticmethod
    def _retry_args_for_timeout(arguments: dict, attempt: int) -> dict | None:
        return retry_args_for_timeout(arguments, attempt)
