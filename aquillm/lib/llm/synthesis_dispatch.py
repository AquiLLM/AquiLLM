"""Finite actual SDK dispatch accounting; estimates do not count as dispatches."""

import asyncio
from os import getenv

from lib.llm.evidence_guard import context_limited, current_protection
from lib.llm.turn_context import check_turn_active


def synthesis_limits(output_tokens):
    from lib.llm.providers.complete_turn_policy import _post_tool_synthesis_retry_count

    def bounded_env(name, default, maximum):
        try:
            return min(maximum, max(0, int(getenv(name, str(default)))))
        except ValueError:
            return default

    calls = (3 + _post_tool_synthesis_retry_count()) * (
        1 + bounded_env("OPENAI_TIMEOUT_RETRIES", 2, 10)
    )
    cap = max(
        output_tokens,
        bounded_env("LLM_CONTINUATION_MAX_TOKENS", 4096, 32768),
        bounded_env("LLM_POST_TOOL_MAX_TOKENS", 8192, 32768),
    )
    seconds = max(1, bounded_env("OPENAI_REQUEST_TIMEOUT_MAX_SECONDS", 360, 360))
    return {
        "calls": calls,
        "output_tokens": calls * cap,
        "timeout_seconds": calls * seconds,
    }


async def dispatch(factory, *, output_reserve, kind="initial"):
    check_turn_active()
    state = current_protection()
    lease = getattr(state, "synthesis_lease", None)
    if lease is None:
        result = await factory()
        check_turn_active()
        return result
    try:
        index = lease.start_dispatch(int(output_reserve), kind)
        try:
            async with asyncio.timeout(lease.remaining_seconds()):
                result = await factory()
        finally:
            lease.finish_dispatch(index)
        lease.check_active()
        return result
    except ValueError as exc:
        context_limited(str(exc))


def check_synthesis_active():
    check_turn_active()
    state = current_protection()
    lease = getattr(state, "synthesis_lease", None)
    if lease:
        try:
            lease.check_active()
        except ValueError as exc:
            context_limited(str(exc))
