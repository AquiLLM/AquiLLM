"""Request-only library bridge for cancellation, tool hops and app handoff."""

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from dataclasses import dataclass
from os import getenv

_TURN = ContextVar("llm_turn_context", default=None)
_IN_HANDOFF = ContextVar("llm_in_evidence_handoff", default=False)


@dataclass
class TurnContext:
    budget: object
    handoff: object = None
    policy: object = None
    max_func_calls: int = 0
    worker_cleanup: object = None


def current_turn():
    return _TURN.get()


def check_turn_active():
    state = current_turn()
    if state is not None:
        state.budget.check_active()


@contextmanager
def bind_turn(state):
    token = _TURN.set(state)
    try:
        yield state
    finally:
        _TURN.reset(token)


async def bounded_retrieval(awaitable, budget, *, cap_ms=None):
    from lib.retrieval.operation import operation_scope

    timeout = budget.remaining_ms()
    if cap_ms is not None:
        timeout = min(timeout, cap_ms)
    try:
        with operation_scope(budget):
            async with asyncio.timeout(timeout / 1000):
                result = await awaitable
        budget.check_active()
        if not budget.can_publish():
            raise TimeoutError("retrieval deadline")
        return result
    except asyncio.CancelledError:
        budget.close("cancelled")
        raise


def is_retrieval_tool(name):
    return name in {
        "vector_search",
        "search_single_document",
        "whole_document",
        "more_context",
    }


def tool_timeout(name=None, *, retrieval=False):
    seconds = min(10.0, max(0.001, float(getenv("TOOL_CALL_TIMEOUT_SECONDS", "10"))))
    state = current_turn()
    return (
        min(seconds, state.budget.remaining_ms() / 1000)
        if state and (retrieval or is_retrieval_tool(name))
        else seconds
    )


async def call_tool_async(llm, message):
    """Both executor hops inherit context; event-loop cancellation remains live."""
    check_turn_active()
    from contextlib import nullcontext

    from lib.retrieval.operation import operation_scope

    state = current_turn()
    try:
        with (
            operation_scope(state.budget)
            if state and is_retrieval_tool(message.tool_call_name)
            else nullcontext()
        ):
            return await asyncio.wait_for(
                asyncio.to_thread(llm.call_tool, message),
                tool_timeout(message.tool_call_name),
            )
    except asyncio.CancelledError:
        if state:
            state.budget.close("cancelled")
        raise
    except TimeoutError:
        from lib.llm.types.messages import ToolMessage

        return ToolMessage(
            tool_name=message.tool_call_name or "invalid_tool",
            for_whom="assistant",
            content="Tool call timed out; available evidence may be partial.",
            arguments=message.tool_call_input or {},
            result_dict={"result": [], "retrieval_status": "context_limited"},
        )


def submit_tool(executor, function):
    def invoke():
        state = current_turn()
        try:
            return function()
        finally:
            if state and state.worker_cleanup:
                state.worker_cleanup()

    return executor.submit(copy_context().run, invoke)


async def evidence_handoff(llm, conversation, max_tokens, stream_func):
    from lib.llm.types.messages import ToolMessage

    state = current_turn()
    if state is None or state.handoff is None or _IN_HANDOFF.get():
        return None
    if not conversation.messages or not isinstance(conversation[-1], ToolMessage):
        return None
    check_turn_active()
    token = _IN_HANDOFF.set(True)
    try:
        return await state.handoff(llm, conversation, max_tokens, stream_func)
    finally:
        _IN_HANDOFF.reset(token)


@contextmanager
def suppress_evidence_handoff():
    token = _IN_HANDOFF.set(True)
    try:
        yield
    finally:
        _IN_HANDOFF.reset(token)


def fenced_callback(callback):
    if callback is None:
        return None
    state = current_turn()

    async def guarded(*args, **kwargs):
        if state:
            state.budget.check_active()
        return await callback(*args, **kwargs)

    return guarded
