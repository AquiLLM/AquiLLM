"""Optional application execution owner, propagated through both tool threads."""

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar

_EXECUTION = ContextVar("chat_execution", default=None)


class UncertainToolExecution(RuntimeError):
    """A prior invocation may have had side effects and must not be replayed."""


UNCERTAIN_MESSAGE = "A tool operation was interrupted and may still be running or may already have finished. It was not run again. Check its output before explicitly requesting another attempt."


def execution_is_bound():
    return _EXECUTION.get() is not None


def execution_publication():
    execution = _EXECUTION.get()
    return execution.publication() if execution is not None else nullcontext()


def uncertain_tool_message(message):
    from lib.llm.types.messages import ToolMessage

    return ToolMessage(
        tool_name=message.tool_call_name or "tool",
        for_whom="user",
        content=UNCERTAIN_MESSAGE,
        arguments=message.tool_call_input,
        result_dict={"exception": UNCERTAIN_MESSAGE},
    )


@contextmanager
def bind_execution(execution):
    token = _EXECUTION.set(execution)
    try:
        yield execution
    finally:
        _EXECUTION.reset(token)


def check_execution_active():
    execution = _EXECUTION.get()
    if execution is not None:
        execution.check_active()


def run_tool_once(message, function):
    execution = _EXECUTION.get()
    return (
        execution.run_tool(message, function) if execution is not None else function()
    )
