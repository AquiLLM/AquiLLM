"""Opt-in private observation sink. Never logs prompt/source/identity content."""

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field
from threading import Event
from time import perf_counter

_sink = ContextVar("evidence_observation", default=None)


@dataclass
class Observation:
    sink: object
    failed: Event = field(default_factory=Event)
    sdk_started: Event = field(default_factory=Event)
    completed: Event = field(default_factory=Event)


@contextmanager
def observe(sink):
    state = Observation(sink)
    token = _sink.set(state)
    try:
        yield state
    finally:
        _sink.reset(token)


def publish(event, data):
    state = _sink.get()
    if state is not None:
        if event == "sdk_start":
            state.sdk_started.set()
        elif event == "turn_complete":
            state.completed.set()
        try:
            state.sink(event, {"at": perf_counter(), **deepcopy(data)})
        except BaseException:
            # Observer failures never replace application returns/exceptions,
            # including cancellation. The owner must reject incomplete evidence.
            state.failed.set()


def active():
    return _sink.get() is not None
