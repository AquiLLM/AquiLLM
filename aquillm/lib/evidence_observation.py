"""Opt-in private observation sink. Never logs prompt/source/identity content."""

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from time import perf_counter

_sink = ContextVar("evidence_observation", default=None)


@contextmanager
def observe(sink):
    token = _sink.set(sink)
    try:
        yield
    finally:
        _sink.reset(token)


def publish(event, data):
    sink = _sink.get()
    if sink is not None:
        sink(event, {"at": perf_counter(), **deepcopy(data)})


def active():
    return _sink.get() is not None
