"""Explicit opt-in acquisition schedule; serving dispatch is separately owned."""

from os import getenv

from django.core.exceptions import ImproperlyConfigured

NATIVE_CLOSURE_RESERVE_MS = 1250
# Single-document searches can recover their captured baseline after graph
# traversal expires. This slice is taken from graph time, never final closure.
SCOPED_BASELINE_RESERVE_MS = 2500
NATIVE_CLOSURE_SCHEDULE_VERSION = "native-closure-v3"


def acquisition_schedule():
    value = getenv("RAG_ACQUISITION_SCHEDULE", "scored").strip().lower()
    if value not in ("scored", "native_closure"):
        raise ImproperlyConfigured("invalid RAG_ACQUISITION_SCHEDULE")
    return value
