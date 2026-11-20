"""Explicit opt-in acquisition schedule; serving dispatch is separately owned."""

from os import getenv

from django.core.exceptions import ImproperlyConfigured

NATIVE_CLOSURE_RESERVE_MS = 1250
# Captured baseline completion borrows graph time, never final closure.
SCOPED_BASELINE_RESERVE_MS = 2500
NATIVE_CLOSURE_SCHEDULE_VERSION = "native-closure-v4"


def scoped_baseline_recovery_enabled():
    key = "RAG_SCOPED_BASELINE_RECOVERY_ENABLED"
    value = getenv(key, "0").strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise ImproperlyConfigured(f"invalid {key}")


def native_closure_version(scoped_recovery):
    return "native-closure-v5" if scoped_recovery else NATIVE_CLOSURE_SCHEDULE_VERSION


def acquisition_schedule():
    value = getenv("RAG_ACQUISITION_SCHEDULE", "scored").strip().lower()
    if value not in ("scored", "native_closure"):
        raise ImproperlyConfigured("invalid RAG_ACQUISITION_SCHEDULE")
    return value
