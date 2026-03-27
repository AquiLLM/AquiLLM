"""Logging configuration for Django (split from settings for file-length budget)."""
from __future__ import annotations

import os
from pathlib import Path

# Match aquillm.settings BASE_DIR (this file lives alongside settings.py).
_BASE = Path(__file__).resolve().parent.parent
LOGS_DIR = os.path.join(_BASE, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "{levelname} {asctime} {module} {process:d} {thread:d} trace_id={otel_trace_id} {message}",
            "style": "{",
        },
        "simple": {
            "format": "{levelname} {message}",
            "style": "{",
        },
    },
    "filters": {
        "trace_context": {
            "()": "aquillm.observability.TraceContextFilter",
        },
    },
    "handlers": {
        "file": {
            "level": "INFO",
            "class": "logging.FileHandler",
            "filename": os.path.join(LOGS_DIR, "django.log"),
            "formatter": "verbose",
            "filters": ["trace_context"],
        },
        "console": {
            "level": "DEBUG",
            "class": "logging.StreamHandler",
            "formatter": "verbose",
            "filters": ["trace_context"],
        },
    },
    "loggers": {
        "django": {
            "handlers": ["file", "console"],
            "level": "INFO",
            "propagate": True,
        },
        "django.request": {
            "handlers": ["file", "console"],
            "level": "DEBUG",
            "propagate": False,
        },
        "aquillm": {
            "handlers": ["file", "console"],
            "level": "DEBUG",
            "propagate": True,
        },
        "chat": {
            "handlers": ["file", "console"],
            "level": "DEBUG",
            "propagate": True,
        },
        "celery": {
            "handlers": ["file", "console"],
            "level": "DEBUG",
            "propagate": True,
        },
        "ingest": {
            "handlers": ["file", "console"],
            "level": "DEBUG",
            "propagate": True,
        },
        # lib.llm.* (context packer, prompt_budget, etc.) — not under the "aquillm" package name
        "lib.llm.utils": {
            "handlers": ["file", "console"],
            "level": "INFO",
            "propagate": False,
        },
        "kombu": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
    },
}

__all__ = ["LOGGING", "LOGS_DIR"]
