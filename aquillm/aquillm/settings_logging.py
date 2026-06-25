"""Logging configuration for Django — structlog with direct Loki push."""
from __future__ import annotations

import os

import structlog

from aquillm.observability import add_otel_trace_context

DEBUG = os.environ.get("DJANGO_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")

# --------------------------------------------------------------------------- #
# Shared structlog processors (used by both structlog loggers and stdlib       #
# loggers routed through ProcessorFormatter).                                  #
# --------------------------------------------------------------------------- #
shared_processors: list = [
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_logger_name,
    structlog.stdlib.add_log_level,
    structlog.stdlib.PositionalArgumentsFormatter(),
    structlog.processors.TimeStamper(fmt="iso"),
    structlog.processors.StackInfoRenderer(),
    structlog.processors.UnicodeDecoder(),
    add_otel_trace_context,
    structlog.processors.format_exc_info,
]

# --------------------------------------------------------------------------- #
# Configure structlog itself (for code that calls structlog.get_logger()).     #
# --------------------------------------------------------------------------- #
structlog.configure(
    processors=[structlog.stdlib.filter_by_level]
    + shared_processors
    + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# --------------------------------------------------------------------------- #
# Handlers — console always, loki only when LOKI_PUSH_URL is set.             #
# --------------------------------------------------------------------------- #
_default_handlers = ["console"]

if os.environ.get("LOKI_PUSH_URL"):
    _default_handlers.append("loki")

_loki_handler_cfg = {
    "level": "INFO",
    "class": "logging_loki.LokiHandler",
    "url": os.environ.get("LOKI_PUSH_URL", ""),
    "tags": {"service_name": os.environ.get("OTEL_SERVICE_NAME", "aquillm")},
    "version": "1",
    "formatter": "json",
}

# --------------------------------------------------------------------------- #
# stdlib LOGGING dict — Django applies this via dictConfig.                    #
# --------------------------------------------------------------------------- #
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": structlog.stdlib.ProcessorFormatter,
            "processors": [
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer(),
            ],
            "foreign_pre_chain": shared_processors,
        },
        "console": {
            "()": structlog.stdlib.ProcessorFormatter,
            "processors": [
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.dev.ConsoleRenderer()
                if DEBUG
                else structlog.processors.JSONRenderer(),
            ],
            "foreign_pre_chain": shared_processors,
        },
    },
    "handlers": {
        "console": {
            "level": "DEBUG",
            "class": "logging.StreamHandler",
            "formatter": "console",
        },
        **({"loki": _loki_handler_cfg} if os.environ.get("LOKI_PUSH_URL") else {}),
    },
    "loggers": {
        "django": {
            "handlers": _default_handlers,
            "level": "INFO",
            "propagate": True,
        },
        "django.request": {
            "handlers": _default_handlers,
            "level": "DEBUG",
            "propagate": False,
        },
        "aquillm": {
            "handlers": _default_handlers,
            "level": "DEBUG",
            "propagate": True,
        },
        "chat": {
            "handlers": _default_handlers,
            "level": "DEBUG",
            "propagate": True,
        },
        "celery": {
            "handlers": _default_handlers,
            "level": "DEBUG",
            "propagate": True,
        },
        "ingest": {
            "handlers": _default_handlers,
            "level": "DEBUG",
            "propagate": True,
        },
        "lib.llm.utils": {
            "handlers": _default_handlers,
            "level": "INFO",
            "propagate": False,
        },
        "kombu": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
        "uvicorn.access": {
            "handlers": _default_handlers,
            "level": "WARNING",
            "propagate": False,
        },
    },
}

__all__ = ["LOGGING"]
