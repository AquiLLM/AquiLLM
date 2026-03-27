"""Observability bootstrap — OpenTelemetry tracing + Pyroscope profiling.

Call setup() once at process startup (from settings.py).
Everything is guarded by the OTEL_ENABLED env var so this is a
complete no-op when the observability stack is not running.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


class TraceContextFilter(logging.Filter):
    """Inject OpenTelemetry trace/span IDs into every log record.

    This enables Grafana's derived-fields feature to link log lines
    in Loki to the corresponding trace in Tempo.
    """

    def filter(self, record):
        try:
            from opentelemetry import trace

            span = trace.get_current_span()
            ctx = span.get_span_context()
            record.otel_trace_id = (
                format(ctx.trace_id, "032x") if ctx.trace_id else "0"
            )
            record.otel_span_id = (
                format(ctx.span_id, "016x") if ctx.span_id else "0"
            )
        except Exception:
            record.otel_trace_id = "0"
            record.otel_span_id = "0"
        return True


def _init_tracing():
    """Configure OpenTelemetry tracing with OTLP HTTP export to Tempo."""
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = Resource.create(
        {"service.name": os.environ.get("OTEL_SERVICE_NAME", "aquillm")}
    )
    provider = TracerProvider(resource=resource)

    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://tempo:4318")
    exporter = OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces")
    provider.add_span_processor(BatchSpanProcessor(exporter))

    # Link Pyroscope profiles to trace spans.
    try:
        from pyroscope.otel import PyroscopeSpanProcessor

        provider.add_span_processor(PyroscopeSpanProcessor())
        logger.info("PyroscopeSpanProcessor attached to TracerProvider")
    except Exception:
        logger.debug("pyroscope-otel not available, span profiling disabled")

    trace.set_tracer_provider(provider)

    # Auto-instrument Celery, psycopg2, Redis.
    # Django/ASGI HTTP spans are handled by OpenTelemetryMiddleware in asgi.py.
    from opentelemetry.instrumentation.celery import CeleryInstrumentor
    from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor
    from opentelemetry.instrumentation.redis import RedisInstrumentor

    CeleryInstrumentor().instrument()
    Psycopg2Instrumentor().instrument()
    RedisInstrumentor().instrument()

    logger.info("OpenTelemetry tracing initialized (endpoint=%s)", endpoint)


def _init_pyroscope():
    """Configure Pyroscope continuous profiling."""
    server_address = os.environ.get("PYROSCOPE_SERVER_ADDRESS")
    if not server_address:
        return

    import pyroscope

    app_name = os.environ.get("PYROSCOPE_APPLICATION_NAME", "aquillm")
    pyroscope.configure(
        application_name=app_name,
        server_address=server_address,
        tags={"service": os.environ.get("OTEL_SERVICE_NAME", "aquillm")},
    )
    logger.info("Pyroscope profiling initialized (server=%s)", server_address)


def _init_pyroscope_celery_hook():
    """Re-initialize Pyroscope in each Celery forked worker process."""
    from celery.signals import worker_process_init

    @worker_process_init.connect
    def _on_worker_init(**kwargs):
        _init_pyroscope()


def _init_celery_user_tagging():
    """Propagate enduser.id from HTTP request spans to Celery task spans."""
    from celery.signals import before_task_publish, task_prerun
    from opentelemetry import trace

    @before_task_publish.connect
    def _inject_user_id(headers=None, **kwargs):
        if headers is None:
            return
        try:
            span = trace.get_current_span()
            if span.is_recording():
                user_id = span.attributes.get("enduser.id")
                if user_id:
                    headers["user_id"] = user_id
        except Exception:
            pass

    @task_prerun.connect
    def _extract_user_id(task=None, **kwargs):
        if task is None:
            return
        try:
            user_id = getattr(task.request, "user_id", None)
            if user_id is None:
                headers = getattr(task.request, "headers", None) or {}
                user_id = headers.get("user_id")
            if user_id:
                span = trace.get_current_span()
                if span.is_recording():
                    span.set_attribute("enduser.id", str(user_id))
        except Exception:
            pass


def setup():
    """Bootstrap observability — no-op unless OTEL_ENABLED=1."""
    if os.environ.get("OTEL_ENABLED", "").strip().lower() not in ("1", "true", "yes"):
        return

    try:
        _init_tracing()
    except Exception:
        logger.exception("Failed to initialize OpenTelemetry tracing")

    try:
        _init_pyroscope()
    except Exception:
        logger.exception("Failed to initialize Pyroscope profiling")

    try:
        _init_pyroscope_celery_hook()
    except Exception:
        logger.exception("Failed to initialize Pyroscope Celery hook")

    try:
        _init_celery_user_tagging()
    except Exception:
        logger.exception("Failed to initialize Celery user tagging")
