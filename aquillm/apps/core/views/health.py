"""Bounded dependency probes, separate from the process liveness endpoint."""
import os

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.db import connections
from django.db.backends.base.base import NO_DB_ALIAS
from django.http import JsonResponse
from django.views.decorators.http import require_GET
from redis import Redis
from redis.backoff import NoBackoff
from redis.retry import Retry

from aquillm.ingestion.media import _openai_client


def database_ready():
    # A separate connection keeps probe timeouts away from application queries.
    # SELECT 1 needs no custom type registration. The no-DB alias prevents
    # PostgreSQL signals from opening the ordinary, unbounded app connection.
    # copy retains the configured target database and isolates this connection.
    connection = connections["default"].copy(alias=NO_DB_ALIAS)
    options = connection.settings_dict.setdefault("OPTIONS", {})
    options["connect_timeout"] = 2
    options["options"] = options.get("options", "") + " -c statement_timeout=1000"
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
        return True
    except Exception:
        return False
    finally:
        connection.close()


def broker_ready():
    client = None
    try:
        client = Redis.from_url(
            settings.CELERY_BROKER_URL, socket_connect_timeout=1,
            socket_timeout=1, retry_on_timeout=False, retry=Retry(NoBackoff(), 0),
        )
        return bool(client.ping())
    except Exception:
        return False
    finally:
        if client is not None:
            client.close()


def transcription_ready():
    if os.getenv("INGEST_TRANSCRIBE_PROVIDER", "").strip().lower() != "openai":
        return False
    client = None
    try:
        client = _openai_client().with_options(timeout=2, max_retries=0)
        model = os.getenv("INGEST_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe").strip()
        return any(item.id == model for item in client.models.list().data)
    except Exception:
        return False
    finally:
        if client is not None:
            client.close()


@require_GET
def readiness_check(request):
    dependencies = {"database": database_ready(), "broker": broker_ready()}
    ready = all(dependencies.values())
    return JsonResponse({"ready": ready, "dependencies": dependencies}, status=200 if ready else 503)


@login_required
@require_GET
def capabilities(request):
    # Optional providers do not take an otherwise healthy chat service offline.
    return JsonResponse({"transcription": {"available": transcription_ready()}})
