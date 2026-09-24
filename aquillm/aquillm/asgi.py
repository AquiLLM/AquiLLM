"""Django/Channels ASGI with default-off per-serving-worker capability lifespan."""

import os

from django.core.asgi import get_asgi_application


def create_application():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "aquillm.settings")
    asgi_app = get_asgi_application()
    # Channels auth and app routes require Django setup before importing models.
    from channels.auth import AuthMiddlewareStack
    from channels.routing import ProtocolTypeRouter, URLRouter
    from channels.security.websocket import AllowedHostsOriginValidator

    from apps.chat.routing import websocket_urlpatterns as chat_patterns
    from apps.documents.services.pair_worker_lifecycle import PairCapabilityLifespan
    from apps.ingestion.routing import websocket_urlpatterns as ingest_patterns

    from .routing import websocket_urlpatterns as crawl_status_patterns

    application = ProtocolTypeRouter(
        {
            "http": asgi_app,
            "websocket": AllowedHostsOriginValidator(
                AuthMiddlewareStack(
                    URLRouter(chat_patterns + ingest_patterns + crawl_status_patterns)
                )
            ),
        }
    )
    if os.environ.get("OTEL_ENABLED", "").strip().lower() in ("1", "true", "yes"):
        from opentelemetry.instrumentation.asgi import OpenTelemetryMiddleware

        application = OpenTelemetryMiddleware(application)
    # Lifespan runs after fork/reload in each worker, never in AppConfig.ready.
    return PairCapabilityLifespan(application)


application = create_application()
