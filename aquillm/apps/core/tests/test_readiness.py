from unittest.mock import Mock, patch

import pytest

from django.test import RequestFactory, SimpleTestCase
from django.urls import resolve


@pytest.mark.django_db(transaction=True)
def test_readiness_opens_a_real_registered_postgres_connection():
    from apps.core.views.health import database_ready
    assert database_ready()


@pytest.mark.django_db(transaction=True)
def test_cold_readiness_does_not_open_the_primary_connection():
    from apps.core.views.health import database_ready
    from django.contrib.postgres.signals import get_citext_oids, get_hstore_oids
    from django.db import connections

    get_citext_oids.cache_clear()
    get_hstore_oids.cache_clear()
    with patch.object(connections["default"], "cursor", side_effect=AssertionError("unbounded primary connection")) as primary:
        assert database_ready()
        primary.assert_not_called()


class ReadinessTests(SimpleTestCase):
    def test_readiness_reports_database_failure_while_liveness_survives(self):
        from apps.core.views import health
        with patch.object(health, "database_ready", return_value=False), patch.object(
            health, "broker_ready", return_value=True
        ):
            request = RequestFactory().get("/ready/")
            response = resolve("/ready/").func(request)
            self.assertEqual(response.status_code, 503)
            self.assertContains(response, '"database": false', status_code=503)
            self.assertEqual(resolve("/health/").func(request).status_code, 200)

    def test_readiness_requires_broker(self):
        from apps.core.views import health
        with patch.object(health, "database_ready", return_value=True), patch.object(
            health, "broker_ready", return_value=False
        ):
            self.assertEqual(health.readiness_check(RequestFactory().get("/ready")).status_code, 503)

    def test_optional_transcription_does_not_gate_core_readiness(self):
        from apps.core.views import health
        with patch.object(health, "database_ready", return_value=True), patch.object(
            health, "broker_ready", return_value=True
        ), patch.object(health, "transcription_ready", return_value=False) as transcribe:
            self.assertEqual(health.readiness_check(RequestFactory().get("/ready")).status_code, 200)
            transcribe.assert_not_called()

    def test_database_probe_is_isolated_and_bounded(self):
        from apps.core.views import health
        connection = Mock()
        connection.settings_dict = {"OPTIONS": {}}
        with patch.object(health, "connections") as connections:
            connections.__getitem__.return_value.copy.return_value = connection
            connection.cursor.return_value.__enter__ = Mock(return_value=Mock())
            connection.cursor.return_value.__exit__ = Mock(return_value=False)
            self.assertTrue(health.database_ready())
            self.assertEqual(connection.settings_dict["OPTIONS"]["connect_timeout"], 2)
            self.assertIn("statement_timeout=1000", connection.settings_dict["OPTIONS"]["options"])
            connection.close.assert_called_once()

    def test_transcription_probe_catches_unavailability(self):
        from apps.core.views import health
        with patch.object(health, "_openai_client", side_effect=OSError("unavailable")), patch.dict(
            "os.environ", {"INGEST_TRANSCRIBE_PROVIDER": "openai"}
        ):
            self.assertFalse(health.transcription_ready())

    def test_broker_probe_has_finite_timeouts_and_no_retry(self):
        from apps.core.views import health
        with patch.object(health.Redis, "from_url") as factory:
            factory.return_value.ping.side_effect = OSError("unavailable")
            self.assertFalse(health.broker_ready())
            self.assertEqual(factory.call_args.kwargs["socket_connect_timeout"], 1)
            self.assertEqual(factory.call_args.kwargs["socket_timeout"], 1)
            self.assertEqual(factory.call_args.kwargs["retry"]._retries, 0)
            factory.return_value.close.assert_called_once()

    def test_capability_endpoint_requires_authentication(self):
        from apps.core.views import health
        from django.contrib.auth.models import AnonymousUser
        request = RequestFactory().get("/api/capabilities/")
        request.user = AnonymousUser()
        with patch.object(health, "transcription_ready") as probe:
            self.assertEqual(health.capabilities(request).status_code, 302)
            probe.assert_not_called()
