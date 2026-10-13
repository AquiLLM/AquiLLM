"""Expose only the sanctioned owned Redis target to maintenance tests."""
import os

import pytest

from scripts.run_rag_remediation_tests import IsolationError, validate_environment

_OWNED = "redis://redis:6379/2"


def maintenance_url(environment, configured_broker):
    if environment.get("RAG_REMEDIATION_TEST") != "1":
        return None
    validate_environment(environment)
    if configured_broker != _OWNED or environment.get("CELERY_BROKER_URL") != _OWNED:
        raise IsolationError("maintenance tests require the fixed owned broker")
    return _OWNED


@pytest.fixture
def owned_maintenance_redis(monkeypatch):
    from django.conf import settings

    url = maintenance_url(dict(os.environ), settings.CELERY_BROKER_URL)
    if url is not None:
        monkeypatch.setenv("REDIS_MAINTENANCE_TEST_URL", url)


class RedisBoundary:
    def __init__(self):
        self.now = 0
        self.values = {}
        self.expires = {}

    def get(self, key):
        if self.expires.get(key, float("inf")) <= self.now:
            self.values.pop(key, None)
        return self.values.get(key)

    def set(self, key, value, nx=False, ex=None):
        if nx and self.get(key) is not None:
            return False
        self.values[key] = str(value)
        if ex:
            self.expires[key] = self.now + ex
        return True

    def eval(self, script, count, lock, cursor, token, value):
        assert count == 2
        if self.get(lock) != token:
            return 0
        self.set(cursor, value)
        return 1



def depth(client, keys):
    return sum(client.llen(key) for key in keys)

