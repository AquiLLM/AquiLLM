"""Owned mirrored test DB with actual restricted role logins."""
from copy import deepcopy
import os
from threading import get_ident

import pytest
from django.db import connections
from django.db.backends.signals import connection_created

from aquillm.projection_database_settings import projection_databases


@pytest.fixture
def projection_roles():
    assert os.environ.get("RAG_REMEDIATION_TEST") == "1"
    default = connections["default"]
    default.ensure_connection()
    with default.cursor() as cursor:
        cursor.execute("SELECT current_database()")
        database, = cursor.fetchone()
    assert database == default.settings_dict["NAME"] == "test_rag_test"
    assert default.settings_dict["HOST"] == "db"
    assert str(default.settings_dict["PORT"]) == "5432"
    roles = {
        "projection_source": "aquillm_projection_source",
        "projection_state": "aquillm_projection_state",
    }
    configured = projection_databases(os.environ)
    originals, candidates, observed = {}, {}, []
    # Validate both candidates before opening either restricted alias.
    for alias, expected_role in roles.items():
        candidate = configured[alias]
        assert candidate["USER"] == expected_role
        assert candidate["HOST"] == default.settings_dict["HOST"]
        assert str(candidate["PORT"]) == str(default.settings_dict["PORT"])
        originals[alias] = connections.settings[alias]
        merged = deepcopy(originals[alias])
        for key in ("USER", "PASSWORD", "HOST", "PORT", "OPTIONS"):
            merged[key] = deepcopy(candidate[key])
        merged["NAME"] = database
        assert merged["TEST"]["MIRROR"] == "default"
        candidates[alias] = merged

    def assert_role(sender, connection, **kwargs):
        del sender, kwargs
        if connection.alias not in roles:
            return
        config = connection.settings_dict
        assert config["NAME"] == database
        assert config["HOST"] == default.settings_dict["HOST"]
        assert str(config["PORT"]) == str(default.settings_dict["PORT"])
        assert config["USER"] == roles[connection.alias]
        with connection.cursor() as cursor:
            cursor.execute("SELECT current_user, current_database()")
            role, actual_database = cursor.fetchone()
        assert (role, actual_database) == (roles[connection.alias], database)
        observed.append((connection.alias, get_ident(), role, actual_database))

    connection_created.connect(assert_role, weak=False)
    try:
        for alias in roles:
            connections[alias].close()
            del connections[alias]
            # New thread-local wrappers (including the heartbeat) use this config.
            connections.settings[alias] = candidates[alias]
        for alias in roles:
            connections[alias].ensure_connection()
        assert {row[0] for row in observed} == set(roles)
        yield observed
    finally:
        # Real worker heartbeat joins and closes its own connection before return.
        for alias in roles:
            connections[alias].close()
            del connections[alias]
            connections.settings[alias] = originals[alias]
        connection_created.disconnect(assert_role)
