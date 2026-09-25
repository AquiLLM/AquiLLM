"""Test-only shared database for projection ORM/seed parity contracts.

These tests exercise application query identity and authorization. They do not
validate the deployment's distinct PostgreSQL roles or state-function grants.
Production settings retain their separate, fail-closed projection aliases.
"""

from copy import deepcopy

from .settings import *  # noqa: F403
from .settings import DATABASES as _BASE_DATABASES

DATABASES = deepcopy(_BASE_DATABASES)
for _alias in ("projection_source", "projection_state"):
    DATABASES[_alias] = {
        **deepcopy(DATABASES["default"]),
        "TEST": {"MIRROR": "default"},
    }
