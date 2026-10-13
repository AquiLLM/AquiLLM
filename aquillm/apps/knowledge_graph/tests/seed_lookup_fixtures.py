"""Pytest registration for the shared PostgreSQL seed fixture."""

import pytest

from .seed_lookup_factory import seeds as _build_seeds


@pytest.fixture
def seeds(request):
    return _build_seeds(request)
