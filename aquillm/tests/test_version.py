"""Tests for canonical product version."""

import re

import pytest


SEMVER_RE = re.compile(
    r"^\d+\.\d+\.\d+(-[a-zA-Z0-9.]+)?(\+[a-zA-Z0-9.]+)?$"
)


@pytest.mark.unit
class TestVersion:
    def test_version_is_valid_semver(self):
        from aquillm import __version__

        assert SEMVER_RE.match(__version__), (
            f"__version__ {__version__!r} is not valid SemVer 2.0.0"
        )

    def test_version_matches_canonical(self):
        from aquillm import __version__
        from aquillm.version import VERSION

        assert __version__ == VERSION

    def test_version_available_in_settings(self):
        from django.conf import settings

        assert hasattr(settings, "AQUILLM_VERSION")
        assert settings.AQUILLM_VERSION == __import__("aquillm").version.VERSION
