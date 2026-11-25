"""Shared synthetic bytes and module loading for bounded storage tests."""
import importlib

import pytest


DATA = b"exact attached PDF bytes"


def storage_service():
    try:
        return importlib.import_module("apps.ingestion.services.source_storage")
    except ModuleNotFoundError as exc:
        if exc.name != "apps.ingestion.services.source_storage":
            raise
        pytest.fail("bounded native source capture is not implemented")
