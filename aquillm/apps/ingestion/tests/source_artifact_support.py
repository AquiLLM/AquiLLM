"""source artifact support: synthetic regression coverage."""

import pytest
from django.core.files.storage import FileSystemStorage


@pytest.fixture
def artifact_storage(settings, tmp_path):
    settings.STORAGES = {
        "default": {
            "BACKEND": "django.core.files.storage.FileSystemStorage",
            "OPTIONS": {"location": str(tmp_path)},
        },
        "staticfiles": {
            "BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"
        },
    }
    return FileSystemStorage(location=str(tmp_path))
