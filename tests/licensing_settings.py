"""Minimal, database-free settings for the public licensing endpoint tests."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "aquillm"))
SECRET_KEY = "licensing-tests-only"
ROOT_URLCONF = "apps.core.licensing_urls"
ALLOWED_HOSTS = ["testserver"]
INSTALLED_APPS = ["django.contrib.staticfiles"]
MIDDLEWARE = []
STATIC_URL = "/static/"
TEMPLATES = [{
    "BACKEND": "django.template.backends.django.DjangoTemplates",
    "DIRS": [REPO / "aquillm/templates"],
    "OPTIONS": {"context_processors": []},
}]
AQUILLM_SOURCE_URL = ""
