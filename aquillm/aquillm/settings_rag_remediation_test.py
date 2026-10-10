"""Full installed registry, isolated PG aliases and filesystem-only artifacts."""
import os
from copy import deepcopy

# Guard before base settings import, instrumentation, or Django startup.
from scripts.run_rag_remediation_tests import validate_environment

validate_environment(os.environ)

from .settings import *  # noqa: F403,E402
from .settings import DATABASES as _BASE_DATABASES  # noqa: E402

DATABASES = deepcopy(_BASE_DATABASES)
DATABASES["default"]["TEST"] = {"NAME": "test_rag_test"}
for _alias in ("projection_source", "projection_state"):
    DATABASES[_alias] = {**deepcopy(DATABASES["default"]), "TEST": {"MIRROR": "default"}}

STORAGES = {
    "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
    "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
}
MEDIA_ROOT = "/artifacts/media"
# Keep delayed jobs queued; eager recursion is not representative worker behavior.
CELERY_TASK_ALWAYS_EAGER = False
CELERY_TASK_EAGER_PROPAGATES = True
CELERY_BROKER_URL = os.environ["CELERY_BROKER_URL"]
CELERY_RESULT_BACKEND = os.environ["CELERY_RESULT_BACKEND"]
PASSWORD_HASHERS = ["django.contrib.auth.hashers.MD5PasswordHasher"]
EMAIL_BACKEND = "django.core.mail.backends.locmem.EmailBackend"
STATICFILES_DIRS = [("js/dist", "/frontend-build")]
