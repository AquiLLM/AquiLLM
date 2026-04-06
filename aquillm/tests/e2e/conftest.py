import os

import pytest
from django.contrib.auth import get_user_model
from django.test import Client, override_settings
from playwright.sync_api import sync_playwright

# pytest-asyncio (auto mode) starts an event loop for the session.  Django's
# sync-safety guard then raises SynchronousOnlyOperation when pytest-django
# tries to create/tear-down the test database from within that loop context.
# E2E tests use synchronous Playwright only, so it is safe to allow sync DB
# access from the event loop.
os.environ.setdefault("DJANGO_ALLOW_ASYNC_UNSAFE", "true")

User = get_user_model()


def get_worker_id(request):
    """Return xdist worker id, or 'master' when running without xdist."""
    if hasattr(request.config, "workerinput"):
        return request.config.workerinput["workerid"]
    return "master"


@pytest.fixture(scope="session")
def e2e_settings(request):
    """Apply E2E-specific Django settings overrides for the entire worker session."""
    worker_id = get_worker_id(request)
    overrides = override_settings(
        CELERY_TASK_ALWAYS_EAGER=True,
        CELERY_TASK_EAGER_PROPAGATES=True,
        CACHES={
            "default": {
                "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
                "LOCATION": f"e2e-{worker_id}",
            }
        },
        CHANNEL_LAYERS={
            "default": {
                "BACKEND": "channels.layers.InMemoryChannelLayer",
            }
        },
        STORAGES={
            "default": {
                "BACKEND": "django.core.files.storage.InMemoryStorage",
            },
            "staticfiles": {
                "BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage",
            },
        },
    )
    overrides.enable()
    # Set Qdrant collection name via environment so config_builder picks it up
    import os
    old_collection = os.environ.get("MEM0_COLLECTION_NAME")
    os.environ["MEM0_COLLECTION_NAME"] = f"test_mem0_{worker_id}"
    yield
    os.environ.pop("MEM0_COLLECTION_NAME", None)
    if old_collection is not None:
        os.environ["MEM0_COLLECTION_NAME"] = old_collection
    overrides.disable()


@pytest.fixture(scope="session")
def browser(e2e_settings):
    """Launch a single Chromium instance per xdist worker."""
    pw = sync_playwright().start()
    browser = pw.chromium.launch(headless=True)
    yield browser
    browser.close()
    pw.stop()


@pytest.fixture()
def context(browser):
    """Create a fresh browser context per test (isolated cookies/storage)."""
    ctx = browser.new_context()
    yield ctx
    ctx.close()


@pytest.fixture()
def page(context):
    """Create a fresh page per test."""
    pg = context.new_page()
    yield pg
    pg.close()


@pytest.fixture()
def authenticated_user(db, context, live_server):
    """Create a user, force-login, and inject session cookie into browser context.

    Returns the User instance so tests can do further ORM setup.
    The live_server fixture is declared as a dependency to ensure the server
    is running (it also implicitly provides transactional_db).
    """
    user = User.objects.create_user(
        username="e2e_test_user",
        email="e2e@example.com",
        password="testpass123",
    )
    # Force login via Django test client to get a session
    client = Client()
    client.force_login(user)
    # Extract session cookie from the Django test client
    session_cookie = client.cookies["sessionid"]
    # Inject into Playwright browser context
    context.add_cookies([
        {
            "name": "sessionid",
            "value": session_cookie.value,
            "domain": "localhost",
            "path": "/",
        },
    ])
    return user
