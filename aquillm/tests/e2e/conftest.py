import os
from functools import partial
from pathlib import Path

import pytest
from channels.testing.live import DaphneProcess, make_application, set_database_connection
from django.apps import apps
from django.contrib.auth import get_user_model
from django.contrib.staticfiles.handlers import ASGIStaticFilesHandler
from django.test import Client, override_settings
from playwright.sync_api import sync_playwright

SCREENSHOT_DIR = Path("/app/aquillm/tests/e2e/screenshots")


def pytest_addoption(parser):
    parser.addoption(
        "--e2e-screenshots",
        action="store_true",
        default=False,
        help="Capture screenshots on E2E test failures.",
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if (
        report.when == "call"
        and report.failed
        and item.config.getoption("--e2e-screenshots", default=False)
        and "e2e" in [m.name for m in item.iter_markers()]
    ):
        page = item.funcargs.get("page")
        if page and not page.is_closed():
            SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
            # Build a filename from the test node id: class::method → class--method.png
            safe_name = item.nodeid.split("::")[-2] + "--" + item.nodeid.split("::")[-1]
            path = SCREENSHOT_DIR / f"{safe_name}.png"
            try:
                page.screenshot(path=str(path), full_page=True)
                if hasattr(report, "extra"):
                    report.extra = getattr(report, "extra", [])
                print(f"\n📸 Screenshot saved: {path}")
            except Exception as exc:
                print(f"\n⚠️  Screenshot failed: {exc}")

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


class _ChannelsLiveServer:
    """Lightweight wrapper around a Daphne server process for ASGI/WebSocket support."""

    def __init__(self, host, port):
        self._host = host
        self._port = port

    @property
    def url(self):
        return f"http://{self._host}:{self._port}"

    @property
    def ws_url(self):
        return f"ws://{self._host}:{self._port}"


@pytest.fixture()
def channels_live_server(e2e_settings, transactional_db, echo_llm):
    """Start a Daphne ASGI server so WebSocket endpoints are reachable.

    Depends on ``echo_llm`` which patches the app config's llm_interface
    before Daphne forks — the child process inherits the patched object.

    Function-scoped because each test gets its own transactional database.
    """
    host = "localhost"
    get_application = partial(
        make_application,
        static_wrapper=ASGIStaticFilesHandler,
    )
    server = DaphneProcess(host, get_application, setup=set_database_connection)
    server.start()
    while not server.ready.wait(timeout=1):
        if not server.is_alive():
            raise RuntimeError("Daphne server stopped unexpectedly")
    yield _ChannelsLiveServer(host, server.port.value)
    server.terminate()
    server.join()


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


@pytest.fixture()
def authenticated_ws_user(db, context, channels_live_server):
    """Like authenticated_user but depends on channels_live_server for WebSocket support."""
    user = User.objects.create_user(
        username="e2e_test_user",
        email="e2e@example.com",
        password="testpass123",
    )
    client = Client()
    client.force_login(user)
    session_cookie = client.cookies["sessionid"]
    context.add_cookies([
        {
            "name": "sessionid",
            "value": session_cookie.value,
            "domain": "localhost",
            "path": "/",
        },
    ])
    return user


def _fake_embedding(query, input_type="search_query"):
    """Return a deterministic 1024-dim zero vector."""
    return [0.0] * 1024


def _fake_embeddings(queries, input_type="search_query"):
    """Return deterministic 1024-dim zero vectors for a batch."""
    return [[0.0] * 1024 for _ in queries]


@pytest.fixture()
def echo_llm():
    """Swap the app-wide LLM interface to the EchoLLM and stub embeddings.

    This patches the app config, the ChatConsumer class attribute, and the
    embedding functions so that WebSocket chat connections and document
    ingestion work without external API keys.  Because the Daphne server
    forks *after* this fixture runs, the child process inherits all patches.
    """
    import aquillm.utils as utils_mod
    from apps.chat.consumers.chat import ChatConsumer
    from apps.chat.tests.chat_message_test_support import _EchoLLMInterface

    app_cfg = apps.get_app_config("aquillm")
    original_llm = app_cfg.llm_interface

    echo = _EchoLLMInterface()
    app_cfg.llm_interface = echo
    ChatConsumer.llm_if = echo

    # Patch the imported embedding provider names on aquillm.utils so that
    # get_embedding() / get_embeddings() return zero vectors instead of
    # hitting external APIs.  These names were bound at import time via
    # ``from lib.embeddings import get_embedding_via_local_openai, ...``
    # so we must overwrite them on the utils module directly.
    _orig_local = utils_mod.get_embedding_via_local_openai
    _orig_local_batch = utils_mod.get_embeddings_via_local_openai
    _orig_cohere = utils_mod.get_embedding_via_cohere
    _orig_cohere_batch = utils_mod.get_embeddings_via_cohere

    utils_mod.get_embedding_via_local_openai = lambda q: [0.0] * 1024
    utils_mod.get_embeddings_via_local_openai = lambda qs: [[0.0] * 1024 for _ in qs]
    utils_mod.get_embedding_via_cohere = lambda c, q, t: [0.0] * 1024
    utils_mod.get_embeddings_via_cohere = lambda c, qs, t: [[0.0] * 1024 for _ in qs]

    yield echo

    app_cfg.llm_interface = original_llm
    ChatConsumer.llm_if = original_llm
    utils_mod.get_embedding_via_local_openai = _orig_local
    utils_mod.get_embeddings_via_local_openai = _orig_local_batch
    utils_mod.get_embedding_via_cohere = _orig_cohere
    utils_mod.get_embeddings_via_cohere = _orig_cohere_batch
