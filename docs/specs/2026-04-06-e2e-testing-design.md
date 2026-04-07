# E2E Testing with Playwright via Django Test Infrastructure

## Summary

End-to-end tests using Playwright, run through Django's pytest-based test infrastructure with full database isolation. Tests execute inside Docker, use pytest-xdist for parallel workers with separate databases, and stub external services (Redis, Celery, MinIO, Channels) to prevent cross-worker collisions.

## Architecture

Each E2E test is a pytest test function that:

1. Gets a freshly flushed PostgreSQL database (via `TransactionTestCase` / `LiveServerTestCase` semantics — tables truncated between tests, not transaction rollback)
2. Creates test data via the Django ORM
3. Authenticates programmatically (force_login + cookie injection into Playwright)
4. Drives a Chromium browser via Playwright against Django's live test server
5. Asserts on page state via Playwright's locator API

Parallel execution is handled by **pytest-xdist**. Each worker gets its own test database (`test_gw0`, `test_gw1`, etc.), live server port, browser instance, and namespaced external services.

## Service Isolation

External services are overridden in tests to prevent cross-worker collisions:

| Service | Test Override | Reason |
|---------|--------------|--------|
| **Celery** | `CELERY_TASK_ALWAYS_EAGER = True`, `CELERY_TASK_EAGER_PROPAGATES = True` | Tasks run in-process, using the test's DB connection. No separate worker needed. |
| **Redis Cache** | `LocMemCache` backend | Per-process in-memory cache. Zero collision between workers. |
| **Channel Layer** | `InMemoryChannelLayer` | Per-process WebSocket groups. No Redis-based message leakage. |
| **MinIO/S3** | `InMemoryStorage` | Per-process in-memory file storage. No filesystem or S3 needed. |
| **Qdrant** | Per-worker collection name (`test_mem0_{worker_id}`) | Isolated vector storage. Collection created at worker startup, dropped at teardown. |

These overrides are applied via a session-scoped pytest fixture that uses `django.test.override_settings`.

## Fixtures & Test Lifecycle

All core fixtures live in `aquillm/tests/e2e/conftest.py`.

### Session-scoped (once per xdist worker)

- **Settings overrides**: Applies all service isolation overrides using the xdist `worker_id`.
- **Playwright browser**: Launches a single Chromium instance, shared across all tests in the worker.
- **Qdrant collection**: Creates `test_mem0_{worker_id}` at startup, drops at teardown.

### Function-scoped (fresh per test)

- **Playwright context + page**: New browser context per test (isolated cookies, storage). New page from that context.
- **Authenticated user**: Creates a `User` via ORM, logs in via `Client.force_login()`, extracts `sessionid` and `csrftoken` cookies, injects them into the Playwright browser context. Returns the user object for further ORM setup in the test.
- **Live server**: pytest-django's `live_server` fixture starts Django on a random free port. This requires the `transactional_db` fixture (which pytest-django auto-provides when `live_server` is requested), enabling flush-based isolation.

### Test lifecycle

1. Database flushed (tables truncated)
2. Fixtures create user, inject auth cookies into browser context
3. Test runs — Playwright drives browser against `live_server.url`
4. Browser context closed, fixtures torn down
5. Repeat

## File Structure

```
aquillm/tests/e2e/
├── conftest.py          # E2E fixtures
├── test_login.py        # Login flow
├── test_homepage.py     # Homepage, sidebar, navigation
├── test_collections.py  # Collection CRUD
├── test_chat.py         # Chat/conversation flows
├── test_documents.py    # Document viewing, chunks
├── test_search.py       # Search functionality
├── test_settings.py     # User settings
└── ...
```

All E2E tests are marked with `@pytest.mark.e2e` for selective execution.

## pytest Configuration

Add to `pytest.ini`:

```ini
markers =
    e2e: End-to-end browser tests
```

## Docker Integration

### Container requirements

Playwright and Chromium must be available in the web container. This requires:

- `playwright` and `pytest-xdist` added as test dependencies in `pyproject.toml`
- `playwright install --with-deps chromium` run during image build (or in a test-specific Dockerfile stage)

### Running tests

```bash
# All E2E tests, parallel
docker compose -f deploy/compose/no_gpu_dev.yml exec web pytest -m e2e -n auto

# Single test file
docker compose -f deploy/compose/no_gpu_dev.yml exec web pytest -m e2e aquillm/tests/e2e/test_login.py

# Exclude E2E when running unit/integration tests
docker compose -f deploy/compose/no_gpu_dev.yml exec web pytest -m "not e2e"
```

### Compose dependencies

- **Required**: `db` (PostgreSQL) — each worker creates its own test database
- **Required for memory tests**: `qdrant` — per-worker collections
- **Not needed**: `redis`, `storage` (MinIO), `worker` (Celery) — all stubbed

## Authentication

Most tests authenticate programmatically:

1. Create user via `User.objects.create_user()`
2. Log in via `django.test.Client().force_login(user)`
3. Extract `sessionid` cookie from the client
4. Inject cookie into Playwright browser context via `context.add_cookies()`

Tests that specifically test the login flow use Playwright to fill in the login form directly.

## Migration from Existing Playwright Tests

The 9 `.spec.ts` files in `react/tests/` (login, homepage, collections, search, user settings, sidebar navigation, feedback export, chat, collaboration management) serve as a reference for coverage targets.

- Existing tests continue to work via `npx playwright test` — no immediate removal
- As each flow gets equivalent coverage in `aquillm/tests/e2e/`, the corresponding `.spec.ts` file is removed
- `react/playwright.config.js` and `react/auth.json` are removed once migration is complete
