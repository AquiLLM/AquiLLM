# Observability (Logging, Monitoring, Pyroscope + Grafana) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add structured logging, Prometheus metrics, continuous profiling with Pyroscope, and Grafana dashboards to AquiLLM; wire the stack in both development and production Docker Compose.

**Architecture:** Prometheus scrapes Django (`/metrics`) and (optionally) Celery worker metrics; Pyroscope receives profiles from web and worker; Grafana is provisioned with Prometheus + Pyroscope datasources and dashboards. Logging gets a JSON formatter and `LOG_LEVEL`/`LOG_JSON` env support.

**Tech Stack:** Django, Celery, `prometheus_client`, `pyroscope` (Python), Prometheus, Grafana, Pyroscope (containers), Docker Compose.

---

## Conventions

- Observability config lives under `deployment/` (Prometheus, Grafana provisioning).
- Env-driven: `LOG_LEVEL`, `LOG_JSON`, `GRAFANA_ADMIN_PASSWORD`, `PYROSCOPE_SERVER_ADDRESS` (optional); observability services can be started in both dev and prod compose.
- Preserve existing log handlers; add JSON formatter and conditional handler config.

---

## Task 1: Add Prometheus, Grafana, and Pyroscope to development compose

**Files:**
- Modify: `docker-compose-development.yml`
- Create: `deployment/prometheus.yml` (Prometheus config)

**Step 1: Create Prometheus config**

Create `deployment/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'django'
    static_configs:
      - targets: ['web:8080']
    metrics_path: /metrics
    scrape_interval: 15s
  - job_name: 'pyroscope'
    static_configs:
      - targets: ['pyroscope:4040']
    scrape_interval: 15s
```

**Step 2: Add services to docker-compose-development.yml**

Append to `docker-compose-development.yml` (before `volumes:`), adding `prometheus`, `grafana`, and `pyroscope` services. Ensure `web` and `worker` have no port conflict (Prometheus scrapes `web:8080`; if PORT is different, use `${PORT:-8080}` in the config or set `PROMETHEUS_SCRAPE_PORT`). Add:

- **prometheus:** image `prom/prometheus:latest`, volumes `./deployment/prometheus.yml:/etc/prometheus/prometheus.yml`, command `--config.file=/etc/prometheus/prometheus.yml --storage.tsdb.path=/prometheus --web.enable-lifecycle`, ports `9090:9090`, restart `unless-stopped`.
- **grafana:** image `grafana/grafana:latest`, environment `GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD:-admin}`, `GF_USERS_ALLOW_SIGN_UP=false`, `GF_SERVER_ROOT_URL`, volumes for provisioning (see Task 6), port `3000:3000`, restart `unless-stopped`.
- **pyroscope:** image `grafana/pyroscope:latest`, ports `4040:4040`, restart `unless-stopped`.

Add a named volume for Prometheus data (e.g. `prometheus_data`) and for Grafana if desired (`grafana_data`). Add these volumes under the root `volumes:` section.

**Step 3: Point web service to Pyroscope (optional env)**

In `web` and `worker` services, add environment:
`PYROSCOPE_SERVER_ADDRESS: http://pyroscope:4040`
(So app can send profiles when enabled.)

**Step 4: Commit**

```bash
git add deployment/prometheus.yml docker-compose-development.yml
git commit -m "chore: add Prometheus, Grafana, Pyroscope to development compose"
```

---

## Task 2: Add Python dependencies for metrics and profiling

**Files:**
- Modify: `requirements.txt` (or the file used by Dockerfile.prod)

**Step 1: Add packages**

Append to `requirements.txt`:

```
prometheus_client
pyroscope
```

**Step 2: Commit**

```bash
git add requirements.txt
git commit -m "chore: add prometheus_client and pyroscope deps"
```

---

## Task 3: Expose Django /metrics and optional Pyroscope in web process

**Files:**
- Modify: `aquillm/aquillm/urls.py`
- Create: `aquillm/aquillm/metrics.py` (or use django-prometheus; this plan uses raw prometheus_client for minimal deps)
- Modify: `aquillm/aquillm/settings.py` (LOGGING + optional Pyroscope)
- Modify: `aquillm/aquillm/asgi.py` (start Pyroscope agent when env set)

**Step 1: Create metrics module**

Create `aquillm/aquillm/metrics.py`:

```python
"""Prometheus metrics for Django app."""
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from django.http import HttpResponse

REQUEST_COUNT = Counter(
    "django_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "django_http_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)


def metrics_view(request):
    """Expose Prometheus metrics."""
    return HttpResponse(
        generate_latest(),
        content_type=CONTENT_TYPE_LATEST,
    )
```

**Step 2: Add /metrics URL**

In `aquillm/aquillm/urls.py`, add:
`path("metrics/", views.metrics_view, name="metrics"),`
and ensure `metrics_view` is imported from a module that has it (e.g. create `aquillm/aquillm/views_metrics.py` that imports from `aquillm.aquillm.metrics` and re-exports `metrics_view`, or add `metrics_view` to existing `views` or a small metrics view module and import in urls). Prefer: in `aquillm/aquillm/urls.py` add `from .metrics import metrics_view` and `path("metrics/", metrics_view, name="metrics")`.

**Step 3: Add middleware to record request count and latency (optional but recommended)**

Create a thin middleware in `aquillm/aquillm/middleware.py` that calls `REQUEST_COUNT.labels(method=..., endpoint=..., status=...).inc()` and `REQUEST_LATENCY.labels(...).observe(duration)`, and add it to `MIDDLEWARE` in settings. Use a simple endpoint name (e.g. path prefix or view name) to avoid high cardinality.

**Step 4: Start Pyroscope in ASGI (when env set)**

In `aquillm/aquillm/asgi.py`, at the top (after `import os`), add:

```python
import os
_pyroscope_server = os.environ.get("PYROSCOPE_SERVER_ADDRESS")
if _pyroscope_server:
    try:
        import pyroscope
        pyroscope.configure(
            application_name="aquillm-web",
            server_address=_pyroscope_server,
        )
    except Exception:
        pass
```

Then load Django and build the ASGI app as usual.

**Step 5: Run app and hit /metrics**

Run the dev stack (or `python aquillm/manage.py runserver` with deps installed). `curl http://localhost:8080/metrics/` (or your PORT) should return Prometheus text.

**Step 6: Commit**

```bash
git add aquillm/aquillm/metrics.py aquillm/aquillm/urls.py aquillm/aquillm/asgi.py aquillm/aquillm/middleware.py aquillm/aquillm/settings.py
git commit -m "feat: add /metrics endpoint and optional Pyroscope in web process"
```

---

## Task 4: Structured logging (JSON formatter, LOG_LEVEL, LOG_JSON)

**Files:**
- Modify: `aquillm/aquillm/settings.py`

**Step 1: Add JSON formatter and env-driven config**

In `settings.py`:
- Define `LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()` and `LOG_JSON = env_bool("LOG_JSON", False)`.
- In `LOGGING["formatters"]`, add a `"json"` formatter that outputs a single JSON line per record (e.g. `{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}` — use proper escaping for JSON; or use a small custom formatter that builds a dict and `json.dumps`).
- In `LOGGING["handlers"]`, add a handler (e.g. `"console_json"`) that uses the JSON formatter and `StreamHandler`, or switch existing `"console"` formatter based on `LOG_JSON`.
- Apply `LOG_LEVEL` to the relevant loggers (e.g. `logging.getLogger().setLevel(LOG_LEVEL)` or set in each logger's `level` in the config). Prefer setting `level` in handler and logger config from `LOG_LEVEL`.

**Step 2: Use LOG_LEVEL in loggers**

Set `"level": LOG_LEVEL` (or the string from env) for `aquillm`, `chat`, `celery`, `ingest` loggers so that changing `LOG_LEVEL` changes verbosity.

**Step 3: Commit**

```bash
git add aquillm/aquillm/settings.py
git commit -m "feat: add JSON log formatter and LOG_LEVEL/LOG_JSON env support"
```

---

## Task 5: Celery worker metrics and Pyroscope (optional worker profiling)

**Files:**
- Modify: `docker-compose-development.yml` (worker command to start Pyroscope then Celery)
- Create: `aquillm/celery_profiler.py` or integrate in Celery app (start Pyroscope before worker)

**Step 1: Start Pyroscope in worker process**

Option A: In the Celery app module (e.g. `aquillm/aquillm/celery.py` or where `celery` is instantiated), at import time, if `PYROSCOPE_SERVER_ADDRESS` is set, call `pyroscope.configure(application_name="aquillm-worker", server_address=...)`. Then the worker command stays `celery -A aquillm worker ...`.

Option B: Change worker command to a wrapper script that starts Pyroscope then execs Celery. Prefer Option A.

In `aquillm/aquillm/celery.py` (or the module that defines `app`), add at top:

```python
import os
if os.environ.get("PYROSCOPE_SERVER_ADDRESS"):
    try:
        import pyroscope
        pyroscope.configure(
            application_name="aquillm-worker",
            server_address=os.environ["PYROSCOPE_SERVER_ADDRESS"],
        )
    except Exception:
        pass
```

**Step 2: Commit**

```bash
git add aquillm/aquillm/celery.py docker-compose-development.yml
git commit -m "feat: enable Pyroscope profiling in Celery worker when env set"
```

---

## Task 6: Grafana provisioning (datasources + dashboards)

**Files:**
- Create: `deployment/grafana/provisioning/datasources/datasources.yml`
- Create: `deployment/grafana/provisioning/dashboards/dashboards.yml`
- Create: `deployment/grafana/provisioning/dashboards/json/aquillm-metrics.json` (and optionally `pyroscope-flame.json` or use built-in Pyroscope panels)

**Step 1: Datasources**

Create `deployment/grafana/provisioning/datasources/datasources.yml`:

```yaml
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
  - name: Pyroscope
    type: grafana-pyroscope-datasource
    access: proxy
    url: http://pyroscope:4040
```

**Step 2: Dashboard provisioning**

Create `deployment/grafana/provisioning/dashboards/dashboards.yml`:

```yaml
apiVersion: 1
providers:
  - name: default
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 30
    options:
      path: /etc/grafana/provisioning/dashboards/json
```

**Step 3: Mount provisioning in Grafana service**

In `docker-compose-development.yml`, for the `grafana` service, add volumes:

```yaml
volumes:
  - ./deployment/grafana/provisioning/datasources:/etc/grafana/provisioning/datasources
  - ./deployment/grafana/provisioning/dashboards:/etc/grafana/provisioning/dashboards
  - grafana_data:/var/lib/grafana
```

And add volume `grafana_data` to the root `volumes:` list.

**Step 4: Create a simple dashboard JSON**

Create `deployment/grafana/provisioning/dashboards/json/aquillm-metrics.json` with a minimal dashboard that uses the Prometheus datasource: one panel with a query such as `django_http_requests_total` (or `up{job="django"}`) so the dashboard loads. Export from Grafana UI or write minimal JSON:

```json
{
  "title": "AquiLLM Metrics",
  "uid": "aquillm-metrics",
  "panels": [
    {
      "id": 1,
      "title": "Django requests",
      "type": "graph",
      "datasource": "Prometheus",
      "targets": [{ "expr": "django_http_requests_total", "legendFormat": "{{method}} {{endpoint}}" }]
    }
  ],
  "schemaVersion": 27,
  "version": 1
}
```

(Adjust panel format to match Grafana schema if needed; Grafana 10+ may use different panel types like "timeseries".)

**Step 5: Commit**

```bash
git add deployment/grafana/
git commit -m "feat: add Grafana datasource and dashboard provisioning"
```

---

## Task 7: Add observability stack to production compose

**Files:**
- Modify: `docker-compose-prod.yml`

**Step 1: Add same services as development**

Add `prometheus`, `grafana`, and `pyroscope` services to `docker-compose-prod.yml` with the same images, config mounts, and internal ports. Do not expose Grafana on 0.0.0.0 in prod unless nginx is added to protect it; either expose only on a private network or add an nginx location that proxies to `grafana:3000` with auth.

**Step 2: Set web/worker env for prod**

Ensure `web` and `worker` in prod have `PYROSCOPE_SERVER_ADDRESS: http://pyroscope:4040` so profiling works when stack is up.

**Step 3: (Optional) Nginx route for Grafana**

If you want to access Grafana via the same host (e.g. `/grafana/`), add a location in `deployment/nginx.conf` or `deployment/aquillm.conf.template` that proxies to `http://grafana:3000` and protect with auth. Document in README.

**Step 4: Commit**

```bash
git add docker-compose-prod.yml
git commit -m "chore: add Prometheus, Grafana, Pyroscope to production compose"
```

---

## Task 8: Environment and documentation

**Files:**
- Modify: `.env.example`
- Modify: `README.md` (or docs) with observability section

**Step 1: .env.example**

Add:

```
# Observability (optional)
LOG_LEVEL=INFO
LOG_JSON=false
GRAFANA_ADMIN_PASSWORD=admin
PYROSCOPE_SERVER_ADDRESS=
```

Document that setting `PYROSCOPE_SERVER_ADDRESS=http://pyroscope:4040` (in compose env) enables profiling.

**Step 2: README observability section**

Add a short section describing: how to start the stack with observability (compose up), URLs for Grafana (e.g. http://localhost:3000), Prometheus (http://localhost:9090), and Pyroscope (http://localhost:4040); and that LOG_LEVEL/LOG_JSON control logging.

**Step 3: Commit**

```bash
git add .env.example README.md
git commit -m "docs: add observability env and README section"
```

---

## Verification

- Development: `docker compose -f docker-compose-development.yml up -d` (with web, worker, prometheus, grafana, pyroscope). Open Grafana at http://localhost:3000, log in with admin / GRAFANA_ADMIN_PASSWORD; confirm Prometheus and Pyroscope datasources; open "AquiLLM Metrics" dashboard and see data after generating traffic. Hit http://localhost:8080/metrics and see Prometheus output. Set LOG_JSON=true and LOG_LEVEL=DEBUG, restart web; logs should be JSON and verbose.
- Production: Same compose (or prod compose) runs; Grafana is reachable per deployment (internal or via nginx).

---

## Execution options

**Plan complete and saved to `docs/plans/2026-03-16-observability-implementation-plan.md`.**

Two execution options:

1. **Subagent-driven (this session)** — Dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Parallel session (separate)** — Open a new session with executing-plans and run through the plan with checkpoints.

Which approach do you prefer?
