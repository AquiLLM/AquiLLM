# Observability (Logging, Monitoring, Continuous Profiling) — Design

**Goal:** Add better logging, monitoring, and continuous profiling to AquiLLM, with Grafana panels and Pyroscope, in both development and production compose setups.

**Scope:** Self-hosted stack added to Docker Compose; Django + Celery instrumented; Grafana dashboards for metrics and profiling.

---

## Approaches Considered

| Approach | Description | Trade-offs |
|----------|-------------|------------|
| **A. Metrics + profiling + structured logging** | Add Prometheus, Grafana, Pyroscope to compose. Instrument app with `prometheus_client` and Pyroscope Python agent. Improve logging with JSON formatter and configurable level. Dashboards for app metrics and Pyroscope. | **Recommended.** Delivers monitoring and profiling; logs remain file/console but structured for future aggregation. |
| **B. Same as A + Loki/Promtail** | Add Loki and Promtail for log aggregation; Grafana log panels. | Full observability in one go; more services and config. Can be Phase 2. |
| **C. Grafana Cloud + self-hosted Pyroscope** | Use Grafana Cloud for Prometheus/Loki, run only Pyroscope in Docker. | Less to operate; potential cost and vendor lock-in. |

**Choice:** **A** for initial implementation. Optionally add Loki/Promtail later (Phase 2) without changing app logging.

---

## Architecture

- **Prometheus:** Scrapes metrics from Django (HTTP `/metrics`) and Celery (worker metrics). Single Prometheus service in each compose.
- **Grafana:** One instance per environment. Datasources: Prometheus, Pyroscope. Provisioned via config files (no manual UI setup).
- **Pyroscope:** One server; Django (web) and Celery workers push profiles via Pyroscope Python SDK. Pyroscope stores and serves to Grafana.
- **Logging:** No new log shipping in Phase 1. Current file + console handlers kept; add a JSON formatter and `LOG_LEVEL` (and optional `LOG_JSON`) so output is consistent and ready for a future log shipper.

---

## Components

### 1. Compose services (dev and prod)

- **prometheus** — image `prom/prometheus:latest`, config via bind-mounted `deployment/prometheus.yml`, scrape interval e.g. 15s.
- **grafana** — image `grafana/grafana:latest`, persistent volume for data, env for admin password (or default in dev). Provisioning: `deployment/grafana/provisioning/datasources/` and dashboards in `deployment/grafana/provisioning/dashboards/` + dashboard JSONs.
- **pyroscope** — image `grafana/pyroscope:latest`, single instance; no extra scrape config (app pushes).

Ports (dev): e.g. Prometheus 9090, Grafana 3000, Pyroscope 4040. Prod: same internal ports; Grafana exposed via nginx or direct port with auth.

### 2. Application instrumentation

- **Django:**  
  - `django-prometheus` or manual `prometheus_client` with a `/metrics` view (no auth in dev; in prod protect or keep internal).  
  - Pyroscope: call `pyroscope.start()` (or equivalent) in `asgi.py`/`wsgi.py` so the web process is profiled.
- **Celery:**  
  - Celery Prometheus exporter (e.g. `celery-exporter` sidecar) or expose worker metrics from the worker process (e.g. `prometheus_client` on a small HTTP server in worker, or use a shared Prometheus file). Simpler option: use `django-prometheus` plus a Celery middleware/hook that increments task metrics, and expose `/metrics` from the web app that aggregates or re-exports worker-reported metrics — or run a dedicated exporter. **Practical choice:** `celery-exporter` as a sidecar container that scrapes Celery Flower or worker stats, or instrument worker with `prometheus_client` and a separate scrape target. Easiest: add `prometheus_client` to worker and expose a metrics HTTP server on a fixed port (e.g. 9091) from the worker container; Prometheus scrapes that.  
  - Pyroscope: start Pyroscope agent in the Celery worker entrypoint so worker process is profiled.
- **Logging:**  
  - New formatter: JSON (e.g. `{"time": "...", "level": "...", "logger": "...", "message": "..."}`).  
  - Handler selection: if `LOG_JSON=true`, use JSON formatter for file and/or console; else keep current formatters.  
  - `LOG_LEVEL` (e.g. DEBUG, INFO) applied to root or app loggers.

### 3. Grafana

- **Datasources (provisioned):**  
  - Prometheus: URL `http://prometheus:9090` (internal).  
  - Pyroscope: URL `http://pyroscope:4040` (internal).
- **Dashboards:**  
  - **App / Django:** Request rate, latency, errors (if exposed), and any custom metrics (e.g. ingest jobs, chat requests).  
  - **Celery:** Task rate, queue length, success/failure (from Prometheus metrics).  
  - **Pyroscope:** Single dashboard or panel set for continuous profiling (flame graphs, etc.) using Grafana’s Pyroscope datasource.

Store dashboard JSONs in repo under `deployment/grafana/provisioning/dashboards/` (or a `dashboards/` subdir) and reference them in dashboard provisioning YAML.

### 4. Production specifics

- **Compose:** Same three services (Prometheus, Grafana, Pyroscope); no need to expose Prometheus/Pyroscope publicly.
- **Grafana access:** Either (a) add a location in nginx that proxies to `grafana:3000` and protect with auth (e.g. basic auth or OAuth), or (b) expose port 3000 only on internal network and use SSH/VPN for access. Document chosen approach.
- **Secrets:** Grafana admin password and any API tokens from env (e.g. `.env`); do not commit secrets.
- **Metrics endpoint:** Keep `/metrics` internal (only reachable by Prometheus) or protect in nginx if exposed.

### 5. File/ownership

- **New/edited files:**  
  - `deployment/prometheus.yml`  
  - `deployment/grafana/provisioning/datasources/*.yml`  
  - `deployment/grafana/provisioning/dashboards/dashboards.yml` + JSON dashboards  
  - `.env.example`: `LOG_LEVEL`, `LOG_JSON`, `GRAFANA_ADMIN_PASSWORD`, optional `PYROSCOPE_*`  
  - `docker-compose-development.yml`, `docker-compose-prod.yml`: add `prometheus`, `grafana`, `pyroscope` services and wire `web`/`worker` env (e.g. `PYROSCOPE_SERVER_ADDRESS`, `PROMETHEUS_MULTIPROC_DIR` if needed).

---

## Success Criteria

- Dev: `docker compose -f docker-compose-development.yml up` brings up Prometheus, Grafana, Pyroscope; Grafana shows Prometheus and Pyroscope datasources; at least one dashboard each for app metrics and Pyroscope.
- Prod: Same stack in prod compose; Grafana accessible in a defined way (nginx or internal); no secrets in repo.
- Logs: Setting `LOG_LEVEL=DEBUG` and `LOG_JSON=true` produces JSON log lines; existing behavior preserved when vars unset.

---

## Out of Scope (Phase 2)

- Loki + Promtail for centralized logs and Grafana log panels.
- Alerting (Alertmanager, Grafana alerts).
- Tracing (OpenTelemetry).

---

## References

- [Prometheus Python client](https://github.com/prometheus/client_python)
- [Pyroscope Python SDK](https://github.com/grafana/pyroscope/tree/main/python)
- [Grafana provisioning](https://grafana.com/docs/grafana/latest/administration/provisioning/)
- [Grafana Pyroscope datasource](https://grafana.com/docs/grafana/latest/datasources/grafana-pyroscope/)
