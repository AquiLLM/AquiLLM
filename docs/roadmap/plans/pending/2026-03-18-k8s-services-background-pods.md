# K8s Services for Background Pods — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Kubernetes Service resources for background/workload pods so that web and worker pods can reach them by the same hostnames used in Compose (`vllm`, `vllm_ocr`, etc.) and so future K8s Deployments have a clear service topology.

**Architecture:** Introduce `deploy/k8s/` with per-workload Service manifests. Each background workload that web/worker call over HTTP gets a ClusterIP Service; K8s DNS uses hyphens (e.g. `vllm-ocr`), so deployment env must set base URL env vars to those hostnames. Worker is reached via Redis (Celery) and gets an optional Service only if we add HTTP health/metrics later.

**Tech Stack:** Kubernetes (plain YAML or kustomize), Docker images from `deploy/docker/`

**Context:** Current deployment is Docker Compose (`deploy/compose/base.yml`, `production.yml`). App and worker resolve: `vllm:8000`, `vllm_ocr:8000`, `vllm_transcribe:8000`, `vllm_embed:8000`, `vllm_rerank:8000` (see `.env`, `lib/ocr/config.py`, `lib/embeddings/config.py`, `apps/documents/models/chunks.py`, `ingestion/media.py`). No K8s manifests exist yet (`deploy/k8s/` is "Future" in the refactor spec).

---

## Scope: Which Pods Need Services?

| Workload        | Role              | Other pods call it?        | Service needed |
|----------------|-------------------|----------------------------|----------------|
| web            | User-facing API   | No (ingress calls web)     | Yes (for ingress) — out of scope here |
| worker         | Celery worker     | No (Redis broker)          | Optional (health/metrics later) |
| vllm           | Main LLM          | Yes (web, worker)          | **Yes** |
| vllm_ocr       | OCR (Qwen VL)     | Yes (web, worker)          | **Yes** |
| vllm_transcribe| ASR (Whisper)     | Yes (web, worker)          | **Yes** |
| vllm_embed     | Embeddings        | Yes (web, worker)          | **Yes** |
| vllm_rerank    | Reranker          | Yes (web, worker)          | **Yes** |
| db, redis, qdrant, storage | Data plane | Yes (web, worker) | Yes — out of scope (infra) |
| get_certs      | CronJob           | No                         | No |
| createbuckets  | Init Job          | No                         | No |

This plan focuses **only** on adding **Services** (and minimal Deployment stubs if missing) for the five vLLM-style background workloads so that when K8s Deployments are added, discovery works. Worker gets a short section for an optional Service.

---

## File Structure (after plan)

```
deploy/k8s/
├── base/                          # shared or default namespace/labels
│   └── kustomization.yaml         # (optional) if using kustomize
├── services/
│   ├── vllm-service.yaml
│   ├── vllm-ocr-service.yaml
│   ├── vllm-transcribe-service.yaml
│   ├── vllm-embed-service.yaml
│   ├── vllm-rerank-service.yaml
│   └── worker-service.yaml        # optional
└── README.md                      # when to use which service, naming convention
```

Deployments for vllm/worker can be added in a separate plan or later; this plan only adds the Service resources and a README so that "services for background pods" are defined in one place.

---

## Task 1: Create deploy/k8s directory and README

**Files:**
- Create: `deploy/k8s/README.md`
- Create: `deploy/k8s/services/` (directory)

- [ ] **Step 1: Create directory**

```bash
mkdir -p deploy/k8s/services
```

- [ ] **Step 2: Add README describing services for background pods**

Create `deploy/k8s/README.md` with:

```markdown
# AquiLLM Kubernetes Manifests

## Services for background pods

These Service manifests allow web and worker pods to reach background workloads by the same hostnames used in Docker Compose:

| Service name (K8s) | Port | Compose hostname | Env / use |
|--------------------|------|------------------|-----------|
| vllm               | 8000 | vllm             | VLLM_BASE_URL, main LLM |
| vllm-ocr           | 8000 | vllm_ocr         | APP_OCR_QWEN_BASE_URL (set to http://vllm-ocr:8000/v1 in K8s) |
| vllm-transcribe    | 8000 | vllm_transcribe  | INGEST_TRANSCRIBE_* (set to http://vllm-transcribe:8000/v1 in K8s) |
| vllm-embed         | 8000 | vllm_embed       | APP_EMBED_* / MEM0_EMBED_* (set to http://vllm-embed:8000/v1 in K8s) |
| vllm-rerank        | 8000 | vllm_rerank      | APP_RERANK_* (set to http://vllm-rerank:8000/v1 in K8s) |
| worker             | (optional) | worker      | Celery worker; optional Service for health later |

K8s DNS uses hyphens; Compose uses underscores. When deploying to K8s, set the base URL env vars above so the app resolves the correct Service names.

Apply with: `kubectl apply -f deploy/k8s/services/` (or use kustomize from `deploy/k8s/`).
```

- [ ] **Step 3: Commit**

```bash
git add deploy/k8s/README.md deploy/k8s/services/
git commit -m "chore(k8s): add deploy/k8s layout and README for background services"
```

---

## Task 2: Add ClusterIP Service for vllm

**Files:**
- Create: `deploy/k8s/services/vllm-service.yaml`

- [ ] **Step 1: Create vllm Service manifest**

Create `deploy/k8s/services/vllm-service.yaml`:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: vllm
  labels:
    app: aquillm
    component: vllm
spec:
  type: ClusterIP
  ports:
    - port: 8000
      targetPort: 8000
      protocol: TCP
      name: http
  selector:
    app: aquillm
    component: vllm
```

- [ ] **Step 2: Commit**

```bash
git add deploy/k8s/services/vllm-service.yaml
git commit -m "feat(k8s): add Service for vllm background pod"
```

---

## Task 3: Add ClusterIP Service for vllm_ocr

**Files:**
- Create: `deploy/k8s/services/vllm-ocr-service.yaml`

- [ ] **Step 1: Create vllm_ocr Service manifest**

Create `deploy/k8s/services/vllm-ocr-service.yaml`:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: vllm-ocr
  labels:
    app: aquillm
    component: vllm-ocr
spec:
  type: ClusterIP
  ports:
    - port: 8000
      targetPort: 8000
      protocol: TCP
      name: http
  selector:
    app: aquillm
    component: vllm-ocr
```

- [ ] **Step 2: Commit**

```bash
git add deploy/k8s/services/vllm-ocr-service.yaml
git commit -m "feat(k8s): add Service for vllm-ocr background pod"
```

---

## Task 4: Add ClusterIP Service for vllm-transcribe

**Files:**
- Create: `deploy/k8s/services/vllm-transcribe-service.yaml`

- [ ] **Step 1: Create vllm-transcribe Service manifest**

Create `deploy/k8s/services/vllm-transcribe-service.yaml`:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: vllm-transcribe
  labels:
    app: aquillm
    component: vllm-transcribe
spec:
  type: ClusterIP
  ports:
    - port: 8000
      targetPort: 8000
      protocol: TCP
      name: http
  selector:
    app: aquillm
    component: vllm-transcribe
```

- [ ] **Step 2: Commit**

```bash
git add deploy/k8s/services/vllm-transcribe-service.yaml
git commit -m "feat(k8s): add Service for vllm-transcribe background pod"
```

---

## Task 5: Add ClusterIP Service for vllm-embed

**Files:**
- Create: `deploy/k8s/services/vllm-embed-service.yaml`

- [ ] **Step 1: Create vllm-embed Service manifest**

Create `deploy/k8s/services/vllm-embed-service.yaml`:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: vllm-embed
  labels:
    app: aquillm
    component: vllm-embed
spec:
  type: ClusterIP
  ports:
    - port: 8000
      targetPort: 8000
      protocol: TCP
      name: http
  selector:
    app: aquillm
    component: vllm-embed
```

- [ ] **Step 2: Commit**

```bash
git add deploy/k8s/services/vllm-embed-service.yaml
git commit -m "feat(k8s): add Service for vllm-embed background pod"
```

---

## Task 6: Add ClusterIP Service for vllm-rerank

**Files:**
- Create: `deploy/k8s/services/vllm-rerank-service.yaml`

- [ ] **Step 1: Create vllm-rerank Service manifest**

Create `deploy/k8s/services/vllm-rerank-service.yaml`:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: vllm-rerank
  labels:
    app: aquillm
    component: vllm-rerank
spec:
  type: ClusterIP
  ports:
    - port: 8000
      targetPort: 8000
      protocol: TCP
      name: http
  selector:
    app: aquillm
    component: vllm-rerank
```

- [ ] **Step 2: Commit**

```bash
git add deploy/k8s/services/vllm-rerank-service.yaml
git commit -m "feat(k8s): add Service for vllm-rerank background pod"
```

---

## Task 7 (Optional): Add Service for worker

**Files:**
- Create: `deploy/k8s/services/worker-service.yaml` (optional)

Worker is a background pod but is not called over HTTP (Celery uses Redis). Add a Service only if you later expose an HTTP health or metrics endpoint on the worker. If added: ClusterIP, port as needed (e.g. 8080 for health), selector `app: aquillm`, `component: worker`. Otherwise skip this task.

- [ ] **Step 1 (optional): Create worker Service manifest** — only if worker will expose HTTP.
- [ ] **Step 2 (optional): Commit**

---

## Verification

After applying all manifests (or with a dry-run):

```bash
kubectl apply -f deploy/k8s/services/ --dry-run=client
```

Expected: list of 5 (or 6 with worker) Service resources created. Once Deployments exist with matching `app`/`component` labels, `kubectl get svc` in the same namespace should show the Services and endpoints when pods are running.
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>
StrReplace