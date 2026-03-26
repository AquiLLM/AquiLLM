# Kubernetes Support Services Deployment and Scaling Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a cloud-agnostic Kubernetes deployment model for AquiLLM support services with dual overlays (`portable`, `production`) and balanced default scaling behavior.

**Architecture:** Build a Kustomize-first manifest tree under `deploy/k8s/` with reusable base resources, overlay-specific environment wiring, phase-based rollout (compute then data tier), and scaling profiles tuned for balanced reliability/cost.

**Tech Stack:** Kubernetes YAML, Kustomize, HPA, PDB, optional KEDA, StatefulSets, Deployments, Services

---

## File Structure

| Path | Action | Responsibility |
|------|--------|----------------|
| `deploy/k8s/README.md` | Create | Usage guide for overlays, phases, and scaling profiles |
| `deploy/k8s/base/kustomization.yaml` | Create | Shared base aggregation |
| `deploy/k8s/base/namespace.yaml` | Create | Namespace definition |
| `deploy/k8s/base/config/app-configmap.yaml` | Create | Non-sensitive defaults |
| `deploy/k8s/base/config/app-secrets.example.yaml` | Create | Secret contract template |
| `deploy/k8s/base/web/*.yaml` | Create | Web deployment/service/HPA/PDB |
| `deploy/k8s/base/worker/*.yaml` | Create | Worker deployment/HPA/PDB |
| `deploy/k8s/base/vllm-*/{deployment,service}.yaml` | Create | Model support service workloads and Services |
| `deploy/k8s/base/services/*.yaml` | Create | Shared service manifests where not colocated |
| `deploy/k8s/base/scaling/*.yaml` | Create | Default balanced HPA/PDB policies |
| `deploy/k8s/overlays/portable/kustomization.yaml` | Create | Self-contained in-cluster mode |
| `deploy/k8s/overlays/portable/data/*.yaml` | Create | Postgres/Redis/MinIO/Qdrant workloads and services |
| `deploy/k8s/overlays/portable/patches/*.yaml` | Create | Portable-specific env + scaling patches |
| `deploy/k8s/overlays/production/kustomization.yaml` | Create | External data endpoint mode |
| `deploy/k8s/overlays/production/patches/*.yaml` | Create | Production env/scaling patches |
| `deploy/k8s/overlays/portable-keda/kustomization.yaml` | Create | Optional queue-depth autoscaling overlay |
| `deploy/k8s/components/keda/worker-scaledobject.yaml` | Create | KEDA ScaledObject for worker queue depth |
| `deploy/k8s/profiles/scaling/{balanced,high-reliability,cost-optimized}/*.yaml` | Create | Reusable scaling profile patches |
| `deploy/k8s/scripts/validate.ps1` | Create | Local validation helper |

---

## Chunk 1: Foundation and Contracts

### Task 1: Bootstrap Kubernetes Layout and Top-Level Guide

**Files:**
- Create: `deploy/k8s/README.md`
- Create: `deploy/k8s/base/`
- Create: `deploy/k8s/overlays/portable/`
- Create: `deploy/k8s/overlays/production/`
- Create: `deploy/k8s/profiles/scaling/`
- Create: `deploy/k8s/components/keda/`
- Create: `deploy/k8s/scripts/`

- [ ] **Step 1: Create directory scaffold**

Run:
```powershell
New-Item -ItemType Directory -Force deploy/k8s/base,deploy/k8s/base/config,deploy/k8s/base/web,deploy/k8s/base/worker,deploy/k8s/base/scaling,deploy/k8s/overlays/portable,deploy/k8s/overlays/portable/data,deploy/k8s/overlays/portable/patches,deploy/k8s/overlays/production,deploy/k8s/overlays/production/patches,deploy/k8s/overlays/portable-keda,deploy/k8s/profiles/scaling/balanced,deploy/k8s/profiles/scaling/high-reliability,deploy/k8s/profiles/scaling/cost-optimized,deploy/k8s/components/keda,deploy/k8s/scripts
```

- [ ] **Step 2: Write `deploy/k8s/README.md`**

Include:
- Overlay usage (`portable`, `production`, `portable-keda`)
- Phase rollout sequence (compute then data)
- Env mapping table for vLLM and data services
- Validation and dry-run commands
- Profile selection instructions for scaling patches

- [ ] **Step 3: Commit scaffold and docs**

```bash
git add deploy/k8s
git commit -m "chore(k8s): bootstrap kustomize layout for portable and production modes"
```

---

### Task 2: Build Base Namespace, Aggregation, and Config Contracts

**Files:**
- Create: `deploy/k8s/base/namespace.yaml`
- Create: `deploy/k8s/base/kustomization.yaml`
- Create: `deploy/k8s/base/config/app-configmap.yaml`
- Create: `deploy/k8s/base/config/app-secrets.example.yaml`

- [ ] **Step 1: Create namespace manifest**

Add `deploy/k8s/base/namespace.yaml` with namespace `aquillm` and standard labels:
- `app.kubernetes.io/name: aquillm`
- `app.kubernetes.io/part-of: aquillm-platform`
- `app.kubernetes.io/managed-by: kustomize`

- [ ] **Step 2: Create base `kustomization.yaml`**

Add resources for:
- `namespace.yaml`
- `config/*`
- `web/*`
- `worker/*`
- `vllm*/*`
- `scaling/*`

Set `namespace: aquillm` and add common labels.

- [ ] **Step 3: Create app ConfigMap contract**

Create `deploy/k8s/base/config/app-configmap.yaml` with non-sensitive defaults:
- `PORT`
- feature toggles
- provider mode defaults

Do not include credentials.

- [ ] **Step 4: Create example Secret contract**

Create `deploy/k8s/base/config/app-secrets.example.yaml` with keys and explicit `CHANGE_ME` sample values, including:
- provider API keys
- DB/storage credentials
- optional mem0 credentials

- [ ] **Step 5: Validate base rendering**

Run:
```bash
kubectl kustomize deploy/k8s/base
```
Expected: command exits 0 and emits a fully rendered manifest set.

- [ ] **Step 6: Commit base contracts**

```bash
git add deploy/k8s/base
git commit -m "feat(k8s): add base namespace and app config contracts"
```

---

## Chunk 2: Phase 1 - Compute Tier (Worker + Model Services)

### Task 3: Add Web and Worker Base Workloads with Balanced Defaults

**Files:**
- Create: `deploy/k8s/base/web/deployment.yaml`
- Create: `deploy/k8s/base/web/service.yaml`
- Create: `deploy/k8s/base/web/hpa.yaml`
- Create: `deploy/k8s/base/web/pdb.yaml`
- Create: `deploy/k8s/base/worker/deployment.yaml`
- Create: `deploy/k8s/base/worker/hpa.yaml`
- Create: `deploy/k8s/base/worker/pdb.yaml`

- [ ] **Step 1: Create `web` deployment/service**

Add web Deployment with:
- env from ConfigMap + Secret refs
- readiness/liveness probes (`/health/`)
- resource requests/limits

Add web Service (`ClusterIP`) on app port.

- [ ] **Step 2: Create `web` autoscaling and disruption policies**

Add:
- HPA with CPU + memory metrics
- PDB with `minAvailable: 1` in base (overlays may override)

- [ ] **Step 3: Create `worker` deployment**

Add worker Deployment with:
- explicit celery command
- env refs compatible with existing `.env` contract
- resource requests/limits

- [ ] **Step 4: Create fallback `worker` HPA + PDB**

Add:
- HPA CPU-based fallback (for no-KEDA clusters)
- PDB for graceful voluntary disruptions

- [ ] **Step 5: Validate rendering**

Run:
```bash
kubectl kustomize deploy/k8s/base | rg "kind: Deployment|kind: HorizontalPodAutoscaler|kind: PodDisruptionBudget"
```
Expected: web and worker resources appear exactly once each in base output.

- [ ] **Step 6: Commit web/worker base**

```bash
git add deploy/k8s/base/web deploy/k8s/base/worker
git commit -m "feat(k8s): add base web and worker workloads with balanced scaling defaults"
```

---

### Task 4: Add vLLM Service Family Deployments and Services

**Files:**
- Create: `deploy/k8s/base/vllm/deployment.yaml`
- Create: `deploy/k8s/base/vllm/service.yaml`
- Create: `deploy/k8s/base/vllm-ocr/deployment.yaml`
- Create: `deploy/k8s/base/vllm-ocr/service.yaml`
- Create: `deploy/k8s/base/vllm-transcribe/deployment.yaml`
- Create: `deploy/k8s/base/vllm-transcribe/service.yaml`
- Create: `deploy/k8s/base/vllm-embed/deployment.yaml`
- Create: `deploy/k8s/base/vllm-embed/service.yaml`
- Create: `deploy/k8s/base/vllm-rerank/deployment.yaml`
- Create: `deploy/k8s/base/vllm-rerank/service.yaml`

- [ ] **Step 1: Create `vllm` deployment + service**

Include:
- startup, readiness, liveness probes
- conservative default replica count (1)
- GPU scheduling hooks (`nodeSelector`, `tolerations`) as patch-friendly fields

- [ ] **Step 2: Create `vllm-ocr` deployment + service**

Include service name `vllm-ocr` (hyphenated DNS).

- [ ] **Step 3: Create `vllm-transcribe` deployment + service**

Include service name `vllm-transcribe`.

- [ ] **Step 4: Create `vllm-embed` deployment + service**

Include pooling/embedding environment variable wiring.

- [ ] **Step 5: Create `vllm-rerank` deployment + service**

Include rerank task wiring and service name `vllm-rerank`.

- [ ] **Step 6: Validate compute service DNS mapping**

Run:
```bash
kubectl kustomize deploy/k8s/base | rg "name: vllm$|name: vllm-ocr|name: vllm-transcribe|name: vllm-embed|name: vllm-rerank"
```
Expected: all five Service names render using Kubernetes-compatible hostnames.

- [ ] **Step 7: Commit vLLM family manifests**

```bash
git add deploy/k8s/base/vllm deploy/k8s/base/vllm-ocr deploy/k8s/base/vllm-transcribe deploy/k8s/base/vllm-embed deploy/k8s/base/vllm-rerank
git commit -m "feat(k8s): add base vllm support-service workloads and services"
```

---

### Task 5: Add Optional KEDA Worker Autoscaling Component

**Files:**
- Create: `deploy/k8s/components/keda/worker-scaledobject.yaml`
- Create: `deploy/k8s/overlays/portable-keda/kustomization.yaml`

- [ ] **Step 1: Create KEDA ScaledObject for worker**

Use Redis queue depth trigger and target worker Deployment.

- [ ] **Step 2: Create `portable-keda` overlay**

Include:
- `../portable` base overlay
- KEDA component resource
- patch to disable CPU fallback HPA if desired

- [ ] **Step 3: Validate optional KEDA overlay**

Run:
```bash
kubectl kustomize deploy/k8s/overlays/portable-keda
```
Expected: manifest includes `ScaledObject` and resolves without syntax errors.

- [ ] **Step 4: Commit KEDA component**

```bash
git add deploy/k8s/components/keda deploy/k8s/overlays/portable-keda
git commit -m "feat(k8s): add optional keda queue-depth autoscaling for worker"
```

---

## Chunk 3: Phase 2 - Data Tier (Portable In-Cluster + Production External)

### Task 6: Implement `portable` Overlay with In-Cluster Data Services

**Files:**
- Create: `deploy/k8s/overlays/portable/kustomization.yaml`
- Create: `deploy/k8s/overlays/portable/data/postgres-statefulset.yaml`
- Create: `deploy/k8s/overlays/portable/data/postgres-service.yaml`
- Create: `deploy/k8s/overlays/portable/data/redis-statefulset.yaml`
- Create: `deploy/k8s/overlays/portable/data/redis-service.yaml`
- Create: `deploy/k8s/overlays/portable/data/minio-statefulset.yaml`
- Create: `deploy/k8s/overlays/portable/data/minio-service.yaml`
- Create: `deploy/k8s/overlays/portable/data/qdrant-statefulset.yaml`
- Create: `deploy/k8s/overlays/portable/data/qdrant-service.yaml`
- Create: `deploy/k8s/overlays/portable/patches/app-env-portable.yaml`

- [ ] **Step 1: Create portable overlay aggregation**

`overlays/portable/kustomization.yaml` should include:
- `../../base`
- all `data/*.yaml`
- patches for portable env values

- [ ] **Step 2: Add in-cluster Postgres + service**

StatefulSet with PVC and basic readiness probe.

- [ ] **Step 3: Add in-cluster Redis + service**

StatefulSet or Deployment (pick StatefulSet for persistence consistency in this plan).

- [ ] **Step 4: Add in-cluster MinIO + service**

StatefulSet with PVC; ensure service host is `storage`.

- [ ] **Step 5: Add in-cluster Qdrant + service**

StatefulSet with PVC and service host `qdrant`.

- [ ] **Step 6: Add portable env patch wiring**

Set:
- `POSTGRES_HOST=db`
- `STORAGE_HOST=storage:9000`
- Redis host to in-cluster service
- `MEM0_QDRANT_HOST=qdrant`
- vLLM URLs to in-cluster service DNS names

- [ ] **Step 7: Validate portable overlay**

Run:
```bash
kubectl apply --dry-run=client -k deploy/k8s/overlays/portable
```
Expected: all resources validate client-side without CRD errors.

- [ ] **Step 8: Commit portable overlay**

```bash
git add deploy/k8s/overlays/portable
git commit -m "feat(k8s): add portable overlay with in-cluster data support services"
```

---

### Task 7: Implement `production` Overlay with External Data Endpoints

**Files:**
- Create: `deploy/k8s/overlays/production/kustomization.yaml`
- Create: `deploy/k8s/overlays/production/patches/app-env-production.yaml`
- Create: `deploy/k8s/overlays/production/patches/scaling-production.yaml`

- [ ] **Step 1: Create production overlay aggregation**

`overlays/production/kustomization.yaml` should include `../../base` and production patches only (no in-cluster data stateful workloads by default).

- [ ] **Step 2: Add production env patch**

Wire env values to external endpoints via Secret/ConfigMap references:
- managed DB host
- managed Redis host
- managed object storage endpoint
- managed or external Qdrant endpoint

- [ ] **Step 3: Add production scaling patch**

Set higher minima for user-path reliability:
- `web` min replicas > portable
- worker minimum tuned for baseline reliability
- optional higher PDB thresholds

- [ ] **Step 4: Validate production overlay**

Run:
```bash
kubectl apply --dry-run=client -k deploy/k8s/overlays/production
```
Expected: manifests render and validate without referencing portable-only resources.

- [ ] **Step 5: Commit production overlay**

```bash
git add deploy/k8s/overlays/production
git commit -m "feat(k8s): add production overlay for external data dependencies"
```

---

### Task 8: Add Reusable Scaling Profiles

**Files:**
- Create: `deploy/k8s/profiles/scaling/balanced/kustomization.yaml`
- Create: `deploy/k8s/profiles/scaling/balanced/patches.yaml`
- Create: `deploy/k8s/profiles/scaling/high-reliability/kustomization.yaml`
- Create: `deploy/k8s/profiles/scaling/high-reliability/patches.yaml`
- Create: `deploy/k8s/profiles/scaling/cost-optimized/kustomization.yaml`
- Create: `deploy/k8s/profiles/scaling/cost-optimized/patches.yaml`

- [ ] **Step 1: Implement balanced profile**

Set default HPA targets and min/max replicas aligned to approved policy.

- [ ] **Step 2: Implement high-reliability profile**

Increase minima and faster scale-up behavior for critical services.

- [ ] **Step 3: Implement cost-optimized profile**

Lower minima and slower scale-up for cost-sensitive deployments.

- [ ] **Step 4: Ensure profile composition works with overlays**

Add kustomize composition examples in README, e.g. profile patching via `resources`/`patches` strategy.

- [ ] **Step 5: Validate each profile render**

Run:
```bash
kubectl kustomize deploy/k8s/profiles/scaling/balanced
kubectl kustomize deploy/k8s/profiles/scaling/high-reliability
kubectl kustomize deploy/k8s/profiles/scaling/cost-optimized
```
Expected: each profile renders successfully and contains only patchable policy objects.

- [ ] **Step 6: Commit scaling profiles**

```bash
git add deploy/k8s/profiles/scaling deploy/k8s/README.md
git commit -m "feat(k8s): add reusable scaling profiles for balanced reliability and cost control"
```

---

## Chunk 4: Verification, Tooling, and Handoff

### Task 9: Add Validation Script and CI-Friendly Checks

**Files:**
- Create: `deploy/k8s/scripts/validate.ps1`
- Modify: `deploy/k8s/README.md`

- [ ] **Step 1: Create validation script**

Script should run:
- `kubectl kustomize` for base and overlays
- `kubectl apply --dry-run=client` for overlays
- optional schema lint (if tool exists)

- [ ] **Step 2: Document validation entrypoint**

Add README section:
- local validation command
- expected success output
- common failure troubleshooting

- [ ] **Step 3: Execute validation script locally**

Run:
```powershell
powershell -ExecutionPolicy Bypass -File deploy/k8s/scripts/validate.ps1
```
Expected: all configured checks pass or clearly report missing local dependencies.

- [ ] **Step 4: Commit validation tooling**

```bash
git add deploy/k8s/scripts/validate.ps1 deploy/k8s/README.md
git commit -m "chore(k8s): add manifest validation script and usage docs"
```

---

### Task 10: Perform End-to-End Manifest Verification

**Files:**
- Test: `deploy/k8s/base/*`
- Test: `deploy/k8s/overlays/portable/*`
- Test: `deploy/k8s/overlays/production/*`
- Test: `deploy/k8s/overlays/portable-keda/*`

- [ ] **Step 1: Render all bundles**

Run:
```bash
kubectl kustomize deploy/k8s/base
kubectl kustomize deploy/k8s/overlays/portable
kubectl kustomize deploy/k8s/overlays/production
kubectl kustomize deploy/k8s/overlays/portable-keda
```
Expected: no syntax errors.

- [ ] **Step 2: Client-side apply validation**

Run:
```bash
kubectl apply --dry-run=client -k deploy/k8s/overlays/portable
kubectl apply --dry-run=client -k deploy/k8s/overlays/production
```
Expected: success for all resources (except optional CRDs when KEDA is not installed and not used).

- [ ] **Step 3: Connectivity contract verification**

Check generated manifests include required env targets:
- `VLLM_BASE_URL`, `APP_OCR_QWEN_BASE_URL`, `INGEST_TRANSCRIBE_OPENAI_BASE_URL`, `APP_RERANK_BASE_URL`, `MEM0_EMBED_BASE_URL`, `MEM0_QDRANT_HOST`

Run:
```bash
kubectl kustomize deploy/k8s/overlays/portable | rg "VLLM_BASE_URL|APP_OCR_QWEN_BASE_URL|INGEST_TRANSCRIBE_OPENAI_BASE_URL|APP_RERANK_BASE_URL|MEM0_EMBED_BASE_URL|MEM0_QDRANT_HOST"
kubectl kustomize deploy/k8s/overlays/production | rg "VLLM_BASE_URL|APP_OCR_QWEN_BASE_URL|INGEST_TRANSCRIBE_OPENAI_BASE_URL|APP_RERANK_BASE_URL|MEM0_EMBED_BASE_URL|MEM0_QDRANT_HOST"
```
Expected: all keys present in both overlay render outputs.

- [ ] **Step 4: Commit final verification updates**

```bash
git add deploy/k8s
git commit -m "test(k8s): verify portable and production overlays and env contract wiring"
```

---

## Notes for Executors

- Follow this plan in order; Phase 2 depends on Phase 1 structure.
- Keep manifests small and responsibility-focused (one workload per directory).
- Prefer explicit, patchable defaults over large template indirection.
- Treat KEDA as optional: core overlays must work without it.
- If deploying to production-like clusters, define and bind real Secrets before apply.
