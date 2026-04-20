# Kubernetes Support Services Deployment and Scaling - Design Spec

**Date:** 2026-03-19  
**Last reviewed:** 2026-03-25 (aligned with current repo layout and deployment baseline)  
**Status:** Approved design â€” **implementation not started** (no `deploy/k8s/` tree in repository yet)  
**Goal:** Make AquiLLM support services deployable on any Kubernetes environment using a dual-mode model (`portable` and `production`) with balanced reliability/cost scaling defaults.

**Related artifacts**

- Implementation plan: [`docs/roadmap/plans/pending/2026-03-19-kubernetes-support-services-deployment-scaling.md`](../roadmap/plans/pending/2026-03-19-kubernetes-support-services-deployment-scaling.md) (Kustomize layout, tasks, optional KEDA overlay `overlays/portable-keda`, `components/keda/`)
- Portfolio status: [`docs/roadmap/roadmap-status.md`](../roadmap/roadmap-status.md) (marks this workstream as not started)
- Codebase refactor spec: [`docs/specs/2026-03-18-codebase-refactor-design.md`](2026-03-18-codebase-refactor-design.md) â€” application code now lives under `aquillm/apps/` and `aquillm/lib/`; Kubernetes remains a future deployment track (was listed as `deploy/k8s/` future in that document).

---

## Current repository baseline (pre-Kubernetes)

**Canonical runtime today:** Docker Compose under `deploy/compose/`, with environment from repo `.env`.

| Area | Location | Notes |
|------|----------|--------|
| Base stack | `deploy/compose/base.yml` | `web`, `worker`, `db` (Postgres), `storage` (MinIO), `redis`, `createbuckets`, optional GPU profile services `vllm`, `vllm_ocr`, `vllm_transcribe`, `vllm_embed`, `vllm_rerank` |
| Production-style compose | `deploy/compose/production.yml` | Adds **Qdrant** and other production-oriented wiring not present in `base.yml` alone |
| Development compose | `deploy/compose/development.yml` | Extended dev stack including Qdrant |
| Images | `deploy/docker/web/`, `deploy/docker/vllm/`, etc. | Same images should be referenced from Kubernetes workloads when manifests are added |

**Application packaging:** Django/Celery entrypoints and env-driven configuration are unchanged by this design; overlays only supply different hostnames and secrets.

**Gap vs this spec:** There is **no** `deploy/k8s/` directory yet (globally: no Kustomize bases, overlays, or CI validation for kube manifests). The architecture below remains the target; the implementation plan lists the concrete files to add.

---

## Overview

This design introduces a cloud-agnostic Kubernetes deployment model for AquiLLM support services in two phases:

1. **Phase 1 (Compute Tier):** Deploy and scale async/model support services (`worker`, `vllm`, `vllm-ocr`, `vllm-transcribe`, `vllm-embed`, `vllm-rerank`).
2. **Phase 2 (Data Tier):** Support both in-cluster and external managed data services (`Postgres`, `Redis`, `MinIO`, `Qdrant`) via overlays.

The deployment will be **Kustomize-first** with:

- `base`: shared, portable primitives
- `overlays/portable`: in-cluster dependencies enabled
- `overlays/production`: external managed dependencies preferred
- `profiles/scaling`: reusable scaling and disruption policy defaults
- **Optional:** `overlays/portable-keda` + `components/keda/` for Redis queue-depth autoscaling (see implementation plan; keeps HPA-only path when KEDA is not installed)

This keeps manifests portable across managed cloud clusters and self-managed clusters with the same repository layout and commands.

---

## Architecture

### 1) Manifest Topology

Proposed repository layout (matches implementation plan; not yet present in repo):

```text
deploy/k8s/
  base/
    namespace.yaml
    common-labels.yaml   # optional; may be folded into kustomization labels
    web/
    worker/
    vllm/
    vllm-ocr/
    vllm-transcribe/
    vllm-embed/
    vllm-rerank/
    services/            # shared Services where not colocated with workload dirs
    scaling/             # default balanced HPA/PDB fragments (per implementation plan)
    config/              # e.g. app ConfigMap + secrets example
    kustomization.yaml
  overlays/
    portable/
      kustomization.yaml
      patches/
      config/
      data/              # in-cluster Postgres, Redis, MinIO, Qdrant (portable mode)
    production/
      kustomization.yaml
      patches/
      config/
    portable-keda/       # optional: compose on top of portable for KEDA worker scaling
  components/
    keda/
      worker-scaledobject.yaml   # optional ScaledObject for Celery queue depth
  profiles/
    scaling/
      balanced/
      high-reliability/
      cost-optimized/
```

### 2) Dual-Mode Operation

- **Portable mode:** installs in-cluster stateful support services for a self-contained deployment.
- **Production mode:** points web/worker/model consumers to managed or externally operated endpoints where available.

Mode switching is done by changing overlay target only:

- `kubectl apply -k deploy/k8s/overlays/portable`
- `kubectl apply -k deploy/k8s/overlays/production`

### 3) Service Discovery and Env Contract

The app is **env-driven**. Today, Compose DNS uses **underscore** service names (`vllm_ocr`, `vllm_embed`, â€¦), and `.env` examples match that (for example `APP_OCR_QWEN_BASE_URL=http://vllm_ocr:8000/v1`, `MEM0_EMBED_BASE_URL=http://vllm_embed:8000/v1`).

Kubernetes `Service` names should follow **RFC 1123** hostnames (hyphens, not underscores). Overlays must set env values to **Kubernetes DNS names** so the same variables resolve correctly:

| Concern | Example env vars (non-exhaustive) | Compose-style host (today) | Kubernetes Service host (target) |
|--------|-----------------------------------|----------------------------|------------------------------------|
| Chat / main vLLM | `VLLM_BASE_URL`, Mem0 LLM URLs | `vllm` | `vllm` |
| OCR | `APP_OCR_QWEN_BASE_URL` | `vllm_ocr` | `vllm-ocr` |
| Transcription | `INGEST_TRANSCRIBE_OPENAI_BASE_URL` | `vllm_transcribe` | `vllm-transcribe` |
| Rerank | `APP_RERANK_BASE_URL` | `vllm_rerank` | `vllm-rerank` |
| Embeddings | `MEM0_EMBED_BASE_URL`, `APP_EMBED_BASE_URL` | `vllm_embed` | `vllm-embed` |
| Qdrant | `MEM0_QDRANT_HOST` (and port) | `qdrant` (when compose brings it up) | `qdrant` or managed endpoint |
| Postgres | `POSTGRES_HOST` | `db` in base compose | `db` or external hostname |
| Object storage | `STORAGE_HOST` | `storage:9000` | in-cluster MinIO Service or external |
| Redis | Celery broker URL / cache | `redis` | in-cluster or external |

Example overlay values (illustrative):

- `VLLM_BASE_URL=http://vllm:8000/v1`
- `APP_OCR_QWEN_BASE_URL=http://vllm-ocr:8000/v1`
- `INGEST_TRANSCRIBE_OPENAI_BASE_URL=http://vllm-transcribe:8000/v1`
- `APP_RERANK_BASE_URL=http://vllm-rerank:8000/v1`
- `MEM0_EMBED_BASE_URL=http://vllm-embed:8000/v1`
- `MEM0_QDRANT_HOST=qdrant` (portable) or managed endpoint host (production)

**No application code changes are required** if overlays rewrite URLs/hostnames to match Kubernetes Services; code already reads these variables (see `aquillm/lib/ocr/config.py`, `aquillm/aquillm/ingestion/media.py`, `aquillm/lib/embeddings/config.py`, `aquillm/lib/memory/mem0/client.py`, `aquillm/apps/documents/services/chunk_rerank_config.py`).

---

## Scaling Design

### Scaling Priorities

Primary policy target is **balanced reliability + cost discipline**:

- Avoid underprovisioning that harms availability
- Avoid expensive, aggressive default scale-out (especially GPU workloads)

### 1) Web Deployment

- `HPA` on CPU and memory in both overlays
- Higher `minReplicas` in `production` than `portable`
- `PDB` to preserve availability during voluntary disruptions

### 2) Worker Deployment

- Prefer `KEDA` (Redis queue depth) for demand-based scaling â€” wired as an **optional** overlay/component per implementation plan
- Fallback to standard `HPA` when KEDA is not installed
- Optional future worker pool separation (ingestion-heavy vs default) kept out of initial scope

### 3) vLLM Service Family

For `vllm`, `vllm-ocr`, `vllm-transcribe`, `vllm-embed`, `vllm-rerank`:

- Default to conservative replica counts (typically 1)
- Support opt-in horizontal scale via overlay patches
- Add `startupProbe`, `readinessProbe`, and `livenessProbe` with long startup windows
- Use GPU node selectors/tolerations and anti-affinity when replicas > 1

Compose already uses HTTP `/health` checks for vLLM containers (`deploy/compose/base.yml`); Kubernetes probes should align with the same endpoint and generous `start_period`-style timing.

### 4) Data Services

- `portable`: deploy in-cluster stateful services with explicit PVC sizes and resource envelopes
- `production`: external managed services preferred, with in-cluster stateful fallback optional
- Include backup/restore integration points in design for stateful tiers
- **Note:** Qdrant appears in `production.yml` / `development.yml` compose files but not in `base.yml`; portable Kubernetes overlay should make Qdrant availability explicit for Mem0 paths that default `MEM0_QDRANT_HOST` to `qdrant`.

---

## Security and Configuration Model

### Secrets and Config

- `ConfigMap`: non-sensitive defaults and runtime knobs
- `Secret`: API keys, DB credentials, storage credentials, provider tokens
- Overlay-specific secret references so production can bind environment-specific credentials safely

### Network and Isolation

- Dedicated namespace (default `aquillm`)
- Consistent `app.kubernetes.io/*` labels across all resources
- Optional `NetworkPolicy` templates included as a hardening baseline

---

## Error Handling and Resilience

### Failure Modes and Responses

1. **Model startup delay/failure**
- Use generous startup probes and staggered rollouts
- Keep worker retry behavior bounded and observable

2. **Queue/service spikes**
- KEDA/HPA scales workers based on queue depth or CPU fallback
- Maintain sane `maxReplicas` guardrails to avoid runaway spend

3. **Data service endpoint mismatch**
- Overlay-specific configuration validation and dry-run checks before apply
- Explicit startup health verification for endpoint reachability

4. **Node disruptions**
- `PDB` for user-path and worker services
- Anti-affinity for multi-replica critical services

---

## Testing and Verification Strategy

### Static Validation

- `kubectl kustomize` on each overlay
- `kubectl apply --dry-run=client` for all generated manifests
- Schema linting with kubeconform/kubeval (if available in CI)

### Runtime Validation

- Smoke deploy portable overlay in a local or ephemeral cluster
- Validate service discovery from web/worker to support services
- Validate scaling triggers:
  - web CPU load
  - worker queue depth (KEDA path)
  - worker CPU fallback path
- Validate probe behavior for long model startup

### Operational Checks

- Verify endpoint DNS names and env wiring post-deploy (including **hyphenated** Service names vs legacy Compose underscore hosts)
- Verify pod scheduling constraints for GPU services
- Verify backup hook execution paths for stateful services in portable mode

---

## Scope Boundaries

In scope:

- Kubernetes manifests/design for support services and their scaling
- Overlay-based dual-mode deployment model
- Balanced default autoscaling and resilience policies

Out of scope for this spec:

- Full app-level business feature changes
- Provider-specific managed-service provisioning (Terraform/CloudFormation)
- Multi-region topology and active-active replication

---

## Success Criteria

1. `portable` overlay deploys a fully functional AquiLLM support stack in-cluster.
2. `production` overlay deploys app/model/worker support with external data dependencies.
3. Service hostname/env mapping works **without application code changes** (overlay-only URL/host rewrites from Compose-style names to Kubernetes Services).
4. Balanced scaling defaults improve reliability under load without excessive GPU cost growth.
5. The same repo artifacts can be applied on both managed and self-managed Kubernetes environments.
6. Repository contains `deploy/k8s/` with documented `kubectl`/Kustomize entrypoints and repeatable static validation (per implementation plan).




