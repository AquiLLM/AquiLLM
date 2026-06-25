# OCR Fallback Deprecation vs Local Integration Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make OCR behavior reliable and local-first by choosing one clear path for Tesseract: fully deprecate it or fully operationalize it.

**Architecture:** Keep `vllm_ocr` (Qwen) as the primary OCR path and remove ambiguity in fallback behavior. Add an explicit decision gate, then implement either Track A (deprecate Tesseract and Gemini fallback from runtime flow) or Track B (keep Tesseract as a real local fallback with health checks and deploy guarantees).

**Tech Stack:** Django, Python (`lib.ocr`), Docker Compose, pytest.

---

## Decision Guidance (Recommendation First)

**Recommended:** Track A with a local-only policy (`qwen` primary, optional explicit `local` mode) and no implicit Gemini fallback in `auto`.

Why:
- Your stated direction is local services first.
- Qwen OCR is already your active local service path.
- Current Tesseract fallback is not operational in your real local environment, creating false safety.
- Removing hidden cloud fallback avoids policy drift and surprise costs.

Use Track B only if you require a non-LLM OCR fallback for outage resilience.

---

## File Structure and Ownership

| Path | Track A | Track B | Responsibility |
|---|---|---|---|
| `aquillm/lib/ocr/__init__.py` | Modify | Modify | Provider chain behavior and fallback order |
| `aquillm/lib/ocr/config.py` | Modify | Modify | Env parsing and fallback policy controls |
| `aquillm/lib/ocr/tesseract.py` | Optional remove/keep legacy shim | Modify | Local OCR execution and diagnostics |
| `aquillm/lib/ocr/tests/test_provider_selection.py` | Modify | Modify | Unit coverage for provider routing |
| `aquillm/tests/integration/test_ocr_runtime_policy.py` | Create | Create | Integration policy verification |
| `.env.example` | Modify | Modify | Operator-visible provider contract |
| `deploy/docker/web/Dockerfile` | Optional simplify | Keep + verify | Dev image local OCR dependencies |
| `deploy/docker/web/Dockerfile.prod` | Optional simplify | Keep + verify | Prod image local OCR dependencies |
| `deploy/compose/base.yml` | Modify | Modify | Environment defaults for OCR |
| `README.md` | Modify | Modify | OCR behavior, local-only policy, runbook |
| `docs/documents/operations/ocr.md` | Create | Create | Operational checks and troubleshooting |

---

## Chunk 1: Baseline and Decision Gate

### Task 1: Add an OCR behavior matrix test (fails first)

**Files:**
- Create: `aquillm/tests/integration/test_ocr_runtime_policy.py`
- Test: `aquillm/tests/integration/test_ocr_runtime_policy.py`

- [ ] **Step 1: Write failing tests for expected runtime policy**
- [ ] **Step 2: Cover these scenarios explicitly**
  - `APP_OCR_PROVIDER=qwen` uses Qwen only
  - `APP_OCR_PROVIDER=local` uses Tesseract only
  - `APP_OCR_PROVIDER=auto` matches chosen track behavior
- [ ] **Step 3: Run the new test file**

Run: `cd aquillm && pytest tests/integration/test_ocr_runtime_policy.py -q`  
Expected: FAIL until track implementation is done

- [ ] **Step 4: Commit**

```bash
git add aquillm/tests/integration/test_ocr_runtime_policy.py
git commit -m "test(ocr): add runtime provider policy matrix"
```

### Task 2: Decide and record the chosen track in-repo

**Files:**
- Create: `docs/documents/decisions/2026-03-20-ocr-fallback-direction.md`

- [ ] **Step 1: Write a short ADR-style decision note**
- [ ] **Step 2: Record chosen track and rollback strategy**
- [ ] **Step 3: Commit**

```bash
git add docs/documents/decisions/2026-03-20-ocr-fallback-direction.md
git commit -m "docs(decision): record OCR fallback direction"
```

---

## Chunk 2A: Track A (Deprecate Tesseract Fallback)

### Task 3A: Remove implicit non-local fallback from `auto`

**Files:**
- Modify: `aquillm/lib/ocr/__init__.py`
- Modify: `aquillm/lib/ocr/config.py`
- Test: `aquillm/lib/ocr/tests/test_provider_selection.py`

- [ ] **Step 1: Write failing unit tests asserting `auto` is local-only policy**
- [ ] **Step 2: Update provider flow**
  - `auto`: Qwen first
  - If Qwen fails: fail with clear error (or only allow `local` fallback when explicitly enabled by env flag)
  - Remove implicit Gemini fallback from runtime default path
- [ ] **Step 3: Run OCR provider tests**

Run: `cd aquillm && pytest lib/ocr/tests/test_provider_selection.py -q`  
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add aquillm/lib/ocr/__init__.py aquillm/lib/ocr/config.py aquillm/lib/ocr/tests/test_provider_selection.py
git commit -m "refactor(ocr): enforce local-only auto provider policy"
```

### Task 4A: Deprecate Tesseract code path cleanly

**Files:**
- Modify: `aquillm/lib/ocr/tesseract.py` (or remove + compatibility shim)
- Modify: `README.md`
- Create: `docs/documents/operations/ocr.md`

- [ ] **Step 1: Add deprecation warning path for `APP_OCR_PROVIDER=local`**
- [ ] **Step 2: Document migration to `qwen` as standard local OCR**
- [ ] **Step 3: Add operator troubleshooting for Qwen OCR service health**
- [ ] **Step 4: Run targeted integration tests**

Run: `cd aquillm && pytest tests/integration/test_ocr_runtime_policy.py -q`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aquillm/lib/ocr/tesseract.py README.md docs/documents/operations/ocr.md
git commit -m "docs(ocr): deprecate local tesseract fallback path"
```

### Task 5A: Simplify deploy artifacts (optional hardening)

**Files:**
- Modify: `deploy/docker/web/Dockerfile`
- Modify: `deploy/docker/web/Dockerfile.prod`

- [ ] **Step 1: Remove `tesseract-ocr` package install if no longer needed**
- [ ] **Step 2: Build images to verify no hidden runtime dependency remains**

Run: `docker compose -f deploy/compose/base.yml -f deploy/compose/development.yml build web`  
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add deploy/docker/web/Dockerfile deploy/docker/web/Dockerfile.prod
git commit -m "chore(deploy): remove unused tesseract dependency from web images"
```

---

## Chunk 2B: Track B (Integrate Tesseract Properly as Local Fallback)

### Task 3B: Make fallback eligibility explicit and testable

**Files:**
- Modify: `aquillm/lib/ocr/config.py`
- Modify: `aquillm/lib/ocr/__init__.py`
- Modify: `aquillm/lib/ocr/tests/test_provider_selection.py`

- [ ] **Step 1: Add env flag for fallback policy**
  - `APP_OCR_FALLBACK_CHAIN=qwen,local` (default local-only)
  - Optional: `APP_OCR_ALLOW_GEMINI_FALLBACK=0` (default off)
- [ ] **Step 2: Implement chain-driven routing in `extract_text_from_image`**
- [ ] **Step 3: Add tests for chain parsing and execution order**
- [ ] **Step 4: Run provider tests**

Run: `cd aquillm && pytest lib/ocr/tests/test_provider_selection.py -q`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aquillm/lib/ocr/config.py aquillm/lib/ocr/__init__.py aquillm/lib/ocr/tests/test_provider_selection.py
git commit -m "feat(ocr): add explicit local fallback chain policy"
```

### Task 4B: Add startup/runtime health checks for real local availability

**Files:**
- Modify: `aquillm/lib/ocr/tesseract.py`
- Create: `aquillm/lib/ocr/health.py`
- Create: `aquillm/lib/ocr/tests/test_health.py`
- Modify: `aquillm/aquillm/apps.py` (or existing startup hook)

- [ ] **Step 1: Write failing tests for dependency checks**
  - Detect `pytesseract`/`PIL` import readiness
  - Detect `tesseract --version` availability
- [ ] **Step 2: Add `check_local_ocr_ready()` health function**
- [ ] **Step 3: Warn loudly at startup when chain includes `local` but not ready**
- [ ] **Step 4: Run health and provider tests**

Run: `cd aquillm && pytest lib/ocr/tests/test_health.py lib/ocr/tests/test_provider_selection.py -q`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aquillm/lib/ocr/tesseract.py aquillm/lib/ocr/health.py aquillm/lib/ocr/tests/test_health.py aquillm/aquillm/apps.py
git commit -m "feat(ocr): add local tesseract readiness checks and startup warnings"
```

### Task 5B: Guarantee deploy/runtime dependencies

**Files:**
- Modify: `deploy/docker/web/Dockerfile`
- Modify: `deploy/docker/web/Dockerfile.prod`
- Modify: `deploy/compose/base.yml`
- Modify: `.env.example`
- Modify: `README.md`
- Create: `docs/documents/operations/ocr.md`

- [ ] **Step 1: Ensure `tesseract-ocr` install is retained and validated in both web images**
- [ ] **Step 2: Set local-first defaults in env docs (`auto` => `qwen,local`)**
- [ ] **Step 3: Document operator verification commands**
  - `python -c "import PIL,pytesseract; print('ok')"`
  - `tesseract --version`
  - OCR smoke endpoint/manual check
- [ ] **Step 4: Build and smoke test locally**

Run: `docker compose -f deploy/compose/base.yml -f deploy/compose/development.yml up -d --build web vllm_ocr`  
Expected: services healthy

- [ ] **Step 5: Run integration tests**

Run: `cd aquillm && pytest tests/integration/test_ocr_runtime_policy.py -q`  
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add deploy/docker/web/Dockerfile deploy/docker/web/Dockerfile.prod deploy/compose/base.yml .env.example README.md docs/documents/operations/ocr.md
git commit -m "chore(ocr): operationalize local tesseract fallback path"
```

---

## Chunk 3: Final Verification (Both Tracks)

### Task 6: Run end-to-end OCR verification and lock policy

**Files:**
- Modify: `README.md` (final behavior statement)
- Modify: `.env.example` (final default values)

- [ ] **Step 1: Run full OCR-related test suite**

Run: `cd aquillm && pytest lib/ocr/tests tests/integration/test_ocr_runtime_policy.py -q`  
Expected: PASS

- [ ] **Step 2: Run one real image OCR smoke test with chosen policy**
- [ ] **Step 3: Update docs to clearly state supported fallback modes**
- [ ] **Step 4: Commit**

```bash
git add README.md .env.example
git commit -m "docs(ocr): finalize supported local OCR policy"
```

---

## Rollout and Safety

- Roll out behind env-only policy toggles first.
- Keep `APP_OCR_PROVIDER=qwen` as stable immediate rollback.
- Add a short-lived release note with exact env settings for operations.


