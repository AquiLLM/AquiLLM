# Semantic Versioning Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish a single canonical SemVer product version for AquiLLM, align frontend metadata and documentation, and document the release/tag/changelog workflow so MAJOR/MINOR/PATCH bumps are consistent and auditable.

**Architecture:** Keep versioning lightweight: one version source for the Python package root, mirror to `react/package.json` at release time, expose via Django settings and structured logs for operators. No requirement to publish to PyPI or npm for this effort unless already part of release process.

**Tech Stack:** Python (Django), npm (React), Git tags, Markdown docs.

**Depends on:**

- `docs/specs/2026-03-29-semantic-versioning-design.md`

---

## Scope

### In scope

- Add a **single canonical version** for the product (e.g. `aquillm/aquillm/version.py` or `VERSION` at repo root—pick one pattern and document it).
- Set **`__version__`** (or import from canonical file) from that source; wire **Django `settings`** if useful for templates/health.
- Align **`react/package.json`** `version` field with the policy (same triple at release boundaries).
- Add **contributor-facing documentation** (short section in `CONTRIBUTING.md` or `docs/` release notes) describing bump rules and tag format `vX.Y.Z`.
- Document **changelog** expectations relative to `docs/changelog/README.md`.
- Optional: **pytest** asserting version string matches SemVer regex to prevent typos.

### Out of scope

- Automated GitHub Releases or CI publish pipelines (can be a follow-up unless trivially adding a manual checklist).
- Changing dependency pinning policy in `requirements.txt` / lockfiles.
- API URL versioning or new HTTP headers (separate spec if needed).

---

## Proposed file structure

| Path | Action | Responsibility |
|------|--------|------------------|
| `aquillm/aquillm/version.py` (or repo-root `VERSION`) | Create | Canonical `VERSION` string |
| `aquillm/aquillm/__init__.py` | Modify | Export `__version__` from canonical source |
| `aquillm/aquillm/settings.py` | Modify | `AQUILLM_VERSION` setting for templates/health |
| `aquillm/aquillm/settings_logging.py` | Modify | Add version processor to structured logs |
| `react/package.json` | Modify | Match product version at initial adoption |
| `CONTRIBUTING.md` | Create | SemVer bump + tag + changelog checklist |
| `aquillm/tests/test_version.py` | Create | Format assertion |

---

## Chunk 1: Canonical version module

### Task 1: Add canonical version and export

**Files:**

- Create: `aquillm/aquillm/version.py` (recommended: single constant `VERSION = "0.1.0"` or align with current team choice)
- Modify: `aquillm/aquillm/__init__.py`

- [ ] **Step 1:** Create `version.py` with `VERSION` constant (SemVer string, no `v` prefix in the constant).
- [ ] **Step 2:** In `__init__.py`, set `__version__` from `version.VERSION` (or `importlib.metadata` if you later package with metadata; start simple).
- [ ] **Step 3:** Run a quick import check: `cd aquillm && python -c "from aquillm import __version__; print(__version__)"`.

---

## Chunk 2: Frontend and docs alignment

### Task 2: Align React package version and contributor docs

**Files:**

- Modify: `react/package.json`
- Modify: `CONTRIBUTING.md` (or new `docs/documents/releases/semantic-versioning.md` if no CONTRIBUTING exists—prefer minimal new files)

- [ ] **Step 1:** Set `react/package.json` `version` to the same MAJOR.MINOR.PATCH as `VERSION` (initial adoption).
- [ ] **Step 2:** Document: tag format `vX.Y.Z`, when to bump MAJOR/MINOR/PATCH per design spec, and that changelog under `docs/changelog/` should record user-visible changes per release or branch policy.
- [ ] **Step 3:** Add a one-line pointer from `docs/changelog/README.md` to the semver doc if helpful.

---

## Chunk 3: Structured logging and settings

### Task 3: Attach version to structured logs and Django settings

**Files:**

- Modify: `aquillm/aquillm/settings_logging.py`
- Modify: `aquillm/aquillm/settings.py`

- [ ] **Step 1:** In `settings_logging.py`, import `VERSION` from `aquillm.version` and add a processor function `add_product_version` that sets `event_dict["version"] = VERSION`. Insert it into the `shared_processors` list so every log event (structlog and stdlib) gets the version field.
- [ ] **Step 2:** In `settings.py`, import `VERSION` from `aquillm.version` and set `AQUILLM_VERSION = VERSION` for use in templates/health views.
- [ ] **Step 3:** Verify logs include the version field by starting the dev server and checking JSON output.

---

## Chunk 4: Tests

### Task 4: Version format test

**Files:**

- Create: `aquillm/tests/test_version.py`

- [ ] **Step 1:** Add test that `aquillm.__version__` matches SemVer 2.0.0 pattern (`^\d+\.\d+\.\d+(-[a-zA-Z0-9.]+)?(\+[a-zA-Z0-9.]+)?$`).
- [ ] **Step 2:** Add test that `aquillm.version.VERSION` equals `aquillm.__version__`.
- [ ] **Step 3:** Run `pytest aquillm/tests/test_version.py -q`.

---

## Chunk 5: Verification and handoff

### Task 5: Validate release discipline

- [ ] **Step 1:** Confirm `docs/specs/README.md` lists this spec and plan with status Planned.
- [ ] **Step 2:** Dry-run: describe in docs the manual steps to tag `vX.Y.Z` after bumping `VERSION` and `react/package.json` together.
- [ ] **Step 3:** Commit with message like `docs(release): adopt semantic versioning policy and canonical version`.

---

## Exit criteria

- [ ] Canonical version exists and is importable as `aquillm.__version__`.
- [ ] `react/package.json` matches the adopted baseline version.
- [ ] Written instructions exist for contributors (bump + tag + changelog).
- [ ] Version test passes.
- [ ] Product version appears in structured log output.
- [ ] `AQUILLM_VERSION` is available in Django settings.
