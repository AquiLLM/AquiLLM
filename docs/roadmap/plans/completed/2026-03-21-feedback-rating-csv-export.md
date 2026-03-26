# Feedback Rating CSV Export Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collect and export feedback data with the required columns: `date`, `user_number`, `rating`, `question_number`, `comments`, and make it easily downloadable as CSV.

**Architecture:** Use the existing `apps.chat.models.Message` rating/comment fields as the source of truth, add a dedicated feedback timestamp, compute `question_number` from conversation message order, and expose a staff-only CSV export endpoint. Add a simple UI download affordance in admin-facing React UI.

**Tech Stack:** Django 5.1, Django ORM, Channels chat consumer, React/TypeScript, pytest.

---

## Data Contract

CSV columns (exact order):
1. `date` (feedback submission datetime in ISO 8601 UTC)
2. `user_number` (Django user ID)
3. `rating` (1-5)
4. `question_number` (1-based user prompt index within conversation)
5. `comments` (free text, CSV-escaped)

Row eligibility:
- Include assistant messages with `rating IS NOT NULL` or non-empty `feedback_text`.

---

## Chunk 1: Schema + Feedback Capture Hardening

### Task 1: Add feedback timestamp field to message model

**Files:**
- Modify: `aquillm/apps/chat/models/message.py`
- Create: `aquillm/apps/chat/migrations/0002_message_feedback_submitted_at.py`

- [ ] **Step 1: Add nullable indexed field**

```python
feedback_submitted_at = models.DateTimeField(null=True, blank=True, db_index=True)
```

- [ ] **Step 2: Create migration**

Run: `cd aquillm && python manage.py makemigrations apps_chat`
Expected: migration file created with `AddField`.

- [ ] **Step 3: Commit**

```bash
git add aquillm/apps/chat/models/message.py aquillm/apps/chat/migrations/0002_message_feedback_submitted_at.py
git commit -m "feat(chat): add feedback submission timestamp to messages"
```

### Task 2: Update chat feedback/rating writes to set timestamp and validate input

**Files:**
- Modify: `aquillm/apps/chat/consumers/chat.py`
- Create: `aquillm/apps/chat/services/feedback.py`
- Create: `aquillm/apps/chat/tests/test_feedback_capture.py`

- [ ] **Step 1: Move rating/feedback update logic into service function**
- [ ] **Step 2: Validate rating bounds (`1..5`) and sanitize/truncate comment length**
- [ ] **Step 3: Set `feedback_submitted_at=timezone.now()` on both `rate` and `feedback` actions**
- [ ] **Step 4: Add tests for timestamp update and validation**

Run: `cd aquillm && pytest apps/chat/tests/test_feedback_capture.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add aquillm/apps/chat/consumers/chat.py aquillm/apps/chat/services/feedback.py aquillm/apps/chat/tests/test_feedback_capture.py
git commit -m "feat(chat): persist validated feedback updates with submission timestamp"
```

---

## Chunk 2: Export Query + CSV Endpoint

### Task 3: Build export service that computes `question_number`

**Files:**
- Create: `aquillm/apps/platform_admin/services/feedback_export.py`
- Create: `aquillm/apps/platform_admin/tests/test_feedback_export_service.py`

- [ ] **Step 1: Implement export row query**

Rules:
- `user_number`: `conversation.owner_id`
- `date`: `feedback_submitted_at` if present, else `created_at`
- `question_number`: count of `role='user'` messages in same conversation with `sequence_number <= assistant_message.sequence_number`
- `comments`: `feedback_text` or empty string

- [ ] **Step 2: Add optional filters**

Filters via params:
- `start_date` (inclusive)
- `end_date` (inclusive)
- `min_rating` (optional)
- `user_number` (optional)

- [ ] **Step 3: Test question-number correctness and filter behavior**

Run: `cd aquillm && pytest apps/platform_admin/tests/test_feedback_export_service.py -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add aquillm/apps/platform_admin/services/feedback_export.py aquillm/apps/platform_admin/tests/test_feedback_export_service.py
git commit -m "feat(platform-admin): add feedback export query service with question numbering"
```

### Task 4: Add downloadable CSV API endpoint (superuser-only)

**Files:**
- Modify: `aquillm/apps/platform_admin/views/api.py`
- Modify: `aquillm/apps/platform_admin/urls.py`
- Modify: `aquillm/apps/platform_admin/views/pages.py`
- Modify: `aquillm/aquillm/api_views.py` (compat route export)
- Modify: `aquillm/aquillm/context_processors.py` (expose API URL to frontend)
- Create: `aquillm/apps/platform_admin/tests/test_feedback_csv_export_api.py`

- [ ] **Step 1: Add superuser-only endpoint**

Endpoint:
- `GET /api/feedback/ratings.csv`
- name: `api_feedback_ratings_csv`
- response headers:
  - `Content-Type: text/csv; charset=utf-8`
  - `Content-Disposition: attachment; filename="feedback_ratings_<YYYYMMDD>.csv"`
- permission: `request.user.is_superuser` required (`403` otherwise)

- [ ] **Step 2: Stream CSV rows from export service**
- [ ] **Step 3: Add permission + CSV formatting tests (including commas/quotes/newlines in comments)**
- [ ] **Step 4: Ensure admin page route hosting the button is also superuser-only**

Run: `cd aquillm && pytest apps/platform_admin/tests/test_feedback_csv_export_api.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add aquillm/apps/platform_admin/views/api.py aquillm/apps/platform_admin/urls.py aquillm/apps/platform_admin/views/pages.py aquillm/aquillm/api_views.py aquillm/aquillm/context_processors.py aquillm/apps/platform_admin/tests/test_feedback_csv_export_api.py
git commit -m "feat(platform-admin): add superuser-only feedback ratings CSV download endpoint"
```

---

## Chunk 3: Easy Download UX + Documentation

### Task 5: Add obvious download control in admin UI (visible to superusers only)

**Files:**
- Modify: `react/src/components/WhitelistEmails.tsx`
- Test: `react/tests/usersettings.spec.ts` (or add `react/tests/feedback-export.spec.ts`)

- [ ] **Step 1: Add â€œDownload Feedback CSVâ€ button/link**
- [ ] **Step 2: Use `window.apiUrls.api_feedback_ratings_csv`**
- [ ] **Step 3: Show button only when server-provided flag indicates superuser**
- [ ] **Step 4: Keep as plain link for direct browser download (no extra JS parsing)**

Run: `cd react && npm run build`
Expected: success.

- [ ] **Step 5: Commit**

```bash
git add react/src/components/WhitelistEmails.tsx react/tests/usersettings.spec.ts
git commit -m "feat(frontend): add superuser-gated one-click feedback CSV download"
```

### Task 6: Docs + operator notes

**Files:**
- Modify: `README.md`
- Create: `docs/roadmap/plans/active/2026-03-21-feedback-rating-csv-export-execution-notes.md`

- [ ] **Step 1: Document endpoint, permissions, and filters**
- [ ] **Step 2: Include sample curl command**

```bash
curl -L -b cookies.txt "http://localhost:8000/api/feedback/ratings.csv?start_date=2026-03-01&end_date=2026-03-31" -o feedback_ratings.csv
```

- [ ] **Step 3: Commit**

```bash
git add README.md docs/roadmap/plans/active/2026-03-21-feedback-rating-csv-export-execution-notes.md
git commit -m "docs: add feedback CSV export usage and operations notes"
```

---

## Final Verification

- [ ] `cd aquillm && pytest apps/chat/tests apps/platform_admin/tests tests/integration -q --tb=short`
- [ ] `cd react && npm run build`

Expected: all pass.

---

## Notes on Format Choice

- Primary storage: **DB table/fields** (queryable, durable, auditable).
- Download format: **CSV** (explicit business requirement).
- Optional future endpoint: JSON (`/api/feedback/ratings.json`) can reuse the same export service without schema changes.

---

**Plan complete and saved to `docs/roadmap/plans/active/2026-03-21-feedback-rating-csv-export.md`.**


