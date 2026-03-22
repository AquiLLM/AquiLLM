# Feedback rating CSV export — execution notes

Implementation follows `docs/superpowers/plans/2026-03-21-feedback-rating-csv-export.md`.

## Schema

- `Message.feedback_submitted_at` — set when a client sends `rate` or `feedback` over the chat WebSocket (validated rating 1–5; feedback text truncated to 10k chars).

## Backend

- **Export logic:** `aquillm/apps/platform_admin/services/feedback_export.py` — queryset, `question_number` via subquery count of `role='user'` messages with `sequence_number <=` the assistant row.
- **CSV view:** `aquillm/apps/platform_admin/views/api.py` → `feedback_ratings_csv`, URL name `api_feedback_ratings_csv`, mounted at `GET /api/feedback/ratings.csv` in `aquillm/aquillm/api_views.py`.
- **Frontend URL:** `aquillm/aquillm/context_processors.py` exposes `api_feedback_ratings_csv` on `window.apiUrls`.

## Permissions

- CSV endpoint: `request.user.is_superuser` (403 otherwise).
- Email whitelist **page:** staff-only (`@user_passes_test(is_staff)`), consistent with whitelist APIs. Superuser is a subset for showing the download link.

## Applying migrations

From `aquillm/`:

```bash
python manage.py migrate apps_chat
```

## Verification commands

```bash
cd aquillm
python -m pytest apps/chat/tests/test_feedback_capture.py apps/platform_admin/tests/test_feedback_export_service.py apps/platform_admin/tests/test_feedback_csv_export_api.py -q
```

```bash
cd react
npm run build
```

## Suggested git commits (same breakdown as the implementation plan)

Use these if you want history to match the plan’s tasks. Paths include a few extras that were not listed in the plan but are part of this feature.

**1 — schema**

```bash
git add aquillm/apps/chat/models/message.py aquillm/apps/chat/migrations/0002_message_feedback_submitted_at.py
git commit -m "feat(chat): add feedback submission timestamp to messages"
```

**2 — WebSocket feedback capture**

```bash
git add aquillm/apps/chat/consumers/chat.py aquillm/apps/chat/services/__init__.py aquillm/apps/chat/services/feedback.py aquillm/apps/chat/tests/test_feedback_capture.py
git commit -m "feat(chat): persist validated feedback updates with submission timestamp"
```

**3 — export query service**

```bash
git add aquillm/apps/platform_admin/services/__init__.py aquillm/apps/platform_admin/services/feedback_export.py aquillm/apps/platform_admin/tests/test_feedback_export_service.py
git commit -m "feat(platform-admin): add feedback export query service with question numbering"
```

**4 — CSV API + pytest discovery**

```bash
git add aquillm/apps/platform_admin/views/api.py aquillm/apps/platform_admin/urls.py aquillm/apps/platform_admin/views/pages.py aquillm/aquillm/api_views.py aquillm/aquillm/context_processors.py aquillm/apps/platform_admin/tests/test_feedback_csv_export_api.py pytest.ini
git commit -m "feat(platform-admin): add superuser-only feedback ratings CSV download endpoint"
```

**5 — frontend**

```bash
git add react/src/components/WhitelistEmails.tsx aquillm/templates/aquillm/email_whitelist.html react/tests/feedback-export.spec.ts
git commit -m "feat(frontend): add superuser-gated one-click feedback CSV download"
```

**6 — docs**

```bash
git add README.md docs/superpowers/plans/2026-03-21-feedback-rating-csv-export-execution-notes.md
git commit -m "docs: add feedback CSV export usage and operations notes"
```

Alternatively, squash into one commit if you prefer.

## Handoff

If work continues in a new session, search the repo for `feedback_submitted_at`, `feedback_ratings_csv`, and `api_feedback_ratings_csv` to find all touchpoints.
