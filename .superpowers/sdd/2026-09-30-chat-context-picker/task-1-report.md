# Task 1 backend report

## Result

Implemented the permission-aware chat context catalog, per-conversation skill overrides, selection acknowledgement/error correlation, and shared collection-skill discovery. The catalog and runtime use the same document eligibility and front-matter naming rules. Skill IDs are `<document model label_lower>:<primary key>`. Runtime access and feature flags are rechecked for every prompt; selecting both a parent and its skill pack injects each document once.

The route is registered in both `apps/collections/urls.py` and the mounted compatibility API in `aquillm/aquillm/api_views.py`. The conversation snapshot now includes `skill_overrides`. `select_collections` saves collection IDs and supplied overrides together and acknowledges the stored values; legacy omission retains the stored map. Optional `request_id` is echoed on success and on correlated selection errors. `append` validates and persists supplied overrides before building its prompt.

## Red/green evidence

- Before implementation, the pack/override regression failed because `load_collection_prompt_skills` accepted only two arguments; both catalog tests returned HTTP 404; selection persistence failed because `WSConversation` lacked `skill_overrides`.
- The correlated rejection regression failed on the old generic `exception` payload before implementing `context_selection_error`.
- Final: `test_collection_prompt_skills.py` — 9 passed. Focused selection and independent-skill append tests in `test_chat_consumer_append.py` — 7 passed. Tests ran against a disposable PostgreSQL/pgvector database at `127.0.0.1:56471` using system Python and `--reuse-db`.
- `python -m ruff check` passed for the new discovery service, catalog view, migration, and receive handler. `git diff --check` passed.
- `sqlmigrate apps_chat 0008` exited 0 and produced the expected `jsonb DEFAULT '{}'::jsonb NOT NULL` add-column SQL.

## Remaining validation limit

`makemigrations --check --dry-run` exits 1 because Django detects unrelated pre-existing drift in chat index/app_version/system_prompt and document model options/constraints/fields. It proposes no change to `skill_overrides`. The combined append suite was interrupted after existing append cases entered slow provider work; focused selection and append-before-prompt cases passed separately.
