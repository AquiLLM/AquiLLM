# Bounded validation debt plan

Date: 2026-10-09. Baseline: `15a8abaf7fbb8fdb1b850b094ebedde6c714b861`.

## Evidence and scope

`rtk npm ci --ignore-scripts --no-audit --no-fund` in `react` installed the locked dependencies without changing manifests. `rtk npm run typecheck` then reproduced exactly nine diagnostics: seven unused declarations, an obsolete `Folder` type import, and a `Set<number>` membership check against a selection API accepting `number | string`.

The source migration history shows three chat state mismatches: the state-only app transfer records a rating index name different from the index physically created by the legacy migration; the app-version migration retains its legacy backfill literal instead of the runtime callable; and the conversation state retains an empty system prompt instead of the runtime callable. The coordinator owns reproduction against the isolated Linux runtime and database.

## Exact steps

1. Record the failing frontend compiler output and read the affected interfaces and call sites.
2. Make runtime-neutral frontend corrections only in `ChatFileUpload.tsx`, `SearchPage.tsx`, `useCollectionViewMoveBatch.ts`, `FileSystemViewer.tsx`, `collectionSchemaEditorHarness.tsx`, and `uiUtils.ts`. Remove unused declarations without changing public props; use the existing `Collection` type; explicitly type the membership set consistently with the existing selection interface.
3. Run the full TypeScript check and existing collections component tests. Do not add tests that merely mirror removed declarations or type annotations.
4. Send migration findings and exact Linux reproduction commands to the coordinator before editing chat migration state. After scope agreement, add one migration after `0009_wsconversation_name_is_manual`: repair the rating index state to the historical physical name and record both callable defaults. Use state-only operations because neither the database index nor the existing row values need changing. Do not edit historical migrations or model behavior.
5. Ask the coordinator to run `makemigrations apps_chat --check --dry-run --verbosity 3`, `sqlmigrate apps_chat 0010`, and isolated migration application/reversal plus the adjacent chat persistence/feedback tests. Verify the historical index remains and legacy row values survive. Missing Linux evidence remains explicit, never replaced with mocked runtime claims.
6. Review the diff, confirm only approved files changed, commit the bounded fixes, and write concrete evidence and limitations to the task-6 integration report and shared validation-debt report.

## Constraints

No embedding, reranker, knowledge-graph, historical migration, dependency manifest, unrelated draft, server, production, or rollout changes. All shell commands start with `rtk`. No subagents or external messages.
