# Collection Schema Editor UI Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a functional, collection-scoped schema editor to the existing collection workspace, using the delivered backend draft, validation, publish, history, and authorization contracts.

**Architecture:** Keep the current collection page and add a query-string-selected `Knowledge Graph` mode beside `Files`. A single API adapter normalizes backend contracts; a server-state reducer and a separate unsaved-form reducer keep concurrency safe; small presentation components render entity/relation forms, validation, diff, and history. The backend remains the only source of truth for permissions, schema capabilities, validation, revisions, and publication.

**Tech Stack:** React 19, TypeScript 5.7, Tailwind CSS, Vitest, Testing Library, Playwright, Django URL context processors, existing CSRF and `formatUrl` utilities.

---

## Scope guardrails

- This is a frontend integration plan. Do not add backend schema-domain models, Memgraph mutations, ontology merge logic, or publish orchestration.
- Do not build a graph canvas, node/edge instance editor, or cross-collection ontology editor.
- Do not infer editability from `origin`, collection-page permission flags, or UI state. Use the schema envelope's server-provided permissions and per-field capabilities.
- Do not add a new visualization dependency. The editor is forms, lists, dialogs, and status panels.
- Keep every new TypeScript/Python source file at or below the repository's 300-line limit.
- Use backend-advertised validation constraints. Do not invent a stricter schema-name grammar in the browser.

## Backend contract preflight

This plan is intentionally blocked at Task 0 until the backend integration commit is present. The inspected local and remote-tracking `development` commit is `0db68efbc9e2b7f5f380f1033d8fa699539533fb`; it contains no collection schema draft/publish API. None of the local knowledge-graph worktree heads contains that API either. Do not start frontend implementation from this commit.

The implementation branch must first contain a backend commit that provides these capabilities:

| Capability | Required behavior |
| --- | --- |
| Read workspace | Published effective schema; authoritative permission snapshot; capabilities/constraints; active draft only for EDIT/MANAGE |
| Draft lifecycle | Create/resume one shared draft; revision-checked discard/replace |
| Definition mutations | Upsert/remove entity and relation changes with exact draft revision |
| Validation | Validate exact `(draft_id, revision)` and return candidate checksum/result identity/issues |
| Diff/publish | Bounded diff; publish exact validated revision/checksum; optional async status URL |
| History/restore | Paginated versions, bounded version diff, restore-to-new-draft with active-draft conflict |

If any required backend capability is absent, stop and report the exact missing contract. Do not fabricate API paths or move domain logic into React.

### Task 0: Pin and prove the backend contract

**Files:**

- Create: `docs/superpowers/contracts/collection-schema-api.md`
- Read/verify: the delivered backend URL, view/controller, serializer/DTO, permission, and contract-test files

- [ ] Fetch the branch the backend agent published, identify its immutable commit SHA, and verify the implementation branch contains that commit. Record both the branch and SHA at the top of `collection-schema-api.md`. Do not use an uncommitted backend worktree as the UI contract.

- [ ] Build a no-placeholder endpoint table in `collection-schema-api.md`. One row per capability must record: exact Django route name, exact `window.apiUrls` key, HTTP method, path kwargs, query/body DTO, revision/challenge headers, success DTO, error/status DTOs, response-size/pagination limit, permission, and whether it is required or conditionally optional.

- [ ] Record the exact workspace, entity, relation, validation, diff, publish, status, discard, versions, version-diff, restore, and atomic restore-replacement contracts. If publish is synchronous, record that no status route is used. If restore replacement uses a challenge token, record its field/header and expiry.

- [ ] Record successful-response size limits as well as error-response limits. The client must reject an oversized or malformed success response as `invalid_response`.

- [ ] Run the backend's named contract test modules and paste the exact commands into the contract document. The tests must prove VIEW draft omission/author-metadata omission, EDIT/MANAGE authorization, exact revision conflicts, validation-result identity, publish binding, pagination bounds, and atomic restore replacement/challenge behavior.

- [ ] Review the completed table: it must contain no `TBD`, placeholder route, guessed field, or unresolved sync/async branch. If it does, stop and return the contract to the backend agent.

- [ ] Commit: `git add docs/superpowers/contracts/collection-schema-api.md && git commit -m "docs(kg): pin collection schema API contract"`

## Chunk 1: Contract boundary and collection navigation

### Task 1: Expose and type the delivered backend routes

**Files:**

- Modify: `aquillm/aquillm/context_processors.py`
- Modify: `aquillm/tests/integration/test_context_processors_urls.py`
- Modify: `react/src/types/index.ts`
- Create: `react/src/features/collections/knowledgeGraph/schemaApiRoutes.ts`
- Create: `react/src/features/collections/knowledgeGraph/schemaApiRoutes.test.ts`

- [ ] Copy the exact route names, URL keys, kwargs, and sync/async decision from the approved `collection-schema-api.md`; do not reinterpret the backend source independently.

- [ ] Add a failing Django context-processor test asserting every exact required route key from the contract is exposed through `window.apiUrls` with its declared placeholders preserved. For publish status, test exactly the pinned mode: absent for synchronous publish; absent when asynchronous publish returns a complete opaque status URL; required when asynchronous polling needs a client-formatted status route.

- [ ] Run `python -m pytest aquillm/tests/integration/test_context_processors_urls.py -q`.

Expected: FAIL because the schema routes are not yet in `_API_URL_SPECS`.

- [ ] Introduce `_REQUIRED_SCHEMA_API_URL_SPECS` and expose it through a strict reverse helper that propagates `NoReverseMatch`. Keep `_safe_reverse` for the existing legacy map only. Add publish status to the strict set only when the pinned contract requires a formatted route; omit it for synchronous or opaque-status-URL contracts. Add a test that temporarily removes/misnames a required schema route spec and observes `NoReverseMatch` rather than silent omission.

- [ ] Add `schemaApiRoutes.test.ts` first with failing cases for the exact complete map, one missing required key, and the one pinned status mode (sync/opaque URL/formatted route). Add `// @vitest-environment jsdom` at the top.

- [ ] Run the focused Vitest file and observe the expected missing-module/export failure.

- [ ] Define a `CollectionSchemaApiRouteMap` in `schemaApiRoutes.ts` using capability names (`workspace`, `createDraft`, `entity`, `relation`, `validate`, `diff`, `publish`, `publishStatus`, `discard`, `versions`, `versionDiff`, `restore`, `restoreReplace`) and map the exact `window.apiUrls` keys once.

- [ ] Make `readCollectionSchemaApiRoutes(window.apiUrls)` return a typed unavailable result listing missing required capabilities rather than throwing or returning partial routes. Implement only enough to pass the three route-map cases.

- [ ] Run:

```powershell
cd react
npm test -- src/features/collections/knowledgeGraph/schemaApiRoutes.test.ts
cd ..
python -m pytest aquillm/tests/integration/test_context_processors_urls.py -q
```

Expected: PASS.

- [ ] Run `python scripts/check_file_lengths.py`.

Expected: PASS.

- [ ] Commit: `git add aquillm/aquillm/context_processors.py aquillm/tests/integration/test_context_processors_urls.py react/src/types/index.ts react/src/features/collections/knowledgeGraph/schemaApiRoutes.ts react/src/features/collections/knowledgeGraph/schemaApiRoutes.test.ts && git commit -m "feat(kg): expose collection schema API routes"`

### Task 2: Define normalized schema contracts and the API adapter

**Files:**

- Create: `react/src/features/collections/knowledgeGraph/schemaTypes.ts`
- Create: `react/src/features/collections/knowledgeGraph/schemaTestFixtures.ts`
- Create: `react/src/features/collections/knowledgeGraph/collectionSchemaHttp.ts`
- Create: `react/src/features/collections/knowledgeGraph/collectionSchemaHttp.test.ts`
- Create: `react/src/features/collections/knowledgeGraph/collectionSchemaApi.ts`
- Create: `react/src/features/collections/knowledgeGraph/collectionSchemaApi.test.ts`

- [ ] Define normalized types for permission level, entity/relation definitions, origin, editable fields, constraints, draft identity/revision, validation identity/issues, diff summary, publish operation, history page, and stable client error kinds.

- [ ] Keep backend DTO parsing private to the adapter. Components must never consume arbitrary response objects.

- [ ] Add test fixtures for a VIEW published envelope, an EDIT shared draft, a MANAGE draft, a validation result, and a conflict payload. Keep fixtures minimal and reusable.

- [ ] Add `// @vitest-environment jsdom` to both test files. Write failing HTTP-boundary tests for credentials, content negotiation, CSRF, bounded success/error reads, JSON/content-type validation, abort propagation, and typed status mapping. Run the HTTP test and observe the missing implementation failure.

- [ ] Implement `collectionSchemaHttp.ts` with the smallest request/parse helpers needed to pass those tests. Apply the pinned byte/depth/item bounds before accepting successful JSON as well as before classifying error text.

- [ ] Write failing operation-adapter tests that assert:

  - `credentials: 'include'` and `Accept: application/json` on every request.
  - JSON content type and the existing CSRF cookie on mutations.
  - Exact revision is sent through the backend's `If-Match` equivalent.
  - Validate sends the exact draft ID and revision.
  - Publish sends the same validated draft ID/revision, candidate checksum, and validation-result identity.
  - Atomic restore replacement sends the challenge token and exact existing-draft revision specified by the contract.
  - Collection/draft placeholders are formatted with `formatUrl`.
  - VIEW responses normalize without draft or draft-author fields.
  - Login redirects, HTML/non-JSON responses, and 401 become `session_expired`.
  - 400/403/404/409/422/429/5xx become stable client error kinds.
  - Missing routes become `schema_unavailable` before `fetch` is called.
  - Response bodies, schema payloads, CSRF values, and credentials are never logged.

- [ ] Run `cd react; npm test -- src/features/collections/knowledgeGraph/collectionSchemaHttp.test.ts src/features/collections/knowledgeGraph/collectionSchemaApi.test.ts`.

Expected: FAIL on missing operation methods/arguments while the HTTP tests pass.

- [ ] Implement `createCollectionSchemaApi(routeMap, httpClient)` covering the exact pinned operations. Keep request/response mechanics in `collectionSchemaHttp.ts` so neither source file approaches 300 lines.

- [ ] Return complete normalized server envelopes from mutation methods. Do not patch individual records into client state in the adapter.

- [ ] Run the focused HTTP, adapter, and route tests.

Expected: PASS.

- [ ] Run `python scripts/check_file_lengths.py`, then commit: `git add react/src/features/collections/knowledgeGraph && git commit -m "feat(kg): add collection schema API adapter"`

### Task 3: Add Files / Knowledge Graph collection modes

**Files:**

- Modify: `react/src/features/collections/components/collectionViewTypes.ts`
- Modify: `react/src/features/collections/components/CollectionView.tsx`
- Modify: `react/src/features/collections/components/CollectionViewShell.tsx`
- Create: `react/src/features/collections/components/CollectionModeNav.tsx`
- Create: `react/src/features/collections/components/CollectionModeNav.test.tsx`
- Create: `react/src/features/collections/components/useCollectionViewData.ts`
- Create: `react/src/features/collections/components/useCollectionViewData.test.tsx`
- Create: `react/src/features/collections/components/CollectionFilesWorkspace.tsx`
- Create: `react/src/features/collections/components/CollectionViewShell.test.tsx`

- [ ] Add `// @vitest-environment jsdom` to every `.test.tsx`. Write tests for `Files` as the default, `?view=knowledge-graph` deep links, invalid values falling back to Files, browser back/forward updates, and preserving unrelated query parameters.

- [ ] Add a test proving the Files-only ingest/browse UI is not rendered in Knowledge Graph mode.

- [ ] Run the two focused component tests and observe failures because mode navigation and conditional content do not exist.

- [ ] In `useCollectionViewData.test.tsx`, first write a failing mapping test for `data.can_edit`/`data.can_manage`, absent flags defaulting false, collection changes, and session/error cleanup.

- [ ] Extract collection fetch/mapping/permission state from the already-284-line `CollectionView.tsx` into `useCollectionViewData.ts` and make its tests pass. Expose typed `initialCanEdit`/`initialCanManage` values, documented as loading affordances only.

- [ ] Implement an accessible submenu with link/button semantics, selected state, and query-string updates through `history.pushState`. Listen for `popstate` and guard mode changes when the schema editor reports an unsaved form buffer.

- [ ] Extract the ingest separator, `FileSystemViewer`, file-operation overlays, and Files-only dialogs from the already-274-line `CollectionViewShell.tsx` into `CollectionFilesWorkspace.tsx`. This extraction is mandatory before adding the submenu. Leave collection-header controls in the shell.

- [ ] Render a temporary typed `knowledgeGraphContent` slot in Knowledge Graph mode. Flow `initialCanEdit`/`initialCanManage` through `CollectionView` and `CollectionViewShell` into that slot; Task 7 replaces it with the workspace and proves the loaded schema envelope immediately overrides these initial flags.

- [ ] Run:

```powershell
cd react
npm test -- src/features/collections/components/CollectionModeNav.test.tsx src/features/collections/components/CollectionViewShell.test.tsx src/features/collections/components/useCollectionViewData.test.tsx
npm run typecheck
```

Expected: PASS.

- [ ] Run `python scripts/check_file_lengths.py`.

Expected: PASS.

- [ ] Commit: `git add react/src/features/collections/components && git commit -m "feat(collections): add knowledge graph workspace mode"`

## Chunk 2: Concurrency-safe state

### Task 4: Implement server-state transitions

**Files:**

- Create: `react/src/features/collections/knowledgeGraph/collectionSchemaReducer.ts`
- Create: `react/src/features/collections/knowledgeGraph/collectionSchemaReducer.test.ts`

- [ ] Write a reducer state table in the test names: initial/loading, published/no-draft, draft/clean, mutation/pending, validating, valid, invalid, publishing, publish-polling, conflict, unavailable, and forbidden/read-only.

- [ ] Write failing tests proving:

  - Every accepted mutation replaces the complete normalized envelope.
  - Any accepted mutation, reload, conflict, discard, or restore clears validation identity.
  - Publish is enabled only when validation matches the loaded `(draft_id, revision, candidate_checksum)` exactly.
  - Selection survives normalization when the definition still exists and falls back predictably when removed.
  - A stale async response cannot replace a newer collection, draft revision, or request generation.
  - VIEW state cannot retain an earlier user's draft after collection/permission changes.

- [ ] Run the reducer test and observe failures for every unimplemented transition/selector before adding production code.

- [ ] Implement the transitions one family at a time (load, mutation, validation, publication/history), rerunning the named failing case after each minimal change. Add pure selectors such as `canValidate`, `canPublish`, and `selectedDefinition`. Do not call APIs or mutate nested data.

- [ ] Run `cd react; npm test -- src/features/collections/knowledgeGraph/collectionSchemaReducer.test.ts`.

Expected: PASS.

- [ ] Commit: `git add react/src/features/collections/knowledgeGraph/collectionSchemaReducer* && git commit -m "feat(kg): add schema workspace state reducer"`

### Task 5: Implement the unsaved form buffer and reviewed rebase

**Files:**

- Create: `react/src/features/collections/knowledgeGraph/schemaFormBuffer.ts`
- Create: `react/src/features/collections/knowledgeGraph/schemaFormBuffer.test.ts`

- [ ] Define the independent form state: definition key, definition kind, base revision, initial values, current values, dirty fields, pending flag, and conflict fields.

- [ ] Write failing tests for opening a definition, editing, reverting to clean, accepted-save reset, rejected-save preservation, selection/mode navigation guard, discard-local, and server reload.

- [ ] Write a three-way rebase test using initial server values, current local values, and latest server values. Non-overlapping local fields may be staged automatically; overlapping fields require an explicit user choice. The output is a new unsaved form against the latest revision, not an automatic mutation.

- [ ] Run the buffer test and observe the missing reducer/rebase failures.

- [ ] Implement open/edit/revert/save transitions first, rerun those cases, then implement the reviewed rebase helper and rerun its cases. Never offer force overwrite and never resend a stale revision.

- [ ] Run `cd react; npm test -- src/features/collections/knowledgeGraph/schemaFormBuffer.test.ts`.

Expected: PASS.

- [ ] Commit: `git add react/src/features/collections/knowledgeGraph/schemaFormBuffer* && git commit -m "feat(kg): preserve schema edits across conflicts"`

### Task 6: Orchestrate requests in one hook

**Files:**

- Create: `react/src/features/collections/knowledgeGraph/useCollectionSchemaEditor.ts`
- Create: `react/src/features/collections/knowledgeGraph/useCollectionSchemaEditor.test.tsx`
- Create: `react/src/features/collections/knowledgeGraph/schemaPublishPolling.ts`
- Create: `react/src/features/collections/knowledgeGraph/schemaPublishPolling.test.ts`
- Create: `react/src/features/collections/knowledgeGraph/useCollectionSchemaPublishing.test.tsx`
- Create: `react/src/features/collections/knowledgeGraph/useCollectionSchemaConflict.test.tsx`

- [ ] Add `// @vitest-environment jsdom` to every hook `.test.tsx`. Write the initial hook tests for load, collection ID change, unmount cancellation, route-unavailable state, session expiration, create draft, mutation acceptance, discard, and history pagination. Run them and observe missing-hook failures.

- [ ] Implement only the base hook request lifecycle, rerunning each test group as it becomes green.

- [ ] In `schemaPublishPolling.test.ts`, prove with fake timers that polling has bounded backoff, stops at a terminal response, stops on cancellation, and exposes a manual retry after exhaustion. Implement the pure/cancellable poller in its own source file and make those tests pass.

- [ ] In `useCollectionSchemaPublishing.test.tsx`, first write failing tests for exact validation request identity; stale validation responses; publish arguments matching validated draft ID/revision/checksum/result identity; synchronous success clearing and reloading the envelope; async terminal success; terminal failure preserving the draft; and projection/rebuild status remaining separate from publish status.

- [ ] Add the minimal hook publishing integration and make each named test pass.

- [ ] In `useCollectionSchemaConflict.test.tsx`, first write failing tests for 409 reload/local-buffer preservation, reviewed reapply using the newly loaded revision, stale reapply rejection without automatic retry, and atomic restore-replacement challenge/token/revision handling.

- [ ] Implement the hook using the adapter and reducers. Give requests monotonically increasing IDs or abort controllers so late responses are ignored.

- [ ] Implement conflict/restore-challenge integration: preserve the local buffer, reload the normalized latest envelope, expose backend conflict locations, and require reviewed rebase/discard-local. Restore replacement must be the pinned atomic backend request; never issue a client-side discard followed by restore.

- [ ] On session expiry, clear draft-bearing client state before presenting the sign-in action. Do not cache schema envelopes in local/session storage.

- [ ] Load history only when its panel opens and keep it server-paginated.

- [ ] Run:

```powershell
cd react
npm test -- src/features/collections/knowledgeGraph/useCollectionSchemaEditor.test.tsx src/features/collections/knowledgeGraph/useCollectionSchemaPublishing.test.tsx src/features/collections/knowledgeGraph/useCollectionSchemaConflict.test.tsx src/features/collections/knowledgeGraph/schemaPublishPolling.test.ts
cd ..
python scripts/check_file_lengths.py
```

Expected: PASS.

- [ ] Commit: `git add react/src/features/collections/knowledgeGraph/useCollectionSchemaEditor* react/src/features/collections/knowledgeGraph/useCollectionSchemaPublishing.test.tsx react/src/features/collections/knowledgeGraph/useCollectionSchemaConflict.test.tsx react/src/features/collections/knowledgeGraph/schemaPublishPolling* && git commit -m "feat(kg): orchestrate collection schema lifecycle"`

## Chunk 3: Functional schema workspace

### Task 7: Build the read-only workspace and definition navigation

**Files:**

- Create: `react/src/features/collections/knowledgeGraph/CollectionKnowledgeGraphWorkspace.tsx`
- Create: `react/src/features/collections/knowledgeGraph/CollectionKnowledgeGraphWorkspace.state.test.tsx`
- Create: `react/src/features/collections/knowledgeGraph/CollectionSchemaNavigation.tsx`
- Modify: `react/src/features/collections/components/CollectionViewShell.tsx`

- [ ] Add `// @vitest-environment jsdom` to the workspace test. Write component tests for loading, unavailable, session expired, forbidden, VIEW read-only, no draft, active draft, empty schema, search, entity/relation filters, origin/status filters, add actions, and keyboard selection.

- [ ] Test entity rows for description summary/origin/draft-change state and relation rows for direction/endpoint summary/origin/draft-change state.

- [ ] Assert unpublished draft content and author metadata are absent in the VIEW fixture and no mutation button is rendered. Pass typed `initialCanEdit`/`initialCanManage` props from the collection shell, then prove a loaded VIEW envelope immediately removes any optimistic edit affordance even if `initialCanEdit` was true.

- [ ] Assert VIEW can inspect published validation summaries and published-version history entry points returned by the server.

- [ ] Run the workspace test and observe missing-component failures.

- [ ] Build a semantic two-pane workspace: searchable definition navigation and selected-definition detail. On narrow screens, stack regions without hiding functionality.

- [ ] Show published version/checksum, origin, status text, and effective read-only values. Treat the schema envelope permission snapshot as authoritative after load.

- [ ] Wire the workspace into the `knowledgeGraphContent` slot. Pass collection ID/name, typed initial permission affordances, and the shared navigation-guard contract; do not pass the Files feature's state tree.

- [ ] Run focused component tests and typecheck.

Expected: PASS.

- [ ] Commit: `git add react/src/features/collections && git commit -m "feat(kg): add collection schema workspace"`

### Task 7A: Build conflict review and dirty-navigation protection

**Files:**

- Create: `react/src/features/collections/knowledgeGraph/SchemaConflictDialog.tsx`
- Create: `react/src/features/collections/knowledgeGraph/SchemaConflictDialog.test.tsx`
- Create: `react/src/features/collections/components/CollectionSchemaNavigationGuard.tsx`
- Create: `react/src/features/collections/components/CollectionSchemaNavigationGuard.test.tsx`
- Modify: `react/src/features/collections/components/collectionViewTypes.ts`
- Modify: `react/src/features/collections/components/CollectionView.tsx`
- Modify: `react/src/features/collections/components/CollectionViewShell.tsx`
- Modify: `react/src/features/collections/components/CollectionModeNav.tsx`
- Modify: `react/src/features/collections/components/CollectionModeNav.test.tsx`
- Modify: `react/src/features/collections/knowledgeGraph/CollectionKnowledgeGraphWorkspace.tsx`
- Modify: `react/src/features/collections/knowledgeGraph/CollectionKnowledgeGraphWorkspace.state.test.tsx`
- Modify: `react/src/features/collections/knowledgeGraph/useCollectionSchemaEditor.ts`
- Modify: `react/src/features/collections/knowledgeGraph/useCollectionSchemaConflict.test.tsx`

- [ ] Add `// @vitest-environment jsdom` to the new component/hook tests. First write failing dialog tests for non-conflicting staged fields, field-by-field latest/local choices on true conflicts, explicit resolution of every conflict, discard-local, Cancel, and `Reapply my changes` using the newly loaded revision.

- [ ] Implement the conflict dialog and wire it to the reviewed rebase output. Disable Reapply until every conflicting field has an explicit choice. Never expose force overwrite.

- [ ] Define one typed parent/child contract: `CollectionView` owns `activeMode`; `CollectionSchemaNavigationGuard` owns dirty/discard registration, pending navigation intent, the unsaved-changes dialog, `pushState`/`popstate`, and `beforeunload`; `CollectionModeNav` and the workspace request navigation through that coordinator; the workspace registers its current dirty state plus a discard-buffer callback.

- [ ] First write failing coordinator/nav/workspace tests for dirty definition selection, Files/KG mode changes, browser Back/Forward, refresh/close `beforeunload`, Cancel, Keep editing, and Discard unsaved form. A cancelled `popstate` must restore the prior URL/mode without adding an infinite history loop.

- [ ] Implement `CollectionSchemaNavigationGuard` and connect its typed context to `CollectionView`, `CollectionViewShell`, `CollectionModeNav`, workspace selection, and `beforeunload`. Use the browser's native refresh/close prompt only while dirty; use the accessible application dialog for in-app navigation.

- [ ] Add automated dialog accessibility tests: accessible name, initial focus, Tab/Shift+Tab containment, Escape behavior where safe, and trigger-focus restoration.

- [ ] Run the new focused tests, all existing conflict/navigation tests, typecheck, and `python scripts/check_file_lengths.py`.

Expected: PASS.

- [ ] Commit: `git add react/src/features/collections/knowledgeGraph react/src/features/collections/components && git commit -m "feat(kg): add reviewed schema conflict recovery"`

### Task 8: Build the entity-type editor

**Files:**

- Create: `react/src/features/collections/knowledgeGraph/EntityTypeEditor.tsx`
- Create: `react/src/features/collections/knowledgeGraph/EntityTypeEditor.test.tsx`
- Create: `react/src/features/collections/knowledgeGraph/SchemaFieldErrorSummary.tsx`
- Modify: `react/src/features/collections/knowledgeGraph/CollectionKnowledgeGraphWorkspace.tsx`
- Create: `react/src/features/collections/knowledgeGraph/CollectionKnowledgeGraphWorkspace.entity.test.tsx`
- Modify: `react/src/features/collections/knowledgeGraph/useCollectionSchemaEditor.ts`
- Modify: `react/src/features/collections/knowledgeGraph/useCollectionSchemaEditor.test.tsx`

- [ ] Add `// @vitest-environment jsdom` to the entity test. Write failing tests for name, description, aliases, retrieval weight, suppression policy, suppression threshold, inherited/local provenance, create, save, remove, cancel, and pending states.

- [ ] Drive required fields, ranges, allowed values, max lengths, patterns, uniqueness messages, `editable_fields`, `renameable`, and `removable` from normalized server constraints/capabilities.

- [ ] Prove that an inherited field omitted from `editable_fields` remains read-only even for MANAGE, and a server-enabled field is editable for EDIT.

- [ ] Implement accessible labels, descriptions, inline errors, an error-summary focus target, alias add/remove controls, and non-color status text.

- [ ] Connect Save/Remove to hook callbacks using the buffer's latest base revision. Keep the buffer until the accepted normalized envelope contains the change.

- [ ] Require a removal confirmation that names the collection, definition, shared draft, and exact draft revision; test Cancel and confirmed removal.

- [ ] Wire entity selection/add/edit/remove into `CollectionKnowledgeGraphWorkspace.tsx` and add an integrated workspace test for each action.

- [ ] Run the focused entity tests.

Expected: PASS.

- [ ] Commit: `git add react/src/features/collections/knowledgeGraph && git commit -m "feat(kg): add entity type schema editor"`

### Task 9: Build the relation-type editor

**Files:**

- Create: `react/src/features/collections/knowledgeGraph/RelationTypeEditor.tsx`
- Create: `react/src/features/collections/knowledgeGraph/RelationTypeEditor.test.tsx`
- Create: `react/src/features/collections/knowledgeGraph/SchemaEntityTypePicker.tsx`
- Modify: `react/src/features/collections/knowledgeGraph/CollectionKnowledgeGraphWorkspace.tsx`
- Create: `react/src/features/collections/knowledgeGraph/CollectionKnowledgeGraphWorkspace.relation.test.tsx`
- Modify: `react/src/features/collections/knowledgeGraph/useCollectionSchemaEditor.ts`
- Modify: `react/src/features/collections/knowledgeGraph/useCollectionSchemaEditor.test.tsx`

- [ ] Add `// @vitest-environment jsdom` to the relation test. Write failing tests for stable name, description, direction, allowed head types, allowed tail types, create, save, remove, pending, and server field capabilities.

- [ ] Test that endpoint pickers use effective entity types from the same normalized draft revision, omit server-unavailable types, preserve canonical server ordering, and reject empty selections when the server marks them required.

- [ ] Implement a keyboard-operable multi-select/listbox or checkbox group with searchable entity types and clear head/tail labels.

- [ ] Do not locally cascade-delete relations when an entity changes. Display the normalized server response and validation issues instead.

- [ ] Require a removal confirmation that names the collection, relation, shared draft, and exact draft revision; test Cancel and confirmed removal.

- [ ] Wire relation selection/add/edit/remove into `CollectionKnowledgeGraphWorkspace.tsx` and add integrated workspace tests.

- [ ] Run the focused relation tests and typecheck.

Expected: PASS.

- [ ] Run `python scripts/check_file_lengths.py`, then commit: `git add react/src/features/collections/knowledgeGraph && git commit -m "feat(kg): add relation type schema editor"`

## Chunk 4: Validation, publishing, history, and completion

### Task 10: Add draft toolbar, validation, diff, and publish review

**Files:**

- Create: `react/src/features/collections/knowledgeGraph/SchemaDraftToolbar.tsx`
- Create: `react/src/features/collections/knowledgeGraph/SchemaDraftToolbar.test.tsx`
- Create: `react/src/features/collections/knowledgeGraph/SchemaValidationPanel.tsx`
- Create: `react/src/features/collections/knowledgeGraph/SchemaValidationPanel.test.tsx`
- Create: `react/src/features/collections/knowledgeGraph/SchemaDiffDialog.tsx`
- Create: `react/src/features/collections/knowledgeGraph/SchemaDiffDialog.test.tsx`
- Create: `react/src/features/collections/knowledgeGraph/SchemaPublishDialog.tsx`
- Create: `react/src/features/collections/knowledgeGraph/SchemaPublishDialog.test.tsx`
- Modify: `react/src/features/collections/knowledgeGraph/CollectionKnowledgeGraphWorkspace.tsx`
- Create: `react/src/features/collections/knowledgeGraph/CollectionKnowledgeGraphWorkspace.lifecycle.test.tsx`
- Modify: `react/src/features/collections/knowledgeGraph/useCollectionSchemaEditor.ts`
- Modify: `react/src/features/collections/knowledgeGraph/useCollectionSchemaPublishing.test.tsx`

- [ ] Add `// @vitest-environment jsdom` to every component `.test.tsx`. In the toolbar test, first write failing tests for VIEW/EDIT/MANAGE controls as a permission matrix:

  - VIEW: inspect only.
  - EDIT: create/resume/edit/validate; no publish, discard, or restore.
  - MANAGE: EDIT actions plus publish, revision-checked discard, and restore.

- [ ] Test exact validation binding: after any mutation/reload/conflict/restore/discard, Publish disables until a new successful validation matches the loaded draft ID, revision, and candidate checksum.

- [ ] Test toolbar display of published version, draft ID/revision, last editor/update time, local dirty state, request-pending state, validation identity/state, and projection/rebuild status as a separate label.

- [ ] In `SchemaValidationPanel.test.tsx`, first test error/warning grouping by stable issue code/location and issue activation; then implement the panel. Selecting an issue chooses the definition and focuses the corresponding field.

- [ ] Add automated tests that validation issue activation moves focus to the exact field and every create/save/validate/publish/discard/poll terminal state emits an `aria-live` message.

- [ ] In `SchemaDiffDialog.test.tsx`, first test bounded base/candidate versions/checksums, entity/relation added/changed/removed counts, backend impact details, and dialog focus behavior; then implement the dialog without calculating a second diff in React.

- [ ] In `SchemaPublishDialog.test.tsx`, first test MANAGE confirmation naming the collection, draft revision, and candidate checksum, exact publish callback arguments, failure preservation, polling state, and dialog focus behavior; then implement the dialog. Do not show a candidate as published while polling.

- [ ] Add revision-checked discard confirmation that names the collection, shared draft, exact draft revision, and last editor.

- [ ] Implement and wire the toolbar, validation panel, diff dialog, and publish dialog into `CollectionKnowledgeGraphWorkspace.tsx`; make each failing permission/gating/focus test pass before moving to the next component.

- [ ] Add automated dialog tests for accessible names, initial focus, Tab containment, Escape behavior, destructive confirmation wording, and trigger-focus restoration.

- [ ] Run focused toolbar/publish tests.

Expected: PASS.

- [ ] Run `python scripts/check_file_lengths.py`, then commit: `git add react/src/features/collections/knowledgeGraph && git commit -m "feat(kg): add schema validation and publish flow"`

### Task 11: Add version history and safe restore

**Files:**

- Create: `react/src/features/collections/knowledgeGraph/SchemaHistoryPanel.tsx`
- Create: `react/src/features/collections/knowledgeGraph/SchemaHistoryPanel.test.tsx`
- Create: `react/src/features/collections/knowledgeGraph/SchemaRestoreDialog.tsx`
- Modify: `react/src/features/collections/knowledgeGraph/CollectionKnowledgeGraphWorkspace.tsx`
- Create: `react/src/features/collections/knowledgeGraph/CollectionKnowledgeGraphWorkspace.history.test.tsx`
- Modify: `react/src/features/collections/knowledgeGraph/useCollectionSchemaEditor.ts`
- Modify: `react/src/features/collections/knowledgeGraph/useCollectionSchemaConflict.test.tsx`

- [ ] Add `// @vitest-environment jsdom` to the history test. First write failing tests for paginated version loading, bounded diff inspection, VIEW-visible published validation summaries, VIEW inspection, MANAGE restore, EDIT restore absence, and terminal load errors.

- [ ] Test restore with no active draft creates a new draft requiring validation/publish.

- [ ] Test restore with an active shared draft shows the pinned backend atomic replacement challenge. Only offer Cancel or separately confirmed atomic replace-and-restore; send its challenge token plus exact existing draft ID/revision in one request and name the collection, draft, revision, and last editor.

- [ ] Test a stale challenge/revision rejection preserves the active draft, invalidates any prior validation, displays the new conflict, and never silently retries.

- [ ] Keep history immutable in the UI; do not expose delete/edit actions on published versions.

- [ ] Wire history and restore into `CollectionKnowledgeGraphWorkspace.tsx`. Add dialog name/focus containment/Escape/trigger-focus restoration tests and `aria-live` messages for load/restore outcomes.

- [ ] Run focused history tests.

Expected: PASS.

- [ ] Run `python scripts/check_file_lengths.py`, then commit: `git add react/src/features/collections/knowledgeGraph && git commit -m "feat(kg): add collection schema history"`

### Task 12: Integrated tests, accessibility, and production verification

**Files:**

- Create: `react/tests/collection-schema-editor.spec.ts`
- Create: `react/tests/fixtures/collection-schema-editor.html`
- Create: `react/src/testHarnesses/collectionSchemaEditorHarness.tsx`
- Create: `react/playwright.schema-editor.config.js`
- Create: `react/vite.schema-editor.config.ts`
- Modify as needed: focused files from Tasks 1–11 only

- [ ] Create a self-contained harness under `react/src/testHarnesses/` (inside the existing TypeScript include) that mounts production `CollectionView` with a fixed collection ID. The fixture HTML imports that harness. Do not use `auth.json`, Django, Docker, a database, or external services.

- [ ] Create `vite.schema-editor.config.ts` with `base: '/'` and no production-only Django build plugins. Create `playwright.schema-editor.config.js` with `baseURL: 'http://127.0.0.1:4173'` and `webServer: { command: 'npx vite --config vite.schema-editor.config.ts --host 127.0.0.1 --port 4173 --strictPort', url: 'http://127.0.0.1:4173/tests/fixtures/collection-schema-editor.html', reuseExistingServer: false }`. Keep the repository's existing Django E2E/Vite configuration unchanged.

- [ ] In Playwright, install `window.apiUrls`/`window.pageUrls` in `page.addInitScript`, add a non-secret `csrftoken=test-csrf-token` cookie, and stub the existing collection-detail/collection-list APIs plus every pinned schema API with `page.route`. Fixtures must exercise permission filtering and exact revision/checksum transport without containing real credentials.

- [ ] Add a deterministic happy path: open collection, select Knowledge Graph, create/resume draft, edit an entity, validate, review diff, publish as MANAGE, and observe the new published version.

- [ ] Add an E2E concurrency case: mutation returns 409, local fields remain visible, latest server revision loads, and reviewed reapply sends the newest revision.

- [ ] Add an E2E VIEW case proving no draft author/content or mutation controls are visible.

- [ ] Add automated E2E keyboard checks for submenu semantics, dialog initial focus/Tab containment/Escape/focus restoration, validation-issue field focus, and operation announcements.

- [ ] Run focused tests first:

```powershell
cd react
npm test -- src/features/collections/components/CollectionModeNav.test.tsx src/features/collections/components/CollectionViewShell.test.tsx src/features/collections/knowledgeGraph
```

Expected: PASS.

- [ ] Run all frontend and URL exposure gates:

```powershell
cd react
npx playwright install chromium
npm test
npm run typecheck
npm run build
npx playwright test --config playwright.schema-editor.config.js tests/collection-schema-editor.spec.ts
cd ..
python -m pytest aquillm/tests/integration/test_context_processors_urls.py -q
python scripts/check_file_lengths.py
python scripts/check_import_boundaries.py
git diff --check
```

Expected: every command exits 0 from a clean checkout after normal Python/Node dependency installation. Playwright starts and stops its own Vite harness; do not silently skip the E2E gate.

- [ ] Manually verify narrow viewport layout and that Files mode is unchanged. Keyboard, focus, query navigation, and announcements are automated gates above, not manual-only checks.

- [ ] Search for accidental secret/payload logging:

```powershell
rg -n "console\.(log|debug)|csrf|X-CSRFToken|response.*body" react/src/features/collections/knowledgeGraph
```

Expected: no credential/CSRF/schema-body logging; any deliberate non-sensitive log must be justified or removed.

- [ ] Review changed files for scope: no graph canvas, no direct Memgraph calls, no backend domain persistence, no global/cross-collection ontology controls, and no new package dependency.

- [ ] Commit: `git add react/tests/collection-schema-editor.spec.ts react/tests/fixtures/collection-schema-editor.html react/src/testHarnesses/collectionSchemaEditorHarness.tsx react/playwright.schema-editor.config.js react/vite.schema-editor.config.ts react/src aquillm/aquillm/context_processors.py aquillm/tests/integration/test_context_processors_urls.py && git commit -m "test(kg): verify collection schema editor workflow"`

## Handoff acceptance criteria

- Every collection has a `Knowledge Graph` submenu that deep-links via `?view=knowledge-graph`.
- The editor displays the collection's effective entity and relation schema and clearly distinguishes inherited versus collection-local definitions.
- VIEW sees published data only; EDIT can change and validate the shared draft; MANAGE can publish, discard, and restore.
- Entity and relation forms honor server-provided field capabilities and constraints.
- One shared draft is protected by exact revisions, preserves unsaved local fields on conflict, and never force-overwrites accepted changes.
- Publish is impossible unless validation matches the exact loaded draft revision and candidate checksum.
- History is paginated and immutable; restore creates a new draft and safely handles an existing shared draft.
- The browser never talks to Memgraph directly and never receives or logs service secrets.
- Focused tests, full Vitest, TypeScript typecheck, production build, URL tests, Playwright, file-length checks, and `git diff --check` pass.
