# Collection Schema Editor UI Design

## Status and scope

This specification defines a frontend-only, collection-scoped schema editor integrated into the existing collection workspace. It assumes the backend already owns schema drafts, validation, publishing, version history, authorization, optimistic concurrency, and projection/rebuild orchestration.

The current pushed `development` tree does not yet expose those backend routes through `window.apiUrls`. Before implementation, the UI agent must locate the delivered backend contracts and map them in one API adapter. The UI agent must not recreate graph-domain persistence or publish logic in React or add a second source of truth.

This plan covers:

- A `Knowledge Graph` submenu/mode under every collection.
- Read-only effective-schema inspection for `VIEW` users.
- Shared collection schema drafts for `EDIT` users.
- Entity-type and relation-type editing within backend-advertised capabilities.
- Validation, diff review, publish, discard, conflict recovery, and version history.
- Thin URL exposure and TypeScript API adaptation when required.

This plan does not cover:

- A node/edge graph-canvas editor.
- Editing extracted evidence, immutable graph artifacts, or Memgraph directly.
- Cross-collection synthesis or global ontology administration.
- New backend schema-domain models, mutation rules, or publication services.
- Visual styling beyond accessible functional structure; a dedicated design agent owns polish.

## Existing architecture constraints

- The collection page mounts `CollectionView` from `react/src/features/collections/components/CollectionView.tsx` and renders its shell in `CollectionViewShell.tsx`.
- Collection detail responses already include `can_edit` and `can_manage`, but the current React collection view does not retain those fields.
- Mutating React requests use session credentials and the Django CSRF cookie.
- Client routes come from Django `reverse()` calls exposed through `window.apiUrls` by `aquillm/context_processors.py`.
- The research ontology currently models entity types with `name`, `description`, `aliases`, retrieval weight, suppression policy, and suppression threshold. Relations use `name`, `description`, `direction`, allowed head types, and allowed tail types.
- Generated graph artifacts and their child rows are immutable. Schema editing must therefore target backend draft contracts, never generated rows or the Memgraph projection.
- Existing ontology extension rules distinguish inherited definitions from collection-local additions. The server remains authoritative about which records and fields are editable.

## Navigation and permissions

The existing collection header gains a stable submenu with at least:

- `Files`: the current ingest and browse experience.
- `Knowledge Graph`: the schema workspace described here.

The selected mode should be represented in the URL query string, for example `?view=knowledge-graph`, so refresh, browser navigation, and shared links preserve the active mode without adding a second Django page.

Permission behavior:

- `VIEW`: inspect the published effective schema, search/filter definitions, inspect inherited versus collection-local provenance, view validation summaries and published history. All mutation controls are hidden or disabled with a clear read-only explanation.
- `EDIT`: create or resume the collection's one shared active draft and modify fields allowed by the server. An editor may reload a conflicted definition and manually reapply that editor's unsaved form buffer against the latest revision, but cannot force-overwrite accepted server changes. An editor cannot publish, discard/reset the shared draft, or restore versions.
- `MANAGE`: all editor behavior plus publish, revision-checked discard/reset, and restore/version rollback actions. Managers use the same optimistic concurrency rules and receive no force-overwrite path.

The backend must re-authorize every request. Client permission state controls affordances only and is never a security boundary.

## Workspace behavior

The `Knowledge Graph` mode contains four functional regions whose exact appearance is left to the design agent:

1. A schema navigation list with search, `Entity types` and `Relation types` categories, origin/status filters, and add actions.
2. A definition editor for the selected type.
3. A draft toolbar showing published version, draft revision, author/update information, dirty/request states, validation status, and permitted draft actions.
4. A collapsible validation/diff/history area.

Initial load requests one bounded effective-schema document. For `EDIT` and `MANAGE`, the response identifies the published version/checksum, active shared draft when present, permissions, definitions, validation constraints, and per-definition/per-field capabilities. For `VIEW`, the backend must omit all unpublished draft content and draft author metadata and return only the published effective schema and permitted history. The schema envelope's permission snapshot is authoritative for all knowledge-graph affordances; collection-detail permissions only decide whether to show an initial loading affordance. The editor must not infer editability from `origin` alone.

The workspace supports these states explicitly:

- Loading skeleton.
- Published schema with no draft.
- Shared draft loaded and clean.
- Draft mutation pending.
- A definition form with an unsaved local buffer based on a named draft revision.
- Draft mutation pending; accepted server state replaces the normalized draft after every mutation, while the form buffer is reset only when the accepted response contains that change.
- Validation in progress, valid, or invalid.
- Publish in progress, succeeded, or failed.
- Optimistic concurrency conflict.
- Backend unavailable or schema unavailable.
- Read-only because of permission or a draft lifecycle state.

## Entity-type functionality

Each entity-type row shows its stable name, description summary, origin (`inherited` or `collection`), and draft change state.

The editor supports the fields returned by the backend contract:

- Stable `name` token.
- Description.
- Aliases, with inherited and collection-added aliases visually distinguishable when the backend provides that provenance.
- Default retrieval weight in `[0, 1]`.
- Default suppression policy.
- Default suppression threshold in `[0, 1]`.

The UI consumes capability fields such as `editable_fields`, `removable`, and `renameable`. It must not assume inherited descriptions, stable published keys, or inherited aliases can be replaced. New collection-local definitions require every field that the backend declares required.

Add, edit, and remove actions update the shared draft through the API adapter and include the last observed draft revision. Client-side validation is driven only by backend-advertised constraints such as required fields, numeric ranges, allowed values, maximum lengths, name patterns, and collection-wide uniqueness rules. The UI must not invent a stricter stable-token grammar than the backend. Server validation remains authoritative.

## Relation-type functionality

Each relation row shows its stable name, direction, endpoint summary, origin, and draft change state.

The editor supports:

- Stable `name` token.
- Description.
- Direction (`directed` or `undirected`).
- Allowed head entity types.
- Allowed tail entity types.

Endpoint selectors use the effective entity-type set returned with the same draft revision. They must exclude types the backend marks unavailable and preserve canonical server ordering. A relation cannot be saved with empty endpoint sets, unknown entity types, or invalid direction.

As with entity types, the server-provided capability matrix determines which inherited and collection-local definitions can be changed or removed.

## Draft lifecycle and concurrency

There is at most one active shared schema draft per collection.

- An `EDIT` or `MANAGE` user selects `Create draft` when none exists.
- All editors see the same draft, base published checksum, current revision, last editor, and update timestamp.
- Every mutation includes `If-Match` or the backend's equivalent exact draft revision.
- A successful mutation returns the complete normalized draft envelope and its next revision. The client replaces its local draft with this response.
- Each definition editor owns a separate local form buffer containing `definition_key`, `base_revision`, initial server values, and current form values. Changing selection, switching collection modes, reloading, or closing a dirty form requires explicit `Keep editing` or `Discard unsaved form` resolution.
- Saving marks the form pending but keeps its buffer until the accepted normalized response arrives. Success replaces the normalized draft, selects the returned definition, updates the form's base revision, and clears its dirty state.
- A stale mutation receives `409 Conflict`. The UI preserves the local form buffer, reloads the newest normalized draft, invalidates validation, and displays backend-supplied conflicting definitions/fields. `EDIT` and `MANAGE` users may discard the local buffer or review a field-by-field rebase preview. `Reapply my changes` creates a new mutation against the latest revision; it never resends the stale revision or silently chooses the local value for a conflicting field.
- The UI never silently overwrites another editor's accepted change.

Draft mutations are not live retrieval changes. Retrieval continues using the last published generation until publish succeeds.

## Validation, diff, publish, and history

Validation is explicit and also required immediately before publish. Validation results are bound to the exact tuple `(draft_id, draft_revision, candidate_checksum)` and use stable issue codes and locations so selecting an issue focuses the relevant type and field. Results are grouped into errors and warnings; errors block publish. A validation response also includes the normalized effective-schema checksum and a summary diff against the current published schema. Any accepted mutation, draft reload, restore, discard, or conflict invalidates the previous validation result. Publish remains disabled until the currently loaded revision and checksum have a successful validation result.

The publish review shows:

- Base and candidate versions/checksums.
- Added, changed, and removed entity/relation counts.
- Validation errors and warnings.
- Any backend-reported impact, such as entities or relations affected by a schema change.

Only `MANAGE` users may confirm publish. The request includes the exact draft ID, validated draft revision, candidate checksum, validation result identity when supplied, and CSRF token. If publication is asynchronous, the UI follows the returned status URL until terminal success or failure. It does not optimistically display a candidate as published.

On success, the UI clears draft state, reloads the effective schema, shows the new published version, and reports any projection/rebuild status separately. On failure, the draft remains intact and editable unless the backend marks it terminal.

History is server-paginated. `VIEW` users can inspect versions and diffs. `MANAGE` users can request restore; restoration creates a new draft based on the selected published version and still requires validation and publish. If a shared draft already exists, restore must fail with a conflict or return an explicit replacement challenge. The UI offers only `Cancel` or a separately confirmed, revision-checked discard/replace operation naming the existing draft and its last editor; it never silently replaces shared work. History rows are never mutated or deleted by the UI.

## Frontend API boundary

All transport details live in one `collectionSchemaApi.ts` adapter. Components and hooks consume normalized TypeScript contracts rather than `window.apiUrls` directly.

The adapter must cover these backend capabilities even if delivered route names differ:

- Fetch effective schema, permissions, capabilities, and active draft.
- Create or resume the shared draft.
- Upsert/remove an entity-type draft change.
- Upsert/remove a relation-type draft change.
- Validate a specific draft revision.
- Fetch a bounded diff.
- Publish a validated revision.
- Read publish status when asynchronous.
- Discard/reset a draft.
- List published versions and fetch a bounded version diff.
- Create a restore draft from a prior version.

Every request uses `credentials: 'include'`, JSON content negotiation, bounded response parsing, typed error mapping, and CSRF protection for mutations. The adapter fails closed into a `schema_unavailable` state when required `window.apiUrls` entries are missing. It recognizes login redirects, HTML/non-JSON responses, `401`, `400`, `403`, `404`, `409`, `422`, `429`, and `5xx`, mapping them into stable client error kinds without attempting unsafe JSON parsing. It never logs full response bodies, schema payloads, credentials, or CSRF values.

## State boundaries

- `CollectionView` owns the active collection mode and retains `can_edit`/`can_manage` from the collection response.
- `CollectionKnowledgeGraphWorkspace` owns workspace-level loading and permission presentation.
- `useCollectionSchemaEditor` owns the server envelope, request state, selection, conflict recovery, validation, publication, and history pagination.
- A pure reducer applies normalized server responses and local selection changes. A separate form-buffer reducer owns unsaved values and their base revision. Async functions never partially mutate either state domain.
- Presentation components receive typed data and callbacks and do not call `fetch`.

This split keeps transport, lifecycle state, and rendering independently testable and prevents `CollectionView.tsx` or `CollectionViewShell.tsx` from becoming a second large editor implementation.

## Error handling and accessibility

- Inline field errors are associated with their inputs and summarized at the editor heading.
- Validation issues focus the corresponding field when selected.
- Conflict and publish confirmations use accessible dialogs with focus restoration.
- Every operation has a non-color status label and an `aria-live` announcement.
- Keyboard users can navigate the submenu, definition list, forms, dialogs, and history without the graph canvas or pointer gestures.
- Destructive actions name the collection and draft revision and require explicit confirmation.
- Network retries are user-triggered except for bounded publish-status polling with backoff.

## Testing strategy

Frontend unit and component tests cover:

- API request construction, CSRF, credentials, revision headers, response normalization, and typed errors.
- Missing route maps, session-expiry redirects, `401`, and non-JSON error handling.
- Reducer transitions for load, mutate, validate, conflict, publish, discard, and restore.
- Form-buffer navigation guards, accepted-save reset, stale-save preservation, and reviewed rebase behavior.
- Collection submenu routing and preservation through refresh/query-string changes.
- Permission matrices for `VIEW`, `EDIT`, and `MANAGE`.
- Entity and relation form validation and server capability enforcement.
- Conflict preservation/reapply behavior.
- Draft omission for `VIEW`, authoritative schema-envelope permissions, and draft visibility for `EDIT`/`MANAGE`.
- Validation issue navigation, revision/checksum invalidation, and publish gating.
- Asynchronous publish success/failure polling.
- Restore conflict behavior when a shared draft already exists.
- Accessibility roles, labels, focus, and keyboard navigation.

Django integration tests only verify that delivered backend route names are exposed through `window.apiUrls`. Backend schema-domain behavior remains covered by the backend implementation's own tests.

The completion gate is React typecheck, focused Vitest coverage, production React build, URL context tests, and a Playwright happy-path test against deterministic mocked schema APIs or a backend fixture.
