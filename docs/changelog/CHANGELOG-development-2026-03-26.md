# Changelog — `development` branch (since fork from `main`)

Date: **2026-03-26**  
Branch: `development`  
Compare range: `main...HEAD`  
Fork point (merge-base): `ef515d315e6254a0acf153317aa7cdf9a326da9f`

## Summary

- **Major architecture refactor**: split monolith into `aquillm/apps/*` (chat, collections, core, documents, ingestion, integrations, memory, platform_admin), with routing/ASGI/URL wiring moved into app modules and stricter import boundaries.
- **RAG performance + token efficiency**: caching for embeddings/doc lookups/images/rerank HTTP, search tuning knobs, and a salience-aware context packer integrated across providers with safer metrics and rollout guidance.
- **Chat feedback export**: superuser-gated feedback ratings CSV export endpoint + frontend one-click download, with persistence improvements and expanded tests.
- **UI overhauls**: frontend feature re-organization/splitting (chat/collections/documents/ingestion/admin), improved image rendering and feedback export UX, and assorted layout/consistency refreshes.
- **Multimodal ingestion & figure extraction**: unified ingestion APIs/services, improved sanitization (incl. null chars), and expanded figure/media handling with integration coverage.
- **Deploy/compose hardening**: secret hygiene fixes, env propagation fixes, vLLM defaults/pins, and improved LM-Lingua2 device handling.
- **Docs reorganization**: `docs/roadmap/*` and `docs/specs/*` with status buckets; refreshed architecture/ops/standards documents.
- **Ingestion throughput upgrade (concurrent uploads)**: ingestion now runs concurrently via per-file task fan-out (a major speedup vs earlier serial upload/processing), improving time-to-first-results and overall throughput.

## Notable changes (grouped)

### Added
- **Backend apps structure** under `aquillm/apps/*` with models, migrations, services, views, routing, and tests.
- **RAG cache helpers** and Django cache configuration to support retrieval hot paths.
- **Feedback CSV export** service + API endpoint and frontend download UI.
- **CI workflows** for hygiene checks and backend/frontend test runs.
- **Extensive unit/integration tests** across chat, documents, ingestion, and platform admin.

### Changed
- **Context management**: shared prompt budget behavior and improved context overflow handling; more compact tool evidence payloads.
- **Frontend organization**: move UI into `react/src/features/*` and `react/src/shared/*`; split large components to comply with file-length budgets.
- **Docs layout**: reorganized and expanded roadmap/specs/documents with clearer status tracking.

### UI / Frontend overhauls
- **Frontend feature re-org**: move UI into `react/src/features/*` (chat/collections/documents/ingestion/platform_admin) and `react/src/shared/*`, splitting large components like `ChatComponent`, `CollectionView`, `CollectionsPage`, `FileSystemViewer`, `IngestRow`, and `UserManagementModal`.
- **Styling/design refresh across pages**: broad layout/consistency updates across chat, collections, documents, ingestion, and admin screens as part of the feature split and follow-up polish passes.
- **Drag-and-drop uploads**: add a drag-and-drop file selection UI for uploads (used in the ingestion flow when adding files into the system/collections), with clearer states for “dragging”, “selected”, and “replace selection”.
- **Chat UX improvements**: more consistent markdown image rendering, fixes for image URL regressions, and improved handling around context overflow retries.
- **Feedback export UX**: add superuser-gated one-click feedback ratings CSV download, with iterative alignment/placement tweaks (header/top-nav positioning).
- **Auth screen refresh**: login screen layout/contrast improvements.

### Fixed
- **Tooling robustness**: normalize tool args, handle empty/truncated IDs safely, and gate image URLs on stored files.
- **Image rendering & overflow**: fix image markdown regressions and improve multimodal message handling.
- **Compose/env issues**: prevent empty env values from wiping vLLM args; isolate satellite vs chat env config.
- **Ingestion sanitization**: sanitize null characters during file upload paths/content handling.

### Performance
- **Retrieval path optimization**: cache hits short-circuit payload shaping; batch document rehydration; defer heavy fields; tune candidate fan-out.
- **Prompt/context efficiency**: salience-aware packing and staged pruning to reduce tokens while preserving relevant evidence.

### Documentation
- **Documentation structure overhaul**: reorganize docs into `docs/roadmap/`, `docs/specs/`, and `docs/documents/` with clearer status buckets and navigation.
- **Expanded operator + architecture documentation**: refresh and add architecture, operations, standards, and decision docs (including deployment/inference strategy, orchestration notes, and updated runbooks).

## Raw commit log

164 commits are included in this range. For the authoritative list, run:

```bash
git log --date=short --pretty=format:"%h %ad %s" main..HEAD
```

