# Chat Context Picker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the approved collection browser and independently controllable collection skills.

**Architecture:** A shared backend catalog supplies the picker and runtime skill resolver. Per-conversation overrides preserve collection-based defaults while supporting explicit on/off choices. React edits draft context and commits it through the acknowledged WebSocket selection action.

**Tech Stack:** Django, Channels, PostgreSQL, React, TypeScript, Tailwind, Vitest, Playwright.

**Spec:** `docs/superpowers/specs/2026-09-30-chat-context-picker.md`

## Global Constraints

- Preserve existing theme variables and the approved Collections/Skills tabs, grouped right summary, Cancel/Apply flow.
- Apply is atomic and acknowledged; no silent successful save when disconnected or rejected.
- Empty skill override maps preserve existing behavior; global/operator skills remain unchanged.
- Runtime permissions and feature flags are authoritative; selecting a parent plus pack must never duplicate instructions.
- No changes to unrelated retrieval work. No deployment or main-branch merge in this task.
- Work only in `C:/Users/jackj/.codex/worktrees/chat-collections-skills/AquiLLM` on `codex/chat-collections-skills`.
- Prefix every shell command with `rtk`. No subagents from implementation workers; the coordinator owns review.

## Task 1: Skill catalog, preference persistence, and runtime

**Files:**
- Modify `aquillm/apps/chat/services/collection_prompt_skills.py` and `skills_runtime.py`.
- Create `aquillm/apps/chat/services/chat_context.py` if necessary for catalog/validation separation.
- Modify `aquillm/apps/collections/urls.py`; add a dedicated view module for `chat-context/`.
- Modify `aquillm/apps/chat/models/conversation.py`, add the next migration, update `consumers/chat.py` and `consumers/chat_receive.py`.
- Extend `aquillm/apps/chat/tests/test_collection_prompt_skills.py`; add focused catalog and preference protocol tests.

**Interfaces:** Produce the exact catalog and socket contract in the spec. Store `skill_overrides = models.JSONField(default=dict, blank=True)`. Extend `load_collection_prompt_skills(user, selected_collection_ids, skill_overrides=None)` so absent keys inherit and explicit booleans override.

- [ ] Write regression tests for a pack selected with its parent and explicit off/on selections. The production change each test guards is duplicate prompt injection or incorrect inheritance:
```python
assert prompt.count("Unique instruction body") == 1
assert "Disabled instruction body" not in prompt
assert "Explicit independent skill" in prompt
```
- [ ] Run these tests before implementing, using the available Python runtime and a disposable test database. Record the observed failures.
- [ ] Add shared permission-aware discovery. Stable IDs use document model identity and PK; catalog names use existing front-matter name/title/id resolution. Default collection IDs include the containing collection and the accessible parent when that container is a skill pack. The catalog and runtime use the same candidate rules and deterministic ordering.
- [ ] Add tests for catalog authorization and disabled feature flags, plus validation/persistence/legacy payload behavior. API tests call the real view and assert returned metadata and exclusion of inaccessible contents.
- [ ] Implement the API, migration, strict boolean map validation, atomic save, initial hydration, selection acknowledgement, and runtime overrides. Preserve stored preferences on payload omission. Validate context before mutating consumer state.
- [ ] Run targeted backend tests and migration checks; commit only Task 1 files and record commands/results in the report.

## Task 2: React picker, context transport, and user documentation

**Files:**
- Replace `react/src/features/chat/components/ChatCollectionsModal.tsx`; extract focused context components/helpers as needed.
- Modify `react/src/features/chat/components/Chat.tsx`, `ChatInputDock.tsx`, `hooks/useChatWebSocket.ts`, and `types/index.ts`.
- Add behavior tests for the modal, extend Chat and socket tests, add a small test harness/Playwright fixture for visual checks.
- Update `user-docs/docs/source/collections/using.rst` and `user-docs/docs/source/skills/markdown.rst` (and obsolete related skill-overview copy).

**Interfaces:** Consume the Task 1 catalog and acknowledged socket contract. `SkillOverrides = Record<string, boolean>`; effective skill enabled iff override key exists ? override : any `default_collection_ids` selected. Show accurate skill and collection counts, including legacy pack selections.

- [ ] Write failing UI tests for cancel discarding draft changes, explicit disabled inherited skills, parent/child selection, source search, instruction previews, and apply transport acknowledgement. Expectations must describe user-visible behavior or actual outgoing payloads.
```ts
expect(screen.getByRole('dialog')).toBeTruthy();
expect(applied.skillOverrides[skill.id]).toBe(false);
expect(applied.selectedCollections.has('parent')).toBe(true);
```
- [ ] Implement the existing approved two-column design using Lucide icons and existing theme tokens. Read the approved prototype at `C:/Users/jackj/.codex/visualizations/2026/09/30/01a0f43d-1d88-7701-9c89-dc706bffa545/collections-and-skills.html` for spacing, hierarchy, and interaction; source it as a reference, not product code. Keep whole-row checkboxes native and instructions plain text.
- [ ] Support accessible dialog focus trapping/restoration, tabs, Escape and backdrop cancel, responsive layout, loading/error/empty states, all nested collections with expandable parents, path-aware search, associated-skill shortcuts, per-source enable/disable-all, follow-collection reset, and separate right-summary groups.
- [ ] Integrate the catalog fetch, persisted override hydration and acknowledged Apply into Chat. Prevent misleading success on closed sockets, timeout, or server rejection. Persist applied preferences on refresh; send overrides with appended messages when appropriate. Existing terminal tool errors still recover normally.
- [ ] Run focused UI tests, TypeScript, and production build. Use the actual component harness to inspect desktop and 320px layouts, keyboard flow, tab switching, and pack controls. Update docs to reflect selection and scope. Commit only Task 2 files and report evidence.

## Task 3: Whole-feature verification and review

- [ ] Review the full branch for permission/runtime/picker consistency and backward compatibility.
- [ ] Run the focused backend and frontend suites, migration checks, typecheck and build after any fixes.
- [ ] Inspect rendered React at desktop and mobile sizes and compare with the approved design.
- [ ] Report the branch, validation results, required migration, and any real environment limitations; leave unrelated working-tree files untouched.
