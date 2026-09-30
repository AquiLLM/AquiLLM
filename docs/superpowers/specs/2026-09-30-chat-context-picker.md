# Collections and skills picker

Approved in chat on 2026-09-30 after interactive previews. Preserve the existing theme palette and use the approved two-column dialog: Collections/Skills tabs on the left, persistent grouped selection summary on the right, Cancel and Apply to chat in the footer. Collections is the initial tab. On narrow screens the summary stacks below the browser.

Collections show hierarchy, parent paths in search results, whole-row selection, and descendant selection. The skills badge opens associated skills. Skills show names, descriptions, source paths, plain-text instruction previews, enable/disable-all by source, individual enable/disable, and whether activation is inherited or explicitly selected. Use existing theme variables, no hardcoded light-only product palette. The composer shows separate collection and skill counts.

Selections are drafts until Apply; cancel, Escape, and backdrop dismissal discard edits. Trap keyboard focus, restore opener focus, provide accessible labels and tab keyboard navigation. Search, empty states, loading, and recoverable fetch failures must be usable. No silent success when disconnected or a save is rejected.

Per-chat skill overrides are a map of stable skill IDs to booleans. Absent means follow the existing collection-based default; true enables the skill independently, false disables it even when its collection is selected. Removing a collection removes inherited skill activation; explicitly enabled skills remain enabled. Clear all disables all available skills and clears collections. Global/operator skills are outside this collection picker and remain unchanged.

Skill discovery and runtime loading share the exact naming/eligibility rules. Only readable collections/documents may enter the catalog or prompt. IDs include document model and primary key. Selecting both a parent and its skill pack must not duplicate a skill. Permission revocation, deleted documents, and feature flags remain authoritative at runtime. Existing chats and old clients preserve automatic skill behavior with an empty override map.

## Shared API contract

`GET /api/collections/chat-context/` returns:

```ts
interface ChatContextCatalog {
  collections: Array<{id: string | number; name: string; parent: string | number | null; path: string; is_skill_pack: boolean}>;
  skills_enabled: boolean;
  skills: Array<{
    id: string; name: string; description: string; instructions: string;
    collection_id: string; collection_name: string; collection_path: string;
    source_path: string; pack_id: string | null; pack_name: string | null;
    default_collection_ids: string[];
  }>;
}
```

The catalog includes accessible descendants even when access comes from an ancestor. Pack collections remain available in Collections to preserve retrieval selection; label their special role and link their skills. Selecting only a skill does not add its collection to retrieval.

WebSocket `select_collections` accepts `collections` and optional `skill_overrides`. Save them atomically. Respond with `{context_selection: {selected_collections: [...], skill_overrides: {...}}}` after a successful save. Initial `conversation` payload includes `skill_overrides`. Reject malformed maps or unknown/inaccessible skill IDs, preserve stored overrides when omitted by legacy clients, recheck access when applying overrides and loading prompts. The append request may include the same optional field and must honor it before prompt construction.

For reliable save/retry correlation, the new client supplies an optional nonempty string `request_id` (1–128 characters) on `select_collections`; echo it inside `context_selection` when provided. Ignore acknowledgments for older requests. The picker does not apply while a reply or another save is pending. Old clients may omit this field.

For a selection failure carrying a valid request ID, respond with `{context_selection_error: {request_id: string, message: string}}`. Validation messages may describe the rejected input; unexpected failures use a sanitized retry message. Old requests without request IDs retain the existing exception format. The client must ignore stale correlated failures and display current selection errors in the dialog without treating them as chat-generation errors.

## Validation

Backend: inherited/direct pack detection, metadata, duplicate suppression, permission isolation, true/false/absent overrides, persistence/reconnect and backward compatibility, feature-disabled behavior. Frontend: draft/cancel/apply, keyboard dismissal/focus, hierarchy/search, linked skills, pack controls, source previews, counts, failure/retry, websocket acknowledgement and reload. Verify typecheck/build and desktop/mobile rendering of the actual React component.
