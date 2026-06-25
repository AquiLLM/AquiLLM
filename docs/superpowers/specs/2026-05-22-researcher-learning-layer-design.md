# Researcher Learning Layer Design

## Goal

Give AquiLLM a learning layer that improves over time with a researcher: remembering stable user preferences globally, learning project-specific research context inside selected collections/projects, and turning feedback/corrections into future retrieval and answer behavior.

This should feel like a research partner that gradually understands the user's methods, vocabulary, open questions, paper trail, preferred level of rigor, and current project direction.

## Non-Goals

- Do not fine-tune the base model in this phase.
- Do not silently rewrite source documents, citations, or extracted paper content.
- Do not inject every learned item into every prompt.
- Do not create an opaque autonomous agent that changes behavior without traceability.
- Do not replace the RAG upgrade plan; this layer should sit on top of the direct RAG pipeline.

## Design Principles

- **Separated memory scopes:** global researcher preferences and project knowledge must be stored and retrieved separately.
- **User-visible trust:** learned items should be inspectable, editable, and removable.
- **Evidence-first research memory:** project learnings should point back to conversations, documents, chunks, citations, or explicit user confirmations.
- **Small prompt footprint:** learning should improve relevance without bloating the system prompt.
- **Conservative promotion:** transient chat content should not become durable memory unless it is confirmed, repeated, high-confidence, or explicitly marked.
- **Fail-open behavior:** if learning retrieval is slow or unavailable, chat and RAG should still work.

## Existing Foundation

AquiLLM already has:

- `UserMemoryFact` in `aquillm/apps/memory/models/facts.py`, with categories for tone, goals, project, preference, and general facts.
- `EpisodicMemory` in `aquillm/apps/memory/models/episodic.py`, embedded for semantic retrieval across conversations.
- `aquillm/aquillm/memory.py`, which injects profile facts and retrieved episodic memories into the system prompt.
- Optional Mem0 integration for intelligent memory writes and retrieval.
- Message-level ratings and free-text feedback in `aquillm/apps/chat/services/feedback.py`.
- Selected collections on `WSConversation`, which can anchor project-scoped learning.

The missing piece is a deliberate learning architecture that knows the difference between "the user likes concise updates" and "this project is investigating DeepSeek V4 attention compression."

## Mem0 Upgrade Note

As part of this work, update the Mem0 integration before leaning on it as a core learning backend.

Current local state:

- `uv.lock` currently resolves `mem0ai` to `1.0.7`.
- `deploy/docker/web/Dockerfile.prod` installs `mem0ai[graph]`.
- `lib.memory.config` still exposes graph-era settings such as `MEM0_GRAPH_ENABLED`, `MEM0_GRAPH_PROVIDER`, and `MEM0_GRAPH_*`.

Relevant current Mem0 direction:

- Mem0's current memory-type docs describe layered conversation, session, user, and organizational memory, with classic factual, episodic, and semantic memory categories mapped onto those layers: https://mem0.mintlify.app/core-concepts/memory-types
- Mem0's platform docs now emphasize entity-scoped memory by user, agent, app, and session/run identifiers so memories do not leak across contexts: https://docs.mem0.ai/platform/features/entity-scoped-memory
- Mem0's memory add pipeline includes extraction, conflict resolution, metadata, and scoped identifiers such as `user_id`, `agent_id`, and `run_id`: https://docs.mem0.ai/core-concepts/memory-operations/add
- Mem0's OSS migration docs describe a newer memory algorithm, `custom_instructions`, entity linking, hybrid search/BM25 support, and the removal of OSS graph-store configuration: https://docs.mem0.ai/migration/oss-v2-to-v3
- Mem0's April 2026 token-efficient memory algorithm emphasizes production retrieval under practical token budgets, single-pass ADD-only extraction, agent-generated facts as first-class memories, entity linking, multi-signal retrieval, keyword normalization, and retrieval fusion: https://mem0.ai/blog/mem0-the-token-efficient-memory-algorithm

Implementation guidance:

- Audit and update the `mem0ai` package version and optional extras deliberately; do not just bump and hope.
- Replace graph-store assumptions with the current entity-linking or platform entity-scoped model where applicable.
- Map AquiLLM scopes onto Mem0 scopes:
  - researcher-global -> `user_id`
  - project/collection -> `run_id` or metadata such as `collection_id`
  - AquiLLM app/agent -> `app_id` or `agent_id` where supported
  - conversation/task -> `run_id`
- Keep the local `UserMemoryFact`, `EpisodicMemory`, `ResearchMemory`, and `LearningSignal` tables as the source of truth for auditability and fallback.
- Treat Mem0 as an acceleration/intelligence layer for extraction, conflict resolution, search, and entity linking, not as the only copy of researcher/project memory.
- Add compatibility tests for both current local fallback and upgraded Mem0-backed behavior.
- Document any Platform-only features separately from OSS features so local/dev deployments remain usable.

Algorithm ideas to carry into AquiLLM even when Mem0 is disabled:

- Prefer token-efficient memory retrieval over large-context replay. Target compact memory context blocks, not "stuff more chat history into the prompt."
- Use **ADD-only project memory** for research evolution. If the researcher changes their mind, store the new state alongside the old state with temporal/evidence metadata rather than overwriting the past.
- Treat assistant-generated facts as first-class when they represent accepted project decisions, completed actions, recommendations, or summaries the user continues from.
- Add entity linking for paper titles, methods, datasets, authors, figures, equations, model names, and user-defined project terms.
- Retrieve with multiple signals in parallel:
  - semantic similarity
  - keyword/BM25 or exact-term matching
  - entity matching
  - recency/temporal relevance
  - feedback/correction weight
- Normalize keywords so variants such as "attend", "attended", and "attending" can match. For research use, also normalize figure labels, equation labels, model aliases, and dataset acronyms.
- Evaluate memory quality under token budgets, not just accuracy with a huge context window.
- Track temporal transitions explicitly, for example "the project originally assumed X, later rejected X after paper Y."

## Proposed Architecture

Add a new learning layer with three memory types:

1. **Researcher Profile Memory**
   - Scope: user-global.
   - Examples:
     - "User prefers rigorous academic explanations."
     - "User wants figures included when discussing papers."
     - "User dislikes generic fallback answers."
   - Backing store: extend `UserMemoryFact` or add metadata fields to it.

2. **Project Research Memory**
   - Scope: user + project/collection.
   - Examples:
     - "Current project compares DeepSeek V4 against Qwen-style sparse attention."
     - "Open question: whether MLA-style KV compression harms long-context retrieval."
     - "User accepted the interpretation of Figure 2 as an ablation over memory usage."
   - Backing store: new model, likely `ResearchMemory` under the memory app domain.

3. **Research Feedback Memory**
   - Scope: message, conversation, project, and optionally user.
   - Examples:
     - "Answer was rated low because it missed figures."
     - "User corrected 'DeepSeek V4' to refer to the selected paper, not public model news."
     - "Citation style should include chunk refs."
   - Backing store: new model, likely `LearningSignal` under the memory app domain, derived from ratings, feedback text, explicit corrections, and assistant self-checks.

## Data Model

### `ResearchMemory`

Proposed fields:

- `id`
- `user`
- `scope_type`: `global`, `collection`, `conversation`, `document`
- `collection`: nullable FK
- `conversation`: nullable FK
- `document_id`: nullable UUID
- `memory_type`: `hypothesis`, `open_question`, `method_preference`, `paper_note`, `decision`, `correction`, `terminology`, `citation_preference`
- `content`: concise learned statement
- `evidence`: JSON with source message UUIDs, document IDs, chunk citations, or feedback IDs
- `entities`: JSON/list of linked project entities such as papers, figures, equations, models, datasets, authors, and methods
- `temporal_event_at`: nullable timestamp or logical event marker when the memory describes a transition
- `supersedes`: nullable FK/self-reference for temporal transitions without destructive overwrites
- `confidence`: float or small enum
- `status`: `active`, `superseded`, `rejected`, `archived`
- `embedding`: vector for retrieval
- `created_at`, `updated_at`
- optional `last_used_at`

### `LearningSignal`

Proposed fields:

- `id`
- `user`
- `conversation`
- `message_uuid`
- `collection`: nullable FK
- `signal_type`: `rating`, `feedback`, `correction`, `explicit_remember`, `accepted_answer`, `rejected_answer`, `manual_note`
- `polarity`: positive, neutral, negative
- `content`: normalized signal text
- `raw_payload`: JSON
- `processed_at`
- `created_at`

### Existing `UserMemoryFact`

Keep it for stable global preferences. Consider adding later:

- `source`: `explicit`, `feedback`, `inferred`, `manual`
- `confidence`
- `status`
- `last_used_at`

Those fields can be a later migration if the first pass can work with the existing table.

## Learning Pipeline

### 1. Capture

Capture possible learning inputs from:

- explicit user commands: "remember that...", "for this project...", "we decided..."
- corrections: "no, I meant...", "that's wrong because..."
- ratings and feedback text
- accepted answer patterns, when rating is high or user continues positively
- RAG sessions with selected collections
- project notes from multi-turn paper discussions

### 2. Classify Scope

Classify each learning candidate into:

- **global**: preference/style that should apply everywhere
- **project**: tied to selected collection(s), current conversation, or named research project
- **document**: tied to a specific paper/document/figure
- **ephemeral**: useful only for the current conversation, not persisted

Default should be conservative: if a candidate contains paper-specific facts, keep it project/document scoped.

### 3. Extract Candidates

Use a small prompt or deterministic heuristics to produce candidate memories:

```json
{
  "scope": "project",
  "memory_type": "open_question",
  "content": "The researcher wants to understand whether the paper's rotating black hole derivation follows from the Einstein field equations or from the Kerr metric ansatz.",
  "evidence": ["message_uuid:..."],
  "confidence": "medium",
  "requires_confirmation": false
}
```

For low-confidence or sensitive items, ask the user before storing.

### 4. Promote

Promotion rules:

- explicit remember: promote immediately
- correction: promote if it changes future answer behavior
- rating with feedback: promote as a `LearningSignal`; only turn into durable memory after summarization
- repeated pattern: promote when seen more than once
- project facts: promote only with collection/conversation/document scope
- state changes: add the new memory with temporal metadata; do not erase the previous memory unless the user explicitly asks to forget it

### 5. Retrieve

Before each answer, retrieve learning context in layers:

1. global researcher preferences, capped and ranked by relevance
2. project memories for selected collections/conversation
3. document memories for documents returned by RAG
4. relevant feedback/corrections that apply to the current intent

The learning context should feed the RAG prompt as a compact "Researcher/project context" block, not as unbounded memory.

Rank memory candidates with fused signals:

- semantic similarity against the latest request or rewritten RAG query
- exact keyword and normalized term overlap
- entity overlap with the request, selected documents, and retrieved chunks
- recency and temporal fit
- confidence and explicit-user-confirmation boost
- feedback/correction weight

### 6. Apply

Learning should influence:

- query rewriting
- retrieval mode choice
- figure inclusion policy
- citation strictness
- answer structure
- preferred depth and tone
- follow-up continuity

It should not override retrieved evidence or invent facts.

## Integration With Direct RAG

The learning layer fits naturally into the planned direct RAG architecture:

1. `rag_intent` classifies the user request.
2. `learning_context` retrieves relevant global/project memories.
3. `rag_query` uses learning context to rewrite follow-up queries.
4. `rag_pipeline` retrieves documents/chunks/figures.
5. `rag_evidence` packages evidence.
6. `rag_synthesis` receives:
   - user request
   - learning context
   - evidence packet
   - citation allowlist
7. `learning_capture` observes the final turn and queues candidate memories.

This avoids a common failure mode: injecting a large generic memory block before retrieval. Learning should guide retrieval and synthesis, but RAG evidence remains the source of truth.

## User Experience

### Minimal First Version

No heavy UI is required for the first implementation. Use:

- explicit phrases:
  - "remember globally..."
  - "remember for this project..."
  - "forget that..."
  - "what do you remember about this project?"
- an assistant confirmation when a meaningful project memory is stored:
  - "Noted for this project: ..."

### Later UI

Add a lightweight "Learning" panel:

- Global preferences
- Project memories
- Open questions
- Corrections
- Recently learned
- Rejected/archived memories

Each memory should support:

- edit
- delete
- archive
- move scope
- source/evidence preview

## Privacy And Safety

- Never store secrets, API keys, credentials, or sensitive personal data from incidental conversation.
- Do not promote medical/legal/financial conclusions as durable facts without explicit user confirmation.
- Mark inferred memories as inferred.
- Allow users to disable learning globally or per project.
- Allow "forget this conversation" and "forget project memory."
- Log memory metadata and IDs, not full private content, in operational logs.

## Configuration

Suggested env flags:

```env
LEARNING_LAYER_ENABLED=0
LEARNING_PROJECT_MEMORY_ENABLED=1
LEARNING_FEEDBACK_SIGNALS_ENABLED=1
LEARNING_AUTO_PROMOTE_CORRECTIONS=1
LEARNING_AUTO_PROMOTE_LOW_CONFIDENCE=0
LEARNING_MAX_CONTEXT_FACTS=6
LEARNING_MAX_CONTEXT_CHARS=1200
LEARNING_CAPTURE_IDLE_SECONDS=60
LEARNING_MEM0_INTEGRATION_MODE=disabled
LEARNING_MEM0_PROJECT_SCOPE_FIELD=run_id
```

Default the whole layer off until tests and manual review pass.

## Standards Alignment

This design must follow `docs/documents/standards/code-style-guide.md`.

- Keep Django-bound models, services, admin, tasks, and permissions in `aquillm/apps/memory/**`, for example:
  - `aquillm/apps/memory/models/research.py`
  - `aquillm/apps/memory/models/signals.py`
  - `aquillm/apps/memory/services/learning_capture.py`
  - `aquillm/apps/memory/services/learning_context.py`
  - `aquillm/apps/memory/tasks.py`
- Put only Django-free extraction, normalization, scoring, or entity-linking helpers in `aquillm/lib/memory/**`; `aquillm/lib/**` must not import `apps.*`.
- Do not add new runtime imports from the compatibility barrel `aquillm.models`.
- Keep chat and RAG consumers thin: they should call memory services, not perform learning writes or retrieval inline.
- Keep every new Python file under the 300-line limit by splitting capture, retrieval, promotion, Mem0 adapter logic, and prompt formatting into separate modules.
- Add module docstrings and type hints on public service boundaries.
- Enforce user, conversation, collection, and document authorization before reading or writing memories.
- Keep Celery task arguments and memory evidence payloads JSON-safe; pass IDs and compact metadata, not ORM objects or large source text.
- Avoid threaded ORM writes. Background promotion should use Celery tasks or explicit service calls.
- Log memory IDs, counts, scopes, statuses, and timings; do not log secrets, credentials, raw private conversations, full source documents, or unredacted feedback bodies.
- Keep the learning layer feature-flagged and default-off until tests and manual review pass.

## Testing Strategy

### Unit Tests

- explicit global remember writes `UserMemoryFact`
- explicit project remember writes `ResearchMemory`
- selected collection scopes project memory correctly
- correction creates a `LearningSignal`
- low-confidence candidate requires confirmation
- retrieval returns only memories in allowed scope
- prompt context cap prevents memory bloat
- ADD-only state changes preserve previous and current project facts
- entity-linked memories are boosted when a query mentions the same paper, figure, equation, model, dataset, or method
- keyword normalization matches common research variants and aliases

### Integration Tests

- Paper discussion followed by "remember for this project..." affects a later follow-up in the same collection.
- A global style preference affects general chat but does not add project facts.
- A project-specific correction does not leak into another collection.
- Negative feedback about missing figures increases figure inclusion behavior in similar project RAG prompts.
- Deleting/archiving a memory prevents future injection.

### Evaluation

Add a small learning eval set:

- remembers user style
- remembers project open question
- respects correction
- avoids cross-project leakage
- avoids storing sensitive text
- improves answer quality without increasing prompt size beyond budget
- answers temporal questions about how a project decision changed over time
- recalls assistant-generated project decisions only after user acceptance or continuation

### Standards Gates

Run the repo standards gates before enabling the feature:

```bash
python scripts/check_file_lengths.py
python scripts/check_import_boundaries.py
pwsh -ExecutionPolicy Bypass -File scripts/check_hygiene.ps1
pytest aquillm/apps/memory/tests aquillm/apps/chat/tests/test_feedback_capture.py aquillm/lib/memory/tests
git diff --check
```

## Rollout Plan

1. Add data models and admin visibility.
2. Audit and upgrade Mem0 integration behind a feature flag.
3. Add explicit remember/forget commands only.
4. Add project-scoped retrieval into direct RAG.
5. Add feedback/correction signals.
6. Add candidate extraction from completed conversations.
7. Add review UI or command-based review.
8. Enable conservative automatic promotion.

## Open Questions

- Should a "project" map directly to selected collection IDs, or do we need a first-class ResearchProject model?
- Should project memories be shared with collaborators who can access the collection, or remain private to the user?
- How much confirmation does the user want before storing inferred research memories?
- Should feedback from low ratings update future behavior automatically, or only create reviewable signals?
- Should project memories be exported with a collection or remain app-local?
- Which Mem0 features should be required for self-hosted/local development versus optional Platform enhancements?

## Recommendation

Start with explicit and correction-driven project memory, not broad automatic learning. The first milestone should make AquiLLM reliably remember what the researcher deliberately tells it, scoped to the right project, and retrieve those learnings during RAG. Once that is trustworthy, add automatic candidate extraction and feedback-driven behavior tuning.
