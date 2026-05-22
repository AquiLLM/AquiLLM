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
   - Backing store: new model, likely `ResearchMemory`.

3. **Research Feedback Memory**
   - Scope: message, conversation, project, and optionally user.
   - Examples:
     - "Answer was rated low because it missed figures."
     - "User corrected 'DeepSeek V4' to refer to the selected paper, not public model news."
     - "Citation style should include chunk refs."
   - Backing store: new model, likely `LearningSignal`, derived from ratings, feedback text, explicit corrections, and assistant self-checks.

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

### 5. Retrieve

Before each answer, retrieve learning context in layers:

1. global researcher preferences, capped and ranked by relevance
2. project memories for selected collections/conversation
3. document memories for documents returned by RAG
4. relevant feedback/corrections that apply to the current intent

The learning context should feed the RAG prompt as a compact "Researcher/project context" block, not as unbounded memory.

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
```

Default the whole layer off until tests and manual review pass.

## Testing Strategy

### Unit Tests

- explicit global remember writes `UserMemoryFact`
- explicit project remember writes `ResearchMemory`
- selected collection scopes project memory correctly
- correction creates a `LearningSignal`
- low-confidence candidate requires confirmation
- retrieval returns only memories in allowed scope
- prompt context cap prevents memory bloat

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

## Rollout Plan

1. Add data models and admin visibility.
2. Add explicit remember/forget commands only.
3. Add project-scoped retrieval into direct RAG.
4. Add feedback/correction signals.
5. Add candidate extraction from completed conversations.
6. Add review UI or command-based review.
7. Enable conservative automatic promotion.

## Open Questions

- Should a "project" map directly to selected collection IDs, or do we need a first-class ResearchProject model?
- Should project memories be shared with collaborators who can access the collection, or remain private to the user?
- How much confirmation does the user want before storing inferred research memories?
- Should feedback from low ratings update future behavior automatically, or only create reviewable signals?
- Should project memories be exported with a collection or remain app-local?

## Recommendation

Start with explicit and correction-driven project memory, not broad automatic learning. The first milestone should make AquiLLM reliably remember what the researcher deliberately tells it, scoped to the right project, and retrieve those learnings during RAG. Once that is trustworthy, add automatic candidate extraction and feedback-driven behavior tuning.

