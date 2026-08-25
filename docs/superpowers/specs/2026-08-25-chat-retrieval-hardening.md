# Chat Retrieval Reliability and Latency Hardening Design

## Status

Proposed hardening design for the deployed `development` stack. This document
extends the existing fast academic RAG design with production evidence from the
August 25 incidents.

## Goal

Make selected-collection chat reliably evidence-first and materially faster,
without weakening academic evidence, reranking, clickable citations, final-answer
thinking, the 131,072-token context limit, or authorization boundaries.

## Incident evidence

Conversation 281 had eight selected collections. Three identical requests for
`what is attensity` did not enter direct RAG:

- Two turns persisted only `Tool:vector_search {"search_string":"attensity","top_k":10}`
  as assistant prose after using 17,473 and 17,422 combined tokens.
- A third turn exhausted the configured 12,288 completion-token budget and ended
  with `stop_reason=length`; the application could only persist its generic clean
  failure message.
- The vector search was never executed for those turns.

The selected collection state was intact. Classification failed because the
current collection-backed rule requires both selected collection IDs and a
literal document cue such as `paper`, `document`, `source`, `collection`, `this`,
or `these`. `what is attensity` has a question cue but no document cue.

Additional live traces showed:

- A WebSocket disconnect during direct synthesis caused the pipeline to repeat
  work in the slower model-driven tool loop.
- The local reranker tried an invalid batch payload that exceeded its 1,024-token
  model context, then recovered through many sequential single-document score
  requests.
- The application persists only combined message usage. Historical prompt,
  completion, and provider reasoning-token counts cannot be separated after the
  response is saved.

## Product semantics

### Selected collections are an evidence contract

When at least one collection is selected, every substantive knowledge question
uses the selected documents as its default evidence source. The user does not
need to repeat `paper`, `document`, or `collection` in each message.

Examples that must enter direct RAG when collections are selected:

- `what is attensity`
- `attensity`
- `explain calibration drift`
- `how do these methods overlap?`
- `compare the reported outcomes`

The following remain outside direct document RAG:

- greetings, thanks, acknowledgements, and conversational small talk;
- collection-picker, upload, account, and other UI-management commands;
- explicit cross-conversation memory requests;
- FITS and other app-local processing tool requests;
- retry commands, which reuse the prior turn's resolved intent and retrieval
  query instead of being classified from the word `retry` alone.

If no collection is selected, existing general-chat behavior remains available.
An explicit request to search documents with no selected collection returns the
existing transparent selection notice.

### Final answers remain academic and evidence-first

Retrieval continues to use the existing authorized hybrid candidate path:
vector, trigram, exact-term, optional ready knowledge-graph candidates, and the
configured reranker. Direct RAG performs one final synthesis over real cited
chunks. Final synthesis keeps thinking enabled; only mechanical routing disables
thinking.

## Request path

1. Persist the user turn and selected collection IDs.
2. Classify the turn with an explicit precedence order:
   local tool, chat history, retry, selected-collection knowledge question,
   explicit document request, small talk/general chat.
3. For selected-collection knowledge questions, bypass model tool selection and
   enter direct RAG.
4. Build one retrieval query for simple questions and at most three deterministic
   clause queries for genuinely multi-part questions.
5. Run authorized retrieval queries concurrently, fail-soft per query, fuse cited
   results, and build the bounded evidence packet.
6. Perform exactly one thinking-enabled synthesis call.
7. Persist the final conversation before any best-effort WebSocket notification.

The normal tool loop remains for non-document app tools and compatibility, but it
is no longer the normal path for selected-collection academic search.

## Tool-routing hardening

### Bounded mechanical generation

Initial and retry tool-selection generations must:

- send `thinking_budget=0` and OpenAI-compatible
  `chat_template_kwargs.enable_thinking=false`;
- use at most 256 completion tokens;
- never interpret an environment value of `0` as permission to use the full
  final-answer budget;
- never run answer continuation, citation repair, or a second unconstrained
  reasoning generation for a cutoff tool-selection result.

Final synthesis and post-tool evidence synthesis do not inherit this 256-token
cap and continue using the configured thinking behavior.

### Structured recovery

Provider text in the exact form `Tool:<allowed_name> {json-object}` is decoded as
a structured tool call only when `<allowed_name>` is present in that request's
authorized tool definitions and the JSON object satisfies the existing argument
validation boundary.

Raw tool syntax is never user-visible. Unknown tools, missing authorization, or
invalid arguments do not execute. Required document retrieval falls back once to
the deterministic authorized vector-search call; it never loops through repeated
model routing attempts.

### Duplicate-call control

The existing identical tool-result reuse remains authoritative. A turn may not
execute the same tool name and normalized arguments more than once. Retry and
reconnect paths reuse persisted results or restart the deterministic direct-RAG
turn, not the model-driven routing loop.

## WebSocket resilience

Transport loss is not a retrieval or model failure.

- Streaming callbacks stop sending after a recognized client disconnect but do
  not raise into direct-RAG fallback.
- Retrieval and synthesis may finish and persist after the client disconnects.
- The final delta is best-effort; reconnect loads the persisted answer.
- A transport exception must never cause the same turn to run both direct RAG
  and the tool loop.
- If a reconnect finds a persisted user turn without an assistant result, it
  resumes through the same classifier and direct-RAG path.

Each turn receives a private correlation ID so logs can join WebSocket, routing,
retrieval, reranker, synthesis, and persistence stages without logging prompts,
document text, private graph data, or environment secrets.

## Reranker latency design

The reranker remains authoritative, but known-invalid payload probing is removed
from the hot path.

- Cache the successful endpoint and payload shape by base URL and model.
- Do not retry payload shapes already known to return validation errors.
- Enforce the reranker model's token limit before sending a pair; character-only
  truncation is not considered sufficient.
- Prefer a supported independent-pair batch request.
- When the server only supports single-pair scoring, score candidates with
  bounded concurrency rather than sequential HTTP calls.
- Preserve complete, finite, candidate-bound scoring checks. Production remains
  fail-open to the deterministic fallback rank if the reranker is unavailable;
  strict evaluation remains fail-closed.
- Cache only successful complete rankings under the existing query/candidate/model
  signature.

## Token and latency observability

Per model request, structured logs record:

- private turn correlation ID and stage name;
- prompt tokens and completion tokens;
- provider-reported reasoning tokens when the API supplies them;
- whether thinking was requested;
- configured and effective completion limit;
- finish reason, time to first token, and total generation duration;
- tool name only for authorized mechanical routing, never arguments;
- retrieval, embedding, reranker, graph, evidence packing, synthesis, persistence,
  and total turn durations.

If the provider does not report reasoning tokens, the field is explicitly
`unavailable`; the application must not invent an exact count. Aggregated message
usage remains compatible with the UI's context indicator.

## Failure behavior

- No retrieval results produce a transparent cited-search notice, not a silent
  spinner.
- Partial multi-query retrieval uses successful query results and logs safe
  failure counts.
- Reranker failure uses deterministic fallback ranking and continues.
- Optional graph timeout/error preserves baseline vector evidence.
- Malformed or cutoff tool routing produces a deterministic authorized call or a
  concise error; raw tool syntax never reaches the UI.
- Client disconnect does not become a tool-loop retry.
- Every terminal path persists either a complete answer or an explicit failure
  message, so reconnect cannot leave an indefinite empty assistant bubble.

## Performance targets

Measured on the development H100 stack after warmup:

- selected-collection routing adds less than 10 ms of server CPU time;
- a simple selected-collection question performs zero LLM tool-selection calls;
- mechanical non-document tool selection uses at most 256 completion tokens;
- warm embedding plus retrieval plus reranking completes within 5 seconds at p95
  for the current development corpus and configured candidate caps;
- time to first visible final-synthesis token is within 10 seconds at p95 for a
  warm simple query;
- one user turn performs at most one final synthesis generation;
- the hot reranker path emits no known-invalid 400 request;
- WebSocket reconnect does not duplicate retrieval or generation.

Long academic answers may take longer after first token because final synthesis
retains thinking and evidence quality. The latency target is removal of routing,
probing, retry, and transport waste—not truncation of the requested answer.

## Security and privacy boundaries

- Retrieval authorization is resolved before candidate materialization and is
  unchanged by this design.
- Textual tool-call recovery may execute only tools supplied to that exact model
  request.
- Logs contain counts, timings, safe status values, and opaque correlation IDs;
  never prompts, tool arguments, document IDs, graph paths, environment values,
  credentials, or source text.
- The remote `.env` remains untracked and must never be added, committed, printed,
  or pushed.

## Deployment and rollback

The required workflow is local commit, push `origin/development`, remote pull,
then targeted deployment. Transcription is explicitly excluded.

Only affected application services are rebuilt/recreated with `--no-deps`; the
deployment must not recreate or start `vllm_transcribe`. Remote-only non-secret
token-cap values may be updated after backing up the server environment outside
the repository.

Rollback restores the previous `development` commit and remote environment
backup, then recreates only the affected application services. Retrieval and
reranker changes retain fail-open fallbacks throughout rollout.

## Acceptance criteria

- `what is attensity` with selected collections enters direct RAG and executes
  authorized retrieval without an LLM tool-selection generation.
- The same query with no collection selected remains normal general chat unless
  it explicitly asks to search documents.
- Greetings with selected collections do not trigger retrieval.
- Initial tool routing has thinking disabled and an effective maximum of 256
  completion tokens even when its environment cap is `0`.
- Authorized same-line `Tool:vector_search {...}` text becomes a structured tool
  call; unauthorized or malformed variants never execute or render.
- Tool-selection cutoff cannot consume the 12,288-token answer budget.
- A WebSocket disconnect during direct synthesis does not invoke the tool loop,
  and the completed answer is available after reconnect.
- Warm reranking uses a supported payload shape without known-invalid probes and
  without sequential per-candidate latency.
- Logs expose prompt/completion/reasoning availability and stage latency without
  sensitive content.
- Final answers retain configured thinking, real reranked evidence, clickable
  chunk citations, and the 131,072-token UI/server context contract.
- Focused Python/React tests, offline RAG evaluation, production build, and a live
  remote request pass before rollout is declared complete.
- The remote transcription service remains stopped.
