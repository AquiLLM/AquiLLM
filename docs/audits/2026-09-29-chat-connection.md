# Chat initial connection and reconnect audit

Date: 2026-09-29. Reviewed checkout: `75efe98c`.

Scope: the red status banner, React WebSocket lifecycle, Django/Channels startup,
conversation hydration, and recovery when a chat page loads or reconnects.
This records the original audit before application changes. Subsequent repairs
and validation are tracked in [Chat connection remediation](2026-09-29-chat-connection-fixes.md).

## Assessment

The red banner does not necessarily indicate a backend outage. Every first
connection writes `Attempting to connect... (Attempt 1 of 5)` into the exception
state, and every exception is rendered with the red error styling. There are also
real lifecycle and backend recovery defects, including a stale-save data-loss risk.

## Findings

### 1. [P1] A successful reconnect closes its own socket

Location: `react/src/features/chat/hooks/useChatWebSocket.ts:94-100,172-206`.

After an initial failure increments `connectionAttempts`, a successful socket's
`onopen` resets that state to zero. Because the state is an effect dependency,
React cleans up the effect, closes that successful socket, and creates another.
The closed socket's handlers remain active and its delayed retry is not cancelled,
so another attempt can interrupt the replacement socket again.

There is a second entry into the same timer problem: the five-second timeout
increments the counter immediately, and closing that socket schedules another
increment two seconds later through `onclose`.

**Evidence:** lifecycle probes show the recovered socket's `close()` called once
immediately after `open`, and three socket instances by approximately seven seconds
after one handshake timeout, although the second socket has not timed out yet.

**Recommended change:** separate the retry counter from effect ownership; retain
and cancel every timer; ignore callbacks from retired sockets; schedule exactly
one retry for each failure. Cleanup should close only its own socket and should
not initiate another retry.

### 2. [P1] Loading a chat can overwrite messages saved by another connection

Locations: `aquillm/apps/chat/consumers/chat.py:184-255`,
`aquillm/lib/llm/providers/base.py:171-183`,
`aquillm/apps/chat/consumers/chat_delta.py:31-38`, and
`aquillm/aquillm/message_adapters.py:145-214`.

Initial connection loads a transcript, then performs memory initialization before
saving it. Even when there is no pending turn, the normal spin calls its publish
callback before checking `changed == "unchanged"`; that callback saves before
checking whether a delta exists. Connect also saves again afterward.

The saver deletes database messages whose UUIDs are absent from its in-memory
snapshot. If tab B loads an old transcript, tab A persists a new turn, and tab B
then finishes initialization, tab B's save deletes tab A's newer messages. A
transaction around the save does not detect that the snapshot was stale. The
turn-local publication fence likewise does not compare conversation revisions.

**Evidence:** an isolated probe of the actual delta callback and saver reduced
the simulated database from three messages to the two-message snapshot, deleted
the newer user message, and emitted no WebSocket delta. The ORM and transaction
boundary were stubbed; this was not a concurrent live-database test.

**Recommended change:** avoid saving unchanged snapshots on initial load, and
protect transcript writes against concurrent stale writers with a revision check
or another explicit concurrency protocol. Skipping just the final connect save
does not remove the save inside the spin callback.

### 3. [P2] Reloaded pending tool calls cannot resume

Locations: `aquillm/aquillm/message_adapters.py:101-111`,
`aquillm/lib/llm/types/conversation.py:30-41`, and
`aquillm/lib/llm/providers/complete_turn.py:474-480`.

If a page reloads after an assistant tool call was persisted but before its tool
result was persisted, loading restores the tool-call metadata but leaves
`message.tools` unset. `rebind_tools()` only updates tool lists that already exist.
The completion path requires both a tool list and a tool-call ID; otherwise it
returns `unchanged`. The frontend sees the pending tool call, disables input, and
continues showing a spinner without receiving a result or terminal error.

**Evidence:** an isolated probe using the repository's message types, loader
conversion, tool rebinding, and completion function leaves the conversation at two
messages with `tools=None`, zero tool executions, and an `unchanged` outcome.

**Recommended change:** restore an authorized runtime tool binding for pending
calls when loading them, or explicitly terminate unrecoverable calls with a
recoverable UI state.

### 4. [P2] Fatal startup errors leave an open but unusable chat

Locations: `aquillm/apps/chat/consumers/chat.py:175-180,258-267`,
`aquillm/apps/chat/consumers/chat_receive.py:213`, and
`react/src/features/chat/hooks/useChatWebSocket.ts:107-113`.

An invalid conversation or provider-overload error sets `dead=True` and sends an
exception without closing the accepted socket. The receive handler silently
ignores subsequent messages on a dead consumer. The frontend enables input on
every exception, so users can send another message, clear the error, and wait
indefinitely for a reply. No transport close occurs to trigger reconnection.

**Evidence:** an isolated overload probe records an accepted, connected, dead
consumer, no close call, and no response to a subsequent append.

**Recommended change:** distinguish fatal initialization errors from recoverable
turn errors. Close fatal connections with an appropriate terminal/retryable code,
and keep the composer disabled unless the application can accept a turn. Invalid
or unauthorized conversation IDs should not enter an endless retry loop.

### 5. [P2] The composer becomes available before chat state is loaded

Locations: `react/src/features/chat/hooks/useChatWebSocket.ts:94-100,118-133`,
`react/src/features/chat/components/Chat.tsx:147-167`, and
`aquillm/apps/chat/consumers/chat.py:155-204`.

The server accepts the WebSocket before loading the conversation and the browser
enables input on that transport event. At that moment it has neither transcript
nor saved collection selection. A quick message can be sent with the initial empty
collection list; the backend append path persists that list. The later initial
snapshot also replaces any optimistic messages shown before hydration. The
five-second timeout ends at socket open, so it cannot diagnose stalled hydration.

**Evidence:** the frontend probe observes enabled input immediately after `open`
without receiving any conversation payload.

**Recommended change:** keep transport-connected and application-ready states
separate. Enable sending only after authoritative conversation state and saved
selection are loaded; give initialization its own failure/timeout handling.

### 6. [P2] Healthy first connection is displayed as a red error

Locations: `react/src/features/chat/hooks/useChatWebSocket.ts:67` and
`react/src/features/chat/components/Chat.tsx:257-263`.

The normal first attempt writes a status string into the same state used for
errors. The component styles every such string with `bg-red-dark`. This makes a
healthy page load appear broken, even before any failure has occurred.

**Evidence:** rendering the actual Chat component before socket open produces
`Attempting to connect... (Attempt 1 of 5)` inside the red banner.

**Recommended change:** represent initial loading, reconnecting, terminal failure,
and turn errors separately. Use a neutral loading presentation for normal startup.

## Other startup observations

- `chat.py:139-145` loads each permission's related collection just to read its ID,
  producing an avoidable N+1 query pattern. The resulting selection is then
  replaced by the saved selection at line 181. Eliminate that redundant scan or
  fetch IDs directly if still required. This cost happens before the initial
  conversation payload.
- Memory augmentation runs on every connect, including completed conversations.
  Without selected collections, an existing chat can perform episodic retrieval
  for its previous user message before processing new input. The answer-generation
  path itself short-circuits for empty and completed chats. An unnamed chat with
  at least two messages can separately invoke title generation during saving.
- Nginx forwards WebSocket upgrade headers and configures a 3,600-second proxy
  read timeout. The observed five-second timeout is in the frontend hook; this
  audit did not observe a live proxy timeout.

## Verification and limits

The existing focused frontend suite passed: **4 tests in 2 files**:

```text
rtk proxy npm test -- src/features/chat/hooks/useChatWebSocket.test.tsx src/features/chat/components/Chat.test.tsx
```

Four temporary intended-behavior probes all failed, reproducing the normal red
banner, successful-reconnect closure, duplicate timeout retry, and premature input
enablement. Those probe files were removed after the audit. Existing tests use a
socket fake whose `close()` never emits a close event, and their reconnect
scenario calls `onopen` again on the same fake instance, so they do not cover the
actual retry lifecycle.

Backend validation uses isolated repository functions with stubbed external
dependencies. It does not establish deployed incident frequency or live database
timings. No production service was changed, and no authenticated browser or full
Django/PostgreSQL integration run was performed.

The backend probes reported:

```text
pending_tool_reconnect: tools=null, outcome=unchanged, messages=2, tool_calls=0
overloaded_connect: accepted=1, dead=true, connected=true, closes=0, retry_responses=0
unchanged_snapshot_save: database_before=3, snapshot=2, database_after=2, newer_message_survives=false, deltas=0
```
