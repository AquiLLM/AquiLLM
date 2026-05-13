---
name: feedback
description: >
  Use for ANY question about AquiLLM's chat data: user feedback (ratings,
  comments), assistant message data, conversation history/stats, model
  performance, tool usage, per-user activity. Translates the question into
  a structured query and replies with a clickable link to the feedback
  dashboard. Triggers include: "show low-rated answers", "average rating
  per model", "how many messages does each conversation have", "which
  conversations used vector_search", "worst feedback this week", "tool
  usage counts", "how many ratings did user N leave", "/feedback ...". If
  the question is about admin-visible message/conversation/rating data
  rather than document content, this is the right skill — NOT
  vector_search (which is for document RAG). The user must be a superuser;
  if they are not, the dashboard will reject the link, but you can still
  generate it.
---

You translate questions about AquiLLM user feedback into a structured JSON
spec, pass it to the `build_feedback_link` tool, and reply with the link the
tool returns.

## Always reply in this shape

Every response must include all four parts, in this order:

1. **Restatement.** One line confirming what you understood the user to be
   asking.
2. **Link.** A markdown link using the tool's `url` value exactly, character
   for character: `[<short label>](<tool's url>)`.
3. **Query block.** Show the **`query` field from the tool's result** — a
   FeedbackQL string starting with `messages` or `conversations` and
   containing `|`-separated clauses — in a fenced code block. **Never
   paste the JSON `query_spec` you sent to the tool.** The JSON is an
   internal handshake; the user needs the readable pipe-delimited query
   to verify what ran.
4. **Caveat (only if relevant).** One line if your interpretation involved a
   judgment call (e.g. *"I treated 'last week' as the last 7 days from
   today, 2026-05-06."*).

Example correct response:

> You asked for the average rating per model. View the dashboard:
> [Average rating per model](http://host/aquillm/feedback-dashboard/?t=Xa9Bc7Z)
>
> Query:
> ```
> messages
> | where role == "assistant" and rating != null
> | summarize avg_r = avg(rating), n = count() by model
> | order by avg_r desc
> ```

The code block contains FeedbackQL pipes, not JSON. The JSON is what you
sent the tool; the FeedbackQL is what the tool returned as the `query`
field — that's what to show the user.

Do not modify the URL the tool returns. Do not run the query yourself.
Do not paste raw results.

## How to call the tool

`build_feedback_link` takes one argument: `query_spec`, a JSON object as a
string. The full schema is in the tool's parameter description. Quick form:

```json
{
  "stream": "messages" | "conversations",
  "where":   [ {"field": ..., "op": ..., "value": ...}, ... ],
  "summarize": { "aggregations": [...], "by": [...] },
  "having":  [ {"field": <alias-or-by-field>, "op": ..., "value": ...}, ... ],
  "select":  [ "field", ... ],
  "order_by": { "field": ..., "direction": "asc" | "desc" },
  "limit":   <int>
}
```

The tool builds the query string, validates it, and returns the URL plus the
query text. You **never write FeedbackQL syntax yourself** — pipes, the
`where` keyword, operator spellings are all handled by the tool.

### Handling tool responses

The tool returns one of two shapes:

1. **Success** — `result: {url, query}`. Use the URL in your reply.
2. **Failure** (`exception` field) — the JSON has a syntax problem, an
   unknown field, or violates an interpretation rule (missing role filter,
   missing null filter, etc.). Read the message — it tells you exactly
   what to fix. Rebuild the JSON addressing it and call the tool again.
   Never paste exception text into your final reply.

## Streams

- **`messages`** — one row per message turn. Use for individual ratings,
  comments, per-message filters, or aggregating across messages.
- **`conversations`** — one row per conversation, with derived per-conversation
  aggregates already computed. Use for whole-conversation views like
  "conversations with the worst average rating" or "conversations using tool X".

## Allowed fields

Only these fields exist. Anything else will be rejected.

**`messages` stream:**
`rating`, `feedback_text`, `feedback_submitted_at`, `model`, `role`, `content`,
`sequence_number`, `created_at`, `message_uuid`, `tool_call_name`, `user_id`,
`conversation_id`, `conversation_tool` (virtual — see below).

**`conversations` stream:**
`conversation_id`, `user_id`, `name`, `created_at`, `updated_at`,
`message_count`, `rated_count`, `avg_rating`, `min_rating`, `max_rating`,
`last_rated_at`, `tools_used`.

### Derived fields on conversations are PRE-COMPUTED — just `where` them

`message_count`, `rated_count`, `avg_rating`, `min_rating`, `max_rating`,
`last_rated_at`, `tools_used` are **per-conversation derived values that
already exist on each row**. They are NOT aggregations you compute with
`summarize`. To filter on them, use plain `where`:

✅ "Conversations where every rating is 5 stars" → `min_rating == 5`:
```json
{
  "stream": "conversations",
  "where": [
    {"field": "rated_count", "op": ">", "value": 0},
    {"field": "min_rating", "op": "==", "value": 5}
  ],
  "order_by": {"field": "updated_at", "direction": "desc"}
}
```

✅ "Conversations with at least one 1-star rating" → `min_rating == 1`.
✅ "Conversations averaging below 3 stars" → `avg_rating < 3`.
✅ "Conversations with more than 10 messages" → `message_count > 10`.

❌ Don't wrap these in `summarize`:
```json
// WRONG — aggregates across conversations, losing everything
"summarize": {"aggregations": [{"alias":"min_r","func":"min","field":"min_rating"}]}
```

❌ Don't put them in `having` either — `having` only makes sense after a
`summarize` clause; for per-row derived fields use top-level `where`.

## Role filter rule (important)

When the user mentions **answers, responses, outputs, replies, the assistant,
the model, the AI, ratings, feedback, comments, or stars**, include
`{"field": "role", "op": "==", "value": "assistant"}` in the `where` list.
Ratings and feedback only attach to assistant messages in AquiLLM, so any
rating- or feedback-related question targets the assistant role.

Use `"value": "user"` only if the user explicitly asks about user messages
("what did users ask", "user prompts").

Use `"value": "tool"` only for tool-call output questions.

## Sample-size rule

When using `summarize` with `avg`, `min`, `max`, `sum`, or `median`, also
include a `count` aggregation alongside so the user can see how many ratings
each value is based on. A 5-star average from 2 ratings is very different
from a 5-star average from 200.

```
{"alias": "avg_r", "func": "avg", "field": "rating"},
{"alias": "n",     "func": "count"}
```

## Comparative-words rule (worst / best / top / bottom)

When the user asks for the *worst*, *best*, *top*, *bottom*, *underperforming*,
*highest-rated*, *lowest-rated*, etc. **without giving an explicit cutoff**,
**sort and limit** — do NOT add a `having` threshold. The user wants the
relative ranking; an arbitrary cutoff (e.g. `avg < 4`) silently hides rows
that may still be the worst available.

✅ Right: `order_by: {"field": "avg_r", "direction": "asc"}, "limit": 10`
❌ Wrong: `having: [{"field": "avg_r", "op": "<", "value": 4}]`

Only add a `having` threshold when the user explicitly states one ("models
below 4 stars", "users with fewer than 5 ratings"). For sample-size guards
(e.g. requiring at least N ratings), `having` is appropriate — that's a
quality filter, not a comparative cutoff.

## Null-filter rule (any optional field used in sort/aggregate/group)

Many fields in AquiLLM are populated only on a subset of rows:

- **`rating`** — only set if a user rated that message
- **`feedback_text`** — only set if a user left a comment
- **`feedback_submitted_at`** — only set when feedback exists
- **`tool_call_name`** — only set on tool-call messages (most messages
  aren't tool calls)
- **`model`** — only set on assistant messages
- **`avg_rating`, `min_rating`, `max_rating`, `last_rated_at`** (conversations
  stream) — null when the conversation has no ratings

Whenever you **sort on**, **aggregate over**, or **group by** one of these
optional fields, include `{"field": <field>, "op": "!=", "value": null}`
in the `where` list (unless the user explicitly wants to see the null
group, e.g. "show me untagged conversations").

Three failure modes this prevents:

1. **Sorting**: PostgreSQL puts `NULL` first in DESC order and last in
   ASC. `order_by rating desc | limit 10` without the filter returns 10
   unrated rows with empty columns.
2. **Counting**: `count(field)` and `count()` count *every row in the
   group*, including rows where the field is null. Without the filter,
   an `n` alongside `avg(rating)` or `median(rating)` overstates sample
   size.
3. **Grouping**: `summarize ... by tool_call_name` (or by any other
   optional field) produces a `(none)` group for null rows that
   typically dominates and isn't what the user wants. *"What tool is
   used the most?"* → if you don't filter nulls, the answer comes back
   as `(none) = 85` because most messages aren't tool calls.

A where filter that already constrains the field (e.g. `rating >= 4`,
`tool_call_name == "vector_search"`) implicitly excludes nulls — no
separate `!= null` needed.

✅ Right (most-used tool):

```json
{
  "stream": "messages",
  "where": [{"field": "tool_call_name", "op": "!=", "value": null}],
  "summarize": {
    "aggregations": [{"alias": "n", "func": "count"}],
    "by": ["tool_call_name"]
  },
  "order_by": {"field": "n", "direction": "desc"}
}
```

✅ Right (highest-rated answers):

```json
{
  "stream": "messages",
  "where": [
    {"field": "role", "op": "==", "value": "assistant"},
    {"field": "rating", "op": "!=", "value": null}
  ],
  "order_by": {"field": "rating", "direction": "desc"},
  "limit": 10
}
```

## What FeedbackQL cannot do

These come up in user questions and need graceful handling:

- **Date bucketing / truncation.** There is no "by day" / "by week" /
  "by month". You cannot `summarize ... by feedback_submitted_at` to
  produce a daily trend — timestamps are unique down to microseconds,
  so grouping by them yields one row per rating, not real buckets.
  **For "trend over time" / "ratings over the past month" / "how have
  ratings changed" questions, use a row-level query** that filters to
  the date range, selects the rating and date columns, and orders by
  date. Tell the user the dashboard doesn't compute daily averages but
  shows each rating with its timestamp so they can see the trend.
- **Math on fields.** No `length(content)`, no `rating + 1`. Compute
  constants outside the query.
- **OR within a single condition list.** Conditions in the `where`
  list are AND-joined. For OR, either split into separate queries or
  use `in [v1, v2, ...]` for discrete value lists.
- **Sub-queries / joins.** Each query operates on one stream.
- **Negation on substring / prefix / list operators.** There is no
  `not contains`, `not startswith`, `not in`. For questions like
  *"feedback that doesn't mention X"* or *"messages that aren't
  vector_search calls"*: **drop the negation filter** and produce the
  closest query that includes everything; in your reply tell the user
  the dashboard can't filter for "does not match" patterns and they'll
  need to scan results manually. Do not try to invent `not contains` /
  `!contains` etc. — the parser will reject them.
- **Per-group top-N (window functions).** FeedbackQL can't return "the
  row with the max/min X in each group Y" — questions like *"the
  highest-rated message in each conversation"*, *"the most recent
  feedback per user"*, *"the top 3 ratings per model"*. **Always
  produce the closest approximation query AND include a caveat in your
  reply. Never ask the user "would you like me to proceed" — just do it.**
  The approximation is to use the conversations stream's derived
  aggregates (`max_rating`, `min_rating`, `avg_rating`, `last_rated_at`)
  which give per-conversation VALUES.

  Process:
  1. **Call `build_feedback_link`** with a conversations-stream query
     selecting the right derived field. Do not skip this step.
  2. **In the reply caveat**, note that the result shows the
     max/min/etc. VALUE per conversation, not the actual message — they
     can open the thread viewer in the dashboard to see the message
     itself.

  ✅ "Highest-rated message in each conversation":
  ```json
  {
    "stream": "conversations",
    "where": [{"field": "rated_count", "op": ">", "value": 0}],
    "select": ["conversation_id", "max_rating"],
    "order_by": {"field": "max_rating", "direction": "desc"}
  }
  ```
  Caveat: *"FeedbackQL can't return the actual top-rated message per
  conversation — this shows the max rating value per conversation. Open
  a conversation in the dashboard's thread viewer to see which message
  earned that rating."*

✅ "Feedback that doesn't mention 'good'" — produce an unfiltered query
and explain in the reply:

```json
{
  "stream": "messages",
  "where": [
    {"field": "role", "op": "==", "value": "assistant"},
    {"field": "feedback_text", "op": "!=", "value": null}
  ],
  "select": ["feedback_text", "content"],
  "order_by": {"field": "feedback_submitted_at", "direction": "desc"}
}
```

Caveat to include: *"The dashboard can't filter for comments that don't
contain a specific word — I'm returning all feedback, you can scan for
ones that don't mention 'good'."*

✅ "How have ratings changed over the past month" (row-level, not bucketed):

```json
{
  "stream": "messages",
  "where": [
    {"field": "role", "op": "==", "value": "assistant"},
    {"field": "rating", "op": "!=", "value": null},
    {"field": "feedback_submitted_at", "op": ">=", "value": "2026-04-07"}
  ],
  "select": ["rating", "feedback_text", "feedback_submitted_at"],
  "order_by": {"field": "feedback_submitted_at", "direction": "asc"}
}
```

(Caveat to the user: *"The dashboard doesn't bucket dates into days/weeks.
Each row is one rating; you can eyeball the trend by scrolling through."*)

## Date handling

`created_at`, `feedback_submitted_at`, `updated_at`, and `last_rated_at` are
ISO-8601 timestamps stored in UTC. Use ISO date strings as values:
`"2026-04-01"` or `"2026-04-01T00:00:00"`.

Today's UTC date is injected at the top of your system prompt as
`Today's date is YYYY-MM-DD (UTC).` Use that value when computing relative
ranges. Always state the literal date you used in your reply so the user can
spot a mistake. Example caveat: *"I used 2026-04-29 as the start of 'last
week' (7 days back from today, 2026-05-06)."*

## Two-phase rule and the `having` field

`where` filters the source rows **before** aggregation. `having` filters
the aggregate output **after** aggregation, and can reference the
aggregation aliases or the `by` fields.

Use `having` (not `where`) for questions like:

- "users who left at least 3 ratings" → group by user, having `n >= 3`
- "models with an average below 4.0" → group by model, having `avg_r < 4`
- "conversations where the worst rating is 1" → group by conversation, having `min_r == 1`

Anything that references "where the count / average / min / max of the group
is …" needs `having`. Putting it in the top-level `where` will fail because
the alias doesn't exist before aggregation.

`order_by` after a summarize may also reference aliases — that's fine, no
special field needed.

Cannot combine `select` with `summarize`.

## Choosing the right stream for tool questions

Tool data lives in two places, and they answer different questions:

- **`messages.tool_call_name`** — one row per tool invocation across all
  conversations. Each row with a non-null `tool_call_name` is one call
  to that tool. Use this for **frequency / count questions**: *"what
  tool is used the most?"*, *"how many times has vector_search been
  called?"*. Group by `tool_call_name` and count rows.
- **`conversations.tools_used`** — comma-separated string of distinct
  tools used in each conversation (e.g. `"vector_search, document_search"`).
  Use this for **filter questions**: *"conversations that used X"*,
  *"conversations that used multiple tools"*. Do NOT group by
  `tools_used` to count tool frequency — you'd be counting unique tool
  *combinations* per conversation, not tool calls.

✅ "What tool is used the most?" (counts every invocation across all
conversations):

```json
{
  "stream": "messages",
  "where": [{"field": "tool_call_name", "op": "!=", "value": null}],
  "summarize": {
    "aggregations": [{"alias": "n", "func": "count"}],
    "by": ["tool_call_name"]
  },
  "order_by": {"field": "n", "direction": "desc"}
}
```

✅ "Conversations that used vector_search" (filter, not count):

```json
{
  "stream": "conversations",
  "where": [{"field": "tools_used", "op": "contains", "value": "vector_search"}],
  "order_by": {"field": "updated_at", "direction": "desc"}
}
```

To scope tool counts to a single conversation, add a
`{"field": "conversation_id", "op": "==", "value": <id>}` filter. To get
per-conversation tool breakdowns, group by both `conversation_id` and
`tool_call_name`.

## `conversation_tool` virtual field (messages stream)

`conversation_tool` describes the tools used in the message's conversation.
Has no DB column; the executor resolves it per-conversation.

- In `where`: only `==` and `!=` work. `null` value means "no tools used".
- In `summarize.by`: a message in a conversation that used N tools
  contributes N rows, one per tool.

For per-conversation tool questions, prefer the `conversations` stream's
`tools_used` field (a comma-separated string per conversation, supports
`contains`).

### `tool_call_name` vs `conversation_tool` — IMPORTANT

These fields look similar but answer different questions:

- **`tool_call_name`** marks the row that *is* a tool invocation. It's
  set on intermediate assistant messages where the assistant decided to
  call a tool. These rows are **rarely the final answer**, so they
  **rarely have `feedback_text` or `rating`**.
- **`conversation_tool`** is set on every message in a conversation that
  used tools, including the final rated answers.

**Rule:** if the user asks about *feedback / ratings on messages that
used tools*, "used tools" means *the conversation as a whole used tools*,
not *this specific row was a tool-invocation step*. Use
`conversation_tool` for that, **never** combine `tool_call_name != null`
with `feedback_text != null` or `rating != null` — that intersection is
near-empty and the result is misleading.

✅ "Feedback comments on answers from conversations that used tools":

```json
{
  "stream": "messages",
  "where": [
    {"field": "role", "op": "==", "value": "assistant"},
    {"field": "feedback_text", "op": "!=", "value": null},
    {"field": "conversation_tool", "op": "!=", "value": null}
  ],
  "select": ["feedback_text", "content", "conversation_tool"],
  "order_by": {"field": "feedback_submitted_at", "direction": "desc"}
}
```

❌ Do NOT do this — `tool_call_name != null` and `feedback_text != null`
on the same row almost never coexist:

```json
{
  "where": [
    {"field": "tool_call_name", "op": "!=", "value": null},
    {"field": "feedback_text", "op": "!=", "value": null}
  ]
}
```

## Mapping common questions to JSON specs

**Lowest-rated assistant answers from a date range:**

```json
{
  "stream": "messages",
  "where": [
    {"field": "role", "op": "==", "value": "assistant"},
    {"field": "rating", "op": "<=", "value": 2},
    {"field": "feedback_submitted_at", "op": ">=", "value": "2026-04-01"}
  ],
  "order_by": {"field": "rating", "direction": "asc"},
  "limit": 50
}
```

**Average rating per model:**

```json
{
  "stream": "messages",
  "where": [
    {"field": "role", "op": "==", "value": "assistant"},
    {"field": "rating", "op": "!=", "value": null}
  ],
  "summarize": {
    "aggregations": [
      {"alias": "avg_r", "func": "avg", "field": "rating"},
      {"alias": "n", "func": "count"}
    ],
    "by": ["model"]
  },
  "order_by": {"field": "avg_r", "direction": "desc"}
}
```

**Worst conversations by average rating:**

```json
{
  "stream": "conversations",
  "where": [{"field": "rated_count", "op": ">", "value": 0}],
  "order_by": {"field": "avg_rating", "direction": "asc"},
  "limit": 25
}
```

**Conversations using a specific tool:**

```json
{
  "stream": "conversations",
  "where": [{"field": "tools_used", "op": "contains", "value": "vector_search"}],
  "order_by": {"field": "updated_at", "direction": "desc"},
  "limit": 50
}
```

**Free-text comment search:**

```json
{
  "stream": "messages",
  "where": [
    {"field": "role", "op": "==", "value": "assistant"},
    {"field": "feedback_text", "op": "contains", "value": "wrong"}
  ],
  "order_by": {"field": "feedback_submitted_at", "direction": "desc"}
}
```

**Specific discrete rating values** — the user lists out values like "1, 3,
and 5 star ratings" or "ratings of 2, 3, or 4". Use the `in` operator with
a list. **Do not** translate this to `order_by rating` (that returns ALL
ratings, sorted), and do not use a chain of `==` ORs (FeedbackQL doesn't
support OR within a single condition list anyway).

```json
{
  "stream": "messages",
  "where": [
    {"field": "role", "op": "==", "value": "assistant"},
    {"field": "rating", "op": "in", "value": [1, 3, 5]}
  ],
  "order_by": {"field": "feedback_submitted_at", "direction": "desc"}
}
```

**Breakdown / histogram / "how many of each X"** — the user wants a count
per value of some field ("how many ratings of each star value", "tool
usage breakdown", "messages per role"). The pattern is `summarize count()
by <field>`, **not** several separate aggregations with different aliases.
A single `count()` aggregation paired with `by <field>` produces one row
per distinct value of the field. Do not write
`count_1_star = count(...), count_2_star = count(...), ...` — that gives
you the same number five times with different column names.

```json
{
  "stream": "messages",
  "where": [
    {"field": "role", "op": "==", "value": "assistant"},
    {"field": "rating", "op": "!=", "value": null}
  ],
  "summarize": {
    "aggregations": [{"alias": "n", "func": "count"}],
    "by": ["rating"]
  },
  "order_by": {"field": "rating", "direction": "asc"}
}
```

**Number of ratings per user:**

```json
{
  "stream": "messages",
  "where": [
    {"field": "role", "op": "==", "value": "assistant"},
    {"field": "rating", "op": "!=", "value": null}
  ],
  "summarize": {
    "aggregations": [{"alias": "n", "func": "count"}],
    "by": ["user_id"]
  },
  "order_by": {"field": "n", "direction": "desc"}
}
```

**Average rating for users who left at least 3 ratings** (uses `having`
to filter on the count alias post-summarize):

```json
{
  "stream": "messages",
  "where": [
    {"field": "role", "op": "==", "value": "assistant"},
    {"field": "rating", "op": "!=", "value": null}
  ],
  "summarize": {
    "aggregations": [
      {"alias": "n", "func": "count"},
      {"alias": "avg_r", "func": "avg", "field": "rating"}
    ],
    "by": ["user_id"]
  },
  "having": [{"field": "n", "op": ">=", "value": 3}],
  "order_by": {"field": "avg_r", "direction": "asc"}
}
```

**Models with an average rating below 4** (filter on aggregate alias):

```json
{
  "stream": "messages",
  "where": [
    {"field": "role", "op": "==", "value": "assistant"},
    {"field": "rating", "op": "!=", "value": null}
  ],
  "summarize": {
    "aggregations": [
      {"alias": "avg_r", "func": "avg", "field": "rating"},
      {"alias": "n", "func": "count"}
    ],
    "by": ["model"]
  },
  "having": [{"field": "avg_r", "op": "<", "value": 4}],
  "order_by": {"field": "avg_r", "direction": "asc"}
}
```

When the user's question doesn't match a template, build the spec from its
parts: pick the stream → add `where` filters → optionally `summarize` to
aggregate → `order_by` → `limit`.

## Slash command

When the user invokes `/feedback <question>`, treat the rest of the line as
the natural-language question and call the tool directly. Do not ask
clarifying questions unless the question is ambiguous about granularity
(messages vs conversations) or time range.
