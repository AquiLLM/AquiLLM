# Offline Evidence Annotation Rubric 1.0

**Written:** 2026-08-06, before fixture labels or production measurements
**Scope:** synthetic-public routing, evidence-packing, and deterministic memory-helper fixtures

## Non-negotiable protocol

Annotators assign gold labels from the text and metadata in these fixtures, without
running the cases through AquiLLM. The frozen cases must not be changed, removed, or
relabeled in response to benchmark outputs. A discovered annotation defect requires a
documented adjudication, a new dataset version, new hashes, and a new canonical run.
Cases that reveal current limitations remain in the dataset and report as negative
cases; they are never hidden or converted into favorable examples after execution.

All content is invented for this evaluation. It contains no user prompts, private
documents, credentials, host details, or production identifiers. `synthetic_public`
is both the provenance and maximum permitted sensitivity.

## Strata

- `favorable`: direct, unambiguous wording representative of the ontology.
- `unfavorable`: valid intent or evidence configuration likely to expose a known
  deterministic limitation; the semantic gold label is retained.
- `ambiguous`: more than one ordinary-language reading is plausible. Annotators use
  the explicit tie-break rules below and record the competing reading in the rationale.
- `adversarial_boundary`: negation, quoted commands, misleading keywords, tight
  budgets, duplicate identities, or prompt-like text probes a decision boundary.

Every dataset balances these strata. Stratum is descriptive metadata, never an input
to the system under evaluation.

## Routing ontology

Classifier gold contains five independent booleans:

- `requires_rag`: the user asks AquiLLM to obtain or ground information in documents.
- `wants_figures`: the requested answer requires a document figure, image, plot,
  chart, graph, visual, or diagram.
- `wants_whole_document`: the user explicitly requests the complete document rather
  than passages. Version 1.0 retains this field even when all cases label it false.
- `is_retry`: the whole utterance asks to repeat the immediately preceding operation.
- `requires_local_tools`: the primary request is local file/FITS processing or an
  explicit local tool operation.

`reason` is a separate single label with this priority: `retry_request`,
`local_tool_request`, `figure_request`, `explicit_search`,
`collection_backed_question`, then `no_retrieval_needed`. Quoted or negated commands
describe rather than request an action and therefore do not activate that action.

`production_action` is annotated independently from classifier fields:

- `retrieve`: retrieval is needed and at least one collection is selected.
- `prompt_select_collection`: retrieval is needed but no collection is selected.
- `local_tool_handling`: a local tool is the primary requested action.
- `skip_normal_tool_loop`: none of the preceding actions applies.

When retrieval-query behavior is part of the case, `expected_query` is exact after
leading/trailing whitespace removal only. A retry reuses the most recent prior vector
query when present. A coreferential follow-up prepends the most recently retrieved
document title as `<title>: <trimmed user text>`. Otherwise it is the trimmed text.

Ambiguity tie-break: prefer the literal requested operation over a possible implied
one; prefer selected-collection grounding only when the utterance actually refers to
documents or their contents; and prefer local-tool handling over retrieval when data
processing is the primary verb. Rationales name any plausible alternative.

## Evidence ontology

Each candidate has a stable `evidence_id`; list position and rank are not identities.
`relevant: true` means the chunk directly supplies a fact necessary to answer the
synthetic case question. Contextually related, redundant-but-nonessential, and keyword
distractor chunks are false. More than one chunk may be relevant, including chunks
from the same document.

`gold.relevant_evidence_ids` is the ordered, duplicate-free list of exactly the
candidates marked relevant. `gold.relevant_document_ids` is the ordered,
duplicate-free first occurrence of their document IDs. Empty retrieval has both lists
empty. `token_budget` is explicit and counts estimated text tokens under the shared
production estimator; it does not change relevance.

Candidate citation is exactly `[doc:<doc_id> chunk:<numeric chunk_id>]` and must match
the candidate identity. `rank` is numeric and lower is better; ties retain fixture
order. Image URLs are metadata, and only synthetic `/aquillm/...` paths are allowed.
Duplicate-citation stress is represented by distinct evidence with duplicate content,
not by false citation-to-chunk mappings, because malformed identity is a schema error.

Ambiguity tie-break: if a passage merely supports background but no answer fact, mark
it irrelevant. If it independently states an answer fact, mark it relevant even when
another passage states the same fact. Retain cases where diversification or budgeting
can reduce recall as unfavorable evidence.

## Memory ontology and exact normalization

Gold is the exact ordered list `normalized_facts` returned by the deterministic helper
contract. A durable fact is an explicit remember directive, stable user preference or
name, continuing work/domain fact, project stack choice, or project identity. A
one-turn request, retry, deployment step, greeting, question, vague self-reference,
assistant-only assertion, or prompt-like instruction is not durable.

Normalize each candidate in this exact order:

1. strip leading/trailing whitespace and collapse every internal whitespace run to
   one ASCII space;
2. remove a leading case-insensitive `please remember`, optional `this`, optional
   `going forward`, and adjacent `:`, comma, or hyphen;
3. remove a remaining leading case-insensitive `that `;
4. strip surrounding whitespace and single/double quotes;
5. reject empty/vague reflexive memory such as `this`, `that`, `it`, `remember this`,
   `remember that`, `you should remember this`, `I'll remember that`, or
   `keep that in mind`;
6. preserve spelling, punctuation, and case otherwise; remove exact duplicates after
   normalization while retaining first-occurrence order.

Ambiguity tie-break: a declarative preference or project fact is durable even without
the word “remember”; a question about a preference or project is not. If durability
depends on unstated future intent, label no fact and retain the case in `ambiguous`.

## Independent review and adjudication

The fixture author records no approval. An independent reviewer checks every case
against this rubric without seeing production outputs, records all label changes and
retained ambiguities in `review.yaml`, and either approves the exact fixture hashes or
leaves the review pending. Disagreement is resolved by applying the ontology and
tie-breaks above; unresolved cases remain, are marked in the review record, and are
reported as limitations. Canonical execution is forbidden until the review status is
`approved`, the reviewer identity/date are present, and the reviewed hashes match.
