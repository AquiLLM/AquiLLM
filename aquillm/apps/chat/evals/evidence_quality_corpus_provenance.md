# Frozen synthetic evidence-quality corpus

`evidence_quality_cases_draft_v5.json` contains 80 wholly synthetic cases. It is an
input draft for Task 6, not a live evaluation or an activation result. No external
source, private user record, provider response, or current implementation output
was used to create labels. The fixture uses distinct topics, source IDs, and source
text in the development and held-out splits.

Draft v2 corrected structural defects found in draft v1. Draft v3 adds long
synthetic source chunks and unequal-length candidate pools to make windowing
bias measurable. These are pre-integration corrections; no tuning has occurred.
The original `evidence_quality_cases_draft.json` and `heldout.sha256` are
retained unchanged as provenance. Its held-out digest is
`c4f80e59b054364910b0947e536865d7b7fbc8f05b708c9e7b4672fc0b7c26b6`.
The preserved v2 corpus/hash are `evidence_quality_cases_draft_v2.json` and
`heldout-v2.sha256`, with digest
`0dc0f06de168ad594a4dee7546a379f06c7dad7532e4bcb6ee1f9af07f8c9b50`.
The preserved frozen v3 digest is in `heldout-v3.sha256`:
`fb3bf7c90107984fc6d5d077b7566e93bee8a35930d01249fc40c559571f6db7`.

Draft v4 is a narrowly scoped, independent label correction before any tuning.
It preserves v1/v2/v3 files and digests, and every V3 source, question, source
identity/revision, gold span, citation list, and action oracle. Only the eight
`negation_boundary` cases change: the controlling correction alone is listed in
the claim's required `support_ids` and `expected_delivered_support_ids`; the
superseded preliminary span remains in `gold_support` and is explicitly listed
in `optional_context_support_ids`. Root `schema_version` is `draft-4`. The full
V3 file SHA-256 was
`6715089ef542311c413605ba89ad4e956806ab1d06c022c394d948bc1f65080b`;
the V4 file SHA-256 is
`ab9a4a150c3a7c60c28ca78cc99650104cba7c003bd74d5678aba0be926b2f88`.
The held-out canonical digest changes from the preserved V3 digest above to
`918e34a76b54583be3ee72192308c443cdaab78b02209542ef0176820654ef69`
in `heldout-v4.sha256`. V4 was frozen before implementation tuning; no runner,
provider, live evaluation, or production output was consulted.

Draft v5 resolves one remaining schema ambiguity before integration or tuning.
For the same eight cases only, it moves the `contradiction` label from the
required claim's `labels` to case-level `optional_context_labels`; required
`negation` and `controlling_date` labels, claim statements, mandatory support,
all questions, source text/identities/revisions, and gold spans are unchanged.
V1–V4 files and digests remain frozen. The full V4 file SHA-256 above precedes
V5 file SHA-256
`6eb8e3617ef8e07a083b33bc502d9615017a1a210997a2c333d3cd822ddc3b07`.
The held-out canonical digest changes from V4's preserved digest above to
`01b3c9fed145c3ef46bf96552e1b4c0cc662260891af6033c114a2a2bac67395`
in `heldout-v5.sha256`. Root `schema_version` is `draft-5`. No implementation,
provider, live evaluation, or production output informed V5.

The root object has `schema_version`, `offset_unit`, and `cases`. Each case has a
stable `case_id`, `split`, `genre`, `depth`, regression classes, prior `turns`, current
`question`, synthetic `sources`, `gold_support`, `required_claims`, citation
allowlists/denylists, expected missing aspects and outcome, and an `action_oracle`.
Source identity is `(source_id, revision)`. Every source additionally has explicit
`document_id`, `chunk_id`, and `chunk_number`; related chunks share a document ID
and retain separately selectable chunk IDs. Live fixture ingestion must preserve
these authored chunk boundaries. Tail cases put the decisive support span after
codepoint 1200 within a single source chunk. In v3, tail and negation cases
each provide a short index, a medium working note, and a long main chunk with
varied synthetic record sections. The shortest long chunk exceeds 8,600
characters, and the minimum decisive tail offset exceeds 8,500. The
`windowing_cohort` metadata identifies the candidate pool and expected
short/medium/long order. These are intended multi-window fixtures; actual
1024-token window counts and aggregation behavior must be verified with the
live pinned tokenizer/model. Character length alone does not prove a window
count. Five-passage cases put five
complementary supports in five chunks of the same document. Plural and ordinal
follow-ups include displayed chunk/document identities and display order in the
prior turn. A support span's `[start, end)` indexes
the exact `text` of that source using Python Unicode codepoints; `quote` is an
independent slicing check. `required_claims[].support_ids` identify evidence
required to support each claim. `gold_support` is the universe of annotated
spans, not the recall denominator. The mandatory support set for ordinary
answer-quality recall is the union of `required_claims[].support_ids`, checked
against actually delivered supporting spans. V4/V5's
`optional_context_support_ids` are annotated background and must not be counted
as required; omission of a superseded preliminary note alone cannot lower
required support recall. Mandatory claim-label scoring uses only
`required_claims[].labels`. Case-level `optional_context_labels` are context
annotations and must not be included in a mandatory claim or support
denominator. Labels name quantities, units, conditions,
exceptions, versions, dates, negation, and superseded records where relevant.
`permitted_citations` records currently authorized revisions, while
`publishable_citations` further restricts publication to revisions available
before a cancellation or budget stop. A final answer may cite only an actually
delivered supporting span from a publishable revision. For cancellation cases
the expected answer is closed with no late publication, so `required_claims`,
`expected_delivered_support_ids`, and `publishable_citations` are empty. For
global-limit cases only the initial fact is required in the limited answer;
the blocked exception appears under `ideal_claims`. Both are marked
`quality_aggregation: safety_only` and must be kept out of normal answer-quality
aggregation.

The 10 scenarios occur four times per split, once per genre. Each split contains
10 cases per genre and 20 routine/20 deeper cases. The six mandatory regression
classes are covered in each split: tail/boundary, four complementary passages
from one document, plural/ordinal follow-up, second-search recovery, stale or
revoked evidence, and cancellation/global limits. A routine case requires a
narrow value, instruction, or authorized-state lookup. A deeper case requires
multiple complementary aspects, comparison, or additional acquisition.

The held-out canonical digest for v5 is in `heldout-v5.sha256`. It is SHA-256 of the
held-out case array serialized as UTF-8 JSON with sorted keys, compact separators,
and a trailing newline. It was saved before any tuning. The ignored
`build_and_validate.py` verifies the preserved v1 and v2 corpus/hash pairs,
reproduces v3, and refuses to overwrite it if frozen held-out content would
change. The separate ignored `derive_v4_optional_context.py` makes the constrained
V4 transformation and refuses to replace a different frozen V4.
`derive_v5_optional_context_labels.py` makes the constrained V5 transformation
and refuses to replace a different frozen V5. Independent
`audit_independent.py v5` checks V5's structure, span/citation semantics,
required versus optional context, and saved digest. The V3 generator checks
case counts, split and stratum balance, unique IDs, source separation, authorized
support, claim references, exact span slicing, source-length diversity and
distinct candidates under the 250,000-character material allowance, the
>1200 tail position, five
same-document chunks, displayed follow-up identities, safety-only publication
rules, citation lists, and all three saved digests.

These are constructed stress fixtures, not representative licensed/public
documents or human-reviewed quality labels. The Task 6 implementer must adapt
them to the final runner schema, independently review labels, add evaluator
tests, and run isolated live providers before making quality or latency claims.

## Integration provenance

The controller independently checked byte and JSON identity against frozen V5.
The independent corpus review preceded runner integration and used source spans,
qualification semantics, authorization and split stratification rather than
implementation answers. All cases are synthetic, with no external licensing or
private user data dependency. New model answers still require named human review.

Operational ops-v2 is separate: 168 literal synthetic records and 30 predeclared
candidate turns; SHA256 ae60c47e40d7fa3574d6e8fa6c32e40eba51c30121e54741316d50314f8d3136.
It preserves all ops-v1 source bytes (parent SHA256
57c199c7b254f4a10277380ceb24130ee8574e618c31a443a9a0e7d29f222da0) and adds eight
calibration logs before live feedback. These operational records never enter
frozen80 quality or latency denominators. Both source arrays were independently
checked for exact support offsets; reachability remains unmeasured.
# Exact-byte repository storage

The two promoted JSON assets disable Git newline conversion. V5 retains its
authored LF bytes; ops-v2 retains its reviewed CRLF bytes. The ops-v2 attribute
recognizes CR as the authored line terminator for whitespace checking. Staged Git
blobs and working files were compared byte-for-byte against both approved hashes;
normalizing ops-v2 to LF would change the approved full-file digest.
