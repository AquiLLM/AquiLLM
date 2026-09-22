# Cross-document acceptance cases

These deliberately synthetic papers contain no uploaded or private document text.
The canonical fixtures and reusable harness are in
`aquillm/apps/chat/tests/cross_document_acceptance_support.py`.

## Expected evidence and answers

| Case | Paper A supplies | Paper B supplies | Required synthesis |
| --- | --- | --- | --- |
| `complementary-capacity` | Dry-mode Lumen capacity is 18 litres per cycle, citation `[doc:synthetic-capacity-a chunk:101]`. | Humid-mode capacity is 7 litres per cycle, citation `[doc:synthetic-capacity-b chunk:201]`. | Humid capacity is 11 litres per cycle lower. The calculation must cite both papers. |
| `conditioned-disagreement` | Controlled trial at 20 C reports a 12% response increase, citation `[doc:synthetic-conflict-a chunk:301]`. | Field trial at 35 C reports a 9% response decrease, citation `[doc:synthetic-conflict-b chunk:401]`. | Report the opposing directions with both citations, preserve the differing trial conditions, and do not invent a causal explanation. |

Each case asks two retrieval questions. The primary and first aspect searches
return A's necessary result followed by A's redundant methods snippet; the second
aspect returns B's necessary result followed by A's. Both necessary results are
therefore in the retrieved union even with two results per search. Reciprocal rank
fusion favors the repeated A rows, so truncation before document diversity loses
B. The final two-row packet must retain each necessary source once. A separate
four-row-limit run checks that a row removed by the per-document cap cannot leak
back into the provider's serialized tool result or citation allowlist.

This checks preservation of evidence that retrieval has returned. It does not
demonstrate discovery of a paper absent from all retrieval results.

## Single-source ablations

Run each case with A omitted, then B omitted. The model must state the retained
paper's supported fact with its citation, identify the missing evidence, and avoid
stating the absent paper's result or a comparison that requires both. A missing
paper's citation must appear neither in the provider evidence/allowlist nor the
answer. These are four additional runs, for six real-provider runs in total.

## Running against a real provider

After normal Django initialization, construct the development deployment's normal
provider instance, then run this sequentially in an async function:

```python
from apps.chat.tests.cross_document_acceptance_support import CASES, run_acceptance_case

reports = []
for case in CASES:
    reports.append(await run_acceptance_case(
        case, llm_if, preserve_citation_settings=True,
    ))
    for paper in case.papers:
        reports.append(await run_acceptance_case(
            case, llm_if, omitted_doc_id=paper["doc_id"],
            preserve_citation_settings=True,
        ))
```

`llm_if` must be an actual configured provider to assess generated-answer quality.
The harness substitutes only the retrieval boundary and records the actual
provider request before forwarding it. Real query building, fusion, packet
selection, synthesis, completion, and citation handling still run. No database
fixture is needed. Environment/retrieval patches are process-global, so use a
dedicated verification process and do not run cases concurrently with application
traffic. The harness creates no collections or documents and never records API
credentials. `preserve_citation_settings=True` keeps the deployment's citation
enforcement and Sources-append settings; deterministic unit tests use explicit
citation defaults to avoid dependence on a developer's shell environment.

Review `outcome`, `evidence_coverage_passed`, `serialized_rows_match`,
`exact_allowlist_passed`, and `answer_checks` in each report, together with the
generated `answer`. Save the actual model/deployment identity beside the reports.
The lexical answer checker requires expected facts and supporting citations in
the same sentence, rejects citations outside the evidence, and ignores a trailing
Sources list for claim attribution. It also tests missing-source restraint. Its
`requires_semantic_review` field is always true: paraphrases can cause false
negatives, and numeric/keyword checks cannot certify semantic entailment, causal
restraint, or the correctness of every additional claim.

## Deterministic regression scope

`test_cross_document_acceptance.py` uses a scripted provider only to inspect
transport behavior and exercise real citation repair. It verifies both-paper
retention, packet-only serialized rows, exact current citation scope (including a
prior-turn citation challenge), source ablations, and positive/negative checks of
the acceptance grader itself. Hand-authored reference answers are test inputs;
passing these tests is **not evidence of model answer quality**. A real-provider
run plus semantic inspection remains necessary for that claim.

Initial complete regression run on the unmodified production path: seven expected
failures and nine passes. The failures showed loss of B at final limit two in both cases,
discarded A snippets in the provider request, historical citation scope leakage,
redundant A snippets leaking in the two omit-B runs, and acceptance of a stale
citation in the final generated response.

## Implementation and local verification

Direct retrieval now requests a candidate pool up to three times the final row
limit, within the existing retrieval-tool ceiling of 15. Reciprocal rank fusion
balances documents before truncation and keeps each document's internal rank
order. The packet skips passages that cannot fit its approximate text-token
budget instead of stopping before later, shorter passages. This preserves more
of the returned papers; it does not guarantee relevance or coverage of every
paper in a collection.

The shared synthesis entry rebuilds both serialized tool content and structured
payload from the selected packet. Counts, titles, citations, and image references
describe those rows only. An isolated request copy omits earlier tool evidence,
including citation/image metadata and files, while persisted conversation history
is preserved. Both automatic direct retrieval and manual search commands use this
handoff. The ordinary model-selected tool loop is outside this change.

Local verification: 303 tests passed across retrieval, synthesis, manual commands,
provider citations/images/context handling, and the new acceptance suite. One
existing context-packer logging test was deselected after the initial broad run
reported 303 passes and that one failure: it expects the obsolete text
`context_pack stats`, while unchanged production code emits
`obs.llm.context_packed`. This is not a fully green repository-wide test run.
The sixteen new acceptance tests passed, and an independent code review found no
blocking issue.

The exact citation-scope guarantee applies at the synthesis handoff. Existing
provider compression or overflow handling can subsequently shorten evidence near
context limits. These changes do not redesign that provider behavior; live tests
must distinguish the normal bounded request from extreme-context behavior.

After the production changes, the same acceptance test file passed all 16 tests
in 0.18 seconds on 2026-09-22. This establishes the deterministic transport and
citation regressions only; real-provider answer quality is assessed separately.
