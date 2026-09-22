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
        case, llm_if, preserve_citation_settings=True, stream_output=True,
    ))
    for paper in case.papers:
        reports.append(await run_acceptance_case(
            case, llm_if, omitted_doc_id=paper["doc_id"],
            preserve_citation_settings=True,
            stream_output=True,
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
`stream_output=True` supplies an in-memory async callback, matching the chat
completion path, and reports whether a final assistant event arrived with content
identical to the stored answer. This exercises callback delivery, not websocket
transport. It preserves the deployment's final-answer streaming configuration.

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

## Live checks and follow-up correction

The first six real-provider runs used `qwen3.6:27b-mtp-awq`. All six retained the
required passages and exact citation allowlist at the final OpenAI-compatible SDK
boundary. Three met the lexical answer checks; the complete two-paper cases
omitted citations on combined conclusions and included unsupported explanatory
claims, and one disagreement ablation timed out. These were not an acceptance
pass. Manual inspection also found unsupported background in answers that passed
the narrow lexical checks.

The initial probe used a 1,024-token synthesis limit and triggered empty-response
retries. The deployment actually uses 4,096 synthesis tokens, a final retrieval
limit of 12, and a 7,000-token approximate evidence-text budget. Repeating the
capacity case with the deployed synthesis setting returned a concise answer in
one call but still omitted citations on the computed difference. The live harness
now preserves the deployed synthesis and citation settings and exercises the
deferred final-stream callback; it still substitutes synthetic ranked retrieval.

Two production causes were corrected: direct empty-response retries asked for
400–900 words, and legacy citation exemptions accepted some fluent answers despite
uncited factual bullets. Direct synthesis now carries request-only grounding
instructions: proportional detail, all supporting citations at combined claims,
explicit calculations, preserved conditions/disagreement, and no invented causes
or missing facts. Its retries follow the same instructions. The citation
exemptions remain available to the ordinary tool loop, but direct synthesis now
attempts citation repair for detected omissions before deferred delivery. This
does not add semantic entailment verification or guarantee citation completeness
for every prose claim.

After these corrections, 312 relevant tests passed with the same one pre-existing
logging test deselected. The acceptance suite now has eighteen passing tests,
including valid paraphrases and citation repair before the final stream callback.
Independent review found no blocking issue. Final real-provider reruns are
recorded below after deployment.

The next live run passed five of six cases, with all six passing final SDK
evidence/citation checks and callback-to-stored-answer equality. The remaining
capacity comparison was correct but uncited prose; the validator previously
checked factual bullets more strictly than sentences. Direct synthesis now opts
into a numeric-prose citation-presence check and requests repair for such
omissions. Decimal values, adjacent citations, code, images, and source-list
boundaries have regression coverage. This is intentionally a citation-presence
check, not a semantic judge. Negative-evidence instructions also scope missing
measurements or studies to the selected passages.

Verification after the numeric-prose fix: 328 relevant tests passed, with the
same pre-existing logging test deselected. Independent review caught and verified
a regression fix for a bold Sources heading that could otherwise lend citations
to preceding prose. No blocking review findings remained.

An earlier real collection probe using normal deployment limits retrieved twelve retained
rows from ten documents and generated an answer citing nine chunks from seven
documents. Both graph branches succeeded for that single-question probe; two
cited chunks were also graph candidates. None of the cited chunks was exclusive
to graph retrieval, so this does not prove a graph-specific quality improvement.
A separate three-query run had direct-branch failures (`backend_unavailable`,
`direct_no_seeds`, `extractor_provenance`) while extended retrieval returned
evidence. Graph reliability across multiple queries remains a separate issue;
this work does not declare the whole graph pipeline consistently healthy.

## Final deployed acceptance results

Application revision: `b694615ee85fd80e8883fb9d7d56c66f196d9141`, following
`6689a20a` (selection/handoff) and `ab0e4491` (grounding/repair). Each was committed
and pushed to `development` before the clean development checkout was
fast-forwarded and its web service rebuilt/restarted. Web, query gateway,
extractor, and Redis health checks passed; all ten Compose services were running.
Environment files, temporary keys, private document contents, and raw server logs
were excluded from Git commits.

Final runs used the deployment's `qwen3.6:27b-mtp-awq`, 4,096 synthesis tokens,
enabled citation enforcement, and deferred final streaming. Each case used the
same deliberately small synthetic retrieval fixture described above.

| Case | Automated answer check | Review |
| --- | --- | --- |
| Capacity, both papers | Pass | 18 and 7 litres are correctly cited; the calculated 11-litre difference cites both inputs after one citation-repair call. |
| Capacity, A omitted | Pass | Gives B's 7-litre result and withholds dry capacity and the comparison. |
| Capacity, B omitted | Pass | Gives A's 18-litre result and withholds humid capacity and the comparison. |
| Disagreement, both papers | Pass | Preserves the 20 C/+12% versus 35 C/−9% results, cites both for opposing directions, and does not invent a cause. |
| Disagreement, A omitted | Conservative false positive | The lexical checker flags “differs,” but B itself states that qualitative fact. The answer cites B and explicitly withholds A's missing numeric result and the direction comparison. Independent review confirmed this distinction. |
| Disagreement, B omitted | Pass, with wording qualification | Withholds field results and comparison. “No field trial was performed” should more explicitly say “within the cited controlled study.” |

All six passed evidence coverage, serialized packet equality, exact citation
allowlists, and final SDK request checks after provider preprocessing. All six
delivered one final callback matching the returned answer, with no extractive
fallback. Seven provider calls served the six cases, including the one citation
repair. Both complete cross-document questions passed their expected factual and
claim-local citation checks. Do not summarize the six outputs as unqualified
semantic passes: the table preserves the lexical false positive and wording
qualification, and the checker cannot establish general semantic entailment.

The final real-collection check on `b694615e`, using the deployed retrieval and
synthesis settings, retained twelve passages from ten documents. Its generated
answer contained ten chunk citations spanning eight documents, all inside the
selected packet and collection. Both direct and extended graph branches
succeeded; three cited chunks were graph candidates, and one was absent from the
observed baseline candidate set. This demonstrates a graph-only contribution in
that run, not a measured accuracy uplift. No extractive fallback was used. Private
passages and answer text were not included in this report or Git.

These probes do not test browser/websocket transport or persist conversations.
They establish a bounded synthesis/hand-off acceptance result, not broad corpus
recall, universal answer faithfulness, new inferred graph relationships, or a
measured quality gain uniquely attributable to the graph. Large-context provider
trimming and intermittent multi-query graph failures remain separate limits.

After the production changes, the same acceptance test file passed all 16 tests
in 0.18 seconds on 2026-09-22. This establishes the deterministic transport and
citation regressions only; real-provider answer quality is assessed separately.
