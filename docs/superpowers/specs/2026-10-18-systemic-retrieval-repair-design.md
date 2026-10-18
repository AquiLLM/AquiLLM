# Systemic retrieval and grounding repair

The user approved this follow-on to the RAG remediation and iterative publication
work on 2026-10-18. The objective is better discovery and use of evidence across
the collection, with Q3/Q4 as diagnostic examples, not production special cases.

## Evidence and boundaries

The initial diagnosis found missing reference passages for Q3/Q4 before final
selection. Read-only rank probes reproduced the saved candidate sets. Source
terminology improves global recall; a focused search within an already visible
document improves recall for the release comparison. These are retrieval probes,
not evidence that a final answer is correct. The implemented repair adds bounded
attempt history, document-diverse whole views and consistent visible-source
coordinates. The completed r10 evaluation still fails these two questions;
visible routing cues and appropriate next-action choices remain open issues.

The current graph has many entities but few relations. Active build diagnostics
show both disallowed and unresolved relation endpoints. The active ontology lacks
dataset lineage/composition and metric comparison relations. Whole-document
diagnostics cannot identify why one passage lost a relation. The repaired shared
extractor identity now binds adapter, schema-contract and inference behavior.
Description-preserving compiler experiments failed their quality/resource checks
and were removed; graph extraction quality remains an open limitation.

Model synthesis is required. Citation/authorization receipts verify bindings,
not entailment. Existing prompts already prohibit universal absence claims, yet
real outputs violate this. A same-model critic previously accepted known bad
answers and rejected a correct one; it must not become a semantic gate.

## Design

1. Restore the existing test baseline without weakening deadline, source-loss or
   late-result fencing assertions. Update stale test doubles and diagnostics.
2. Give the planner bounded factual attempt history and document-diverse whole
   source views. Use source terminology for new global searches, and choose
   document/adjacent search when the visible source supplies the right scope.
   Validate all coordinates against precisely the supplied views. Preserve the
   original question and existing shared budgets. History is retrieval context,
   never an instruction or answer-support verdict.
3. Give graph extraction a canonical, versioned behavior identity shared by
   production and evaluation. Existing artifacts remain readable and immutable;
   new behavior creates distinct build keys. Inspect actual extraction rejection
   stages before changing schema or endpoint resolution. Any schema extension is
   general, versioned, source-bound, and evaluated before background rebuilds.
4. Share evidence-grounding requirements across first synthesis, empty-response
   recovery, continuation and citation repair. Remove pressure to manufacture
   length or irrelevant detail. Distinguish unassessed coverage from missing
   facts, and preserve source qualifiers during repair. The main model still
   explains and compares retrieved evidence with citations.
5. Evaluate acquisition recall, final evidence, answer correctness, citation
   support, latency and failure reasons separately. Freeze held-out cases before
   inspecting their candidate outputs. Compare isolated candidates against the
   same corpus/model/settings, then update the existing PR and tracking issue.

## Global constraints

- Work in `codex/graph-development-repair`; do not create another branch.
- No study question IDs, expected answers, corpus document IDs or special-case aliases in production code.
- Keep main-model synthesis and exact source authorization; do not default to excerpt-only answers.
- A successful graph build, valid citation or model self-critique is not proof of semantic correctness.
- Keep one shared turn budget: at most 3 actions, 2 planner calls, 45 unique sources and current byte/token/scoring limits; preserve stricter operator deadlines.
- Do not widen retrieval caps or require graph evidence merely to improve benchmark scores.
- No request-time graph construction; extraction changes require versioned background artifacts.
- Keep private source text, study traces and corpus-specific diagnostic scripts out of tracked files.
- Root alone runs SSH, shared database tests, model calls and GPU experiments; serialize these resources.
- Keep the serving development stack unchanged while isolated candidate verification is incomplete.

## Acceptance

Focused regressions must exercise authorization, budget exhaustion, truncation,
unseen document coordinates, follow-up history and all synthesis paths. Retrieval
quality must be measured across the study and held-out questions; passing unit
tests is not a quality claim. Record rejected extraction/schema experiments as
rejected, without marking graph quality complete. Deploy only an accepted,
tested candidate and report precisely which code and artifacts are active.

Matched live evaluation is complete at immutable r10 head
`079ab3a895d3e7f8b9f540275f0c8dbffaf2574c`: all 26 cells succeeded operationally
(20 study-arm cells and six frozen held-outs), but **quality acceptance has not
passed**. Q1/Q2 remain partial; Q3/Q4 fail required coverage; Q5–Q10 retain core
answers with citation/scope defects in some cells. Held-out required claims
improve from 8 correct / 1 partial / 12 missing to 18 / 3 / 0, with six of seven
frozen targets acquired and delivered versus four at baseline. Core content and
incidental/citation defects are graded separately, including one borderline
negative-scope judgment. These results are not a generalization claim.

The later targeted resolver-identity fixture repair in
`44957ae0cb2aed157fd187d8143d6a76325daaa5` passed 171 tests and does not change the
evaluated head. Independent reviews of P1/P2 are complete: reject the navigation
output contract and the 32k context increase. P1 adds structural/span failures
and cost without a generic routing gain; P2 exposes more pointers but produces
no clean generic gain and retains three 32k timeouts as failures. P3's 48
completed requests show no useful-action gain from removing support generation
at 16k (both arms 4/8 generic); that shape remains experimental. Its diagnostic
32k improvements do not authorize a production cap increase.

P4's separately frozen 48 long-case and 32 small-control requests are complete
and independently reviewed at 16k. Reject balanced packing and the combined
treatment: the combination improves one generic comparison case, but balanced
packing misroutes the external-method diagnostic in both blocks under both
strategies. Do not promote pointer-only strategy: it adds no generic gain, and
its release-comparison improvement occurs in only one of two blocks per context.
The small controls pass 16/16 per strategy with no gain or regression. No
production contract or cap change is approved. Prompt and packing experimentation
ends for this investigation; any next source-candidate diagnosis is separate.
These decisions do not revise the r10 findings or failed quality acceptance.
The serving development stack remains unchanged.

## Evidence-driven implementation refinements

The source-order repair in `d15430ab` preserves returned retrieval-provenance
order among the same verified, deduplicated sources, replacing cache insertion
order. Existing revision filtering, planner latest-source priority and caps are
unchanged. The source-order/context/acquisition suite passed 32 tests and
independent review passed. This repairs information loss, independently of P4;
it does not establish Q3/Q4 retrieval or answer success.

The lexical configuration repair `888203a6` wires the existing full-text, RRF
and protected-exact settings through `env_bool`, each defaulting to false;
protected exact remains effective only with RRF. Its 24-test suite and independent
review passed. This restores opt-in operability without activation or a recall
claim. Readiness inspection records 530 documents, 2,701 text and 499 image
chunks, zero current FTS projections and zero structured sections; all three
flags remain false. The mixed-modality FTS gate is intentional, and changing it
or the native-to-scored fallback requires a separate decision.

The verified optional rewrite-budget repair `409f9ef4` reserves capacity for a still-pending
original search, preserves vocabulary precedence, and forwards actual prior initial
acquisition explicitly. After RED (5 failures / 6 passes), GREEN passed 151 tests
across seven files with three existing warnings in 391.99 seconds; independent
review, file-length and diff checks passed. Query rewriting stays disabled in the
frozen profile. This later repair establishes no r10 quality gain; r11 is unstaged.

The frozen neighbor policy is rejected: it reaches zero missing targets and
removes observed acquisition paths for five final-selected sources. This is a
coordinate counterfactual, not a replayed answer outcome. The 16 text-only FTS
probes completed without errors; 12 returned zero matches and none of the 11
frozen targets matched anywhere. Reject this literal English plain-query treatment
as a remedy for the recorded misses, not FTS generally: inputs were unchanged
unresolved-context questions, not production-routed queries. The separate all-lexeme
OR contrast completed 44 arm observations and passed mechanical checks. It gains
4/11 targets within eight over plain FTS, but all four were already acquired and
delivered in r10; Q3/Q4 remain 0/4 at that cutoff. Each of the three generic controls
loses its useful source from the returned eight. Reject unconditional OR replacement,
integration and backfill. These SQL counterfactuals do not exercise production
admission, final evidence or answers and establish no r11 result.

Offline inspection confirmed that the pinned GLiNER2 relation builder discards
descriptions. Both versioned structure/field compiler variants retained those
descriptions but produced no accepted relations within their resource bounds. The experiment
was rejected and the unused compiler removed from product code. Shared extraction
identity remains implemented; no endpoint validation was weakened and no serving
artifact was rebuilt. Another compiler implementation is not the current plan.

A frozen held-out question exposed a local pronoun being mistaken for a reference
to a previous source. The implemented bounded same-question noun-anaphora repair
preserves ambiguity handling and explicit source-reference authorization.
This gate distinguishes local question content from historical source references;
it does not choose between ordinary local subjects. A question mentioning two
local objects must not acquire an unrelated historical document merely because
the pronoun could refer to either object. Pass the unchanged question to retrieval
and synthesis, while retaining actual source, citation and quotation barriers.

Integration testing exposed quadratic SQL string serialization when saving a
large legacy source snapshot. A new reversible migration now preserves canonical
bytes and immutable identity/hash checks with bounded work. ASCII, Unicode,
controls, escaping and realistic large-source equivalence/integration checks
passed without shrinking the fixture or bypassing the trigger.
