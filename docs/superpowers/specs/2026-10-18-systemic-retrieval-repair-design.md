# Systemic retrieval and grounding repair

The user approved this follow-on to the RAG remediation and iterative publication
work on 2026-10-18. The objective is better discovery and use of evidence across
the collection, with Q3/Q4 as diagnostic examples, not production special cases.

## Evidence and boundaries

The current candidate retrieves neither reference passage for Q3/Q4 before final
selection. Read-only rank probes reproduce the saved candidate sets. Source
terminology improves global recall; a focused search within an already visible
document improves recall for the release comparison. These are retrieval probes,
not evidence that a final answer is correct. The planner has no attempt history,
packs first-fit views from cache insertion order, and explicitly prefers global
search. Its schema and server validation disagree about document coordinates.

The current graph has many entities but few relations. Active build diagnostics
show both disallowed and unresolved relation endpoints. The active ontology lacks
dataset lineage/composition and metric comparison relations. Whole-document
diagnostics cannot identify why one passage lost a relation. Extractor identity
omits adapter, schema-contract and inference behavior, allowing stale immutable
artifacts to be reused after code changes.

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
tests is not a quality claim. If an extraction/schema change is not validated,
record it as pending rather than marking graph quality complete. Deploy only a
tested candidate and report precisely which code and artifacts are active.

## Evidence-driven implementation refinements

Offline inspection of the pinned GLiNER2 API proves that its relation builder
discards relation descriptions. Evaluate a deterministic structure/field compiler
that retains relation semantics and typed endpoint descriptions. Keep thresholds,
source-span checks and independently emitted typed entities unchanged. Bind this
compiler and effective inference options into extraction identity. Repeat the
frozen extraction sample before enabling the compiler or rebuilding artifacts;
schema retention alone does not establish extraction quality.

A frozen held-out question exposes a local pronoun being mistaken for a reference
to a previous source. Repair bounded same-question noun anaphora generally, while
preserving ambiguity handling and explicit source-reference authorization.

Integration testing also exposes quadratic SQL string serialization when a large
legacy source snapshot is saved. Replace it through a new reversible migration
with byte-identical canonical serialization. Preserve all immutable identity and
hash checks; do not shrink the source fixture or bypass the trigger. Verify ASCII,
Unicode, controls, escaping and realistic large source bodies against Python's
canonical encoder and the previous SQL implementation.
