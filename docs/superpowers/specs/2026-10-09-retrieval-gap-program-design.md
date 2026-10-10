# Development retrieval gap program design

Date: 2026-10-09. Authorized scope: plan and launch parallel agents to fix the remaining internal-repo gaps on development. Production is excluded.

## Objective

Close reproducible correctness and operational gaps, establish an honest embedding and evidence baseline, and promote only reviewed changes with development measurements. A successful graph call does not establish answer support. Proposed ranking and ingestion improvements require controlled experiments, not blanket enablement.

## Independent first wave

A. Embedding compatibility: trace stored/query embedding contracts, detect silent malformed or incompatible vectors and fallback spaces, implement a reproducible compatibility audit and demonstrated contract fixes. Historical precision is unknown until supported by metadata or controlled reproduction. No blanket re-embedding or model switch.

B. Evidence and activation: use existing retrieval replay and activation-v2 machinery to distinguish candidate inclusion, reranking, final packet and exact support. Fix demonstrated evaluation/attestation/scoring integration gaps. Prepare bounded development-only live verification and human-review inputs. Preserve frozen corpus/heldout labels; never invent human judgments. New modes remain off until their existing gates pass.

C. Graph capacity and validation: reproduce the document coreference candidate-limit failure with bounded synthetic data, find whether redundant work rather than required audit semantics consumes the cap, and fix only with equivalent partition/provenance and resource guards. Investigate topology-invalid mapping using current diagnostics and adversarial valid/invalid fixtures; no claim of reproducing the production incident from a generic malformed fixture.

The coordinator owns the development host, runtime observations, shared GPU experiments, integrated replay, deployment and branch integration. Agents own separate worktrees and may not mutate shared services. This is deliberate parallel implementation authorized by the user; shared interfaces are negotiated before edits, and commits are integrated serially.

## Follow-on work

After compatibility and baseline findings, evaluate a real bounded query rewrite and expert vocabulary expansion, full-text/rank fusion, reranker comparisons, structural parsing and reference-list handling one variable at a time. Preserve exact user question text; generated query variants are additional system inputs and cannot invent facts or alter identifiers/constraints. No production corpus mutation. Any parser/re-index experiment uses a dedicated development copy with explicit bounds and rollback.

The program also includes an independent pass on the previously recorded TypeScript diagnostics and migration/model drift after the first wave is under review. Inspect baseline and fix actual defects without generating unrelated migrations or changing historical schema merely to make checks green.

## Binding constraints

- Development host 149.165.150.254 only; do not contact or change production 149.165.169.204.
- Preserve current graph budgets: extractor 3000 ms, direct and extended 4500 ms, overall 5000 ms; preserve authorization, provenance, canonical encoding, caps, cancellation, worker ownership and retry-zero behavior.
- Keep current successful V2 graph transport and graph-off behavior intact.
- Do not change deployed model precision, model revision, tokenizer/template, embedding dimensions or existing vectors without a controlled compatibility result and coordinator-owned rollout.
- Evidence activation-v2 requires real per-worker capability verification, matching runtime identities, frozen comparisons and independent human review. Missing proof is unknown, never pass.
- No private questions, corpus/user identifiers, raw source content, credentials or resolved environments in tracked files. Store private evidence under the task evidence directory.
- Do not alter the 18 unrelated primary-checkout drafts.
- All shell commands begin with rtk. No subagents spawned by workers. Agents do not merge, push, deploy or message external repos.
- Reproduce before changing behavior; tests must exercise behavior, not mirror implementation. No timeout increases or disabled guards to manufacture a pass.

## Acceptance

Each change has a demonstrated failing case, regression test where appropriate, passing adjacent tests, independent review and a concrete limitation statement. Combined changes pass integration checks and the established cold combined/parent/graph-off replay. Retrieval quality claims additionally require source-checked support and no answer-key tuning. Human-dependent gates may remain explicitly pending while their mechanical prerequisites are completed.
