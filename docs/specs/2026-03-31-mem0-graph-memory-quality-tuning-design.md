# Mem0 Graph Memory Quality Tuning Design

## Objective

Improve Mem0 memory quality for AquiLLM's self-hosted vector-plus-graph stack so the system remains willing to remember durable information while producing fewer noisy or self-referential graph edges.

The target operating mode is balanced:

- remember explicit or clearly durable facts reliably
- avoid becoming so strict that memory rarely writes
- reduce low-value graph relations such as self-referential or semantically empty edges

## Current State

The current integration is operational:

- Mem0 OSS initializes successfully
- Qdrant stores vector memory
- Memgraph stores graph memory
- graph writes are occurring end to end

Observed output shows that graph quality is still uneven. Example relations currently written include:

- `user -[:profession]-> scientist`
- `user -[:prefers]-> user`
- `user -[:name]-> user`

This indicates the infrastructure is healthy but extraction and graph normalization still need tuning.

## Problem Statement

The present memory pipeline favors successful writes over semantic quality. That was the right default to make the system operational, but now it creates two quality problems:

1. Extraction can produce weak or ambiguous facts.
2. Graph relation creation can preserve low-value relations instead of filtering them.

If left unchanged, the graph may accumulate noisy edges that reduce retrieval usefulness and user trust.

## Goals

1. Preserve the current Mem0 dual-backend architecture.
2. Increase precision enough to reduce obvious graph noise.
3. Preserve enough recall that explicit remember directives and durable project/user facts still get written consistently.
4. Make tuning behavior observable and testable.
5. Keep rollout low risk and reversible.

## Non-Goals

- Replacing Mem0 as the memory engine.
- Building a custom graph database schema separate from Mem0's model.
- Reintroducing Memgraph vector-index bootstrap during this tuning pass.
- UI changes for graph visualization.
- Backfilling or rewriting historical graph data.

## Tuning Target

Balanced quality means:

- explicit remember intent is high-confidence and should nearly always write
- stable user profile facts should usually write
- durable project/tooling/domain facts should often write
- speculative, low-information, or reflexive graph relations should usually be filtered

This is not a "maximize precision at all costs" plan. The system should still feel helpful and proactive.

## Main Levers

### 1. Stable Fact Extraction

Current fact extraction lives in [stable_facts.py](/c:/Users/jackj/Github/AquiLLM/aquillm/lib/memory/extraction/stable_facts.py).

This layer should become the first quality gate. It should:

- distinguish durable facts from one-turn noise more explicitly
- prioritize explicit remember directives
- better normalize project, tooling, role, and preference facts
- avoid feeding vague or self-referential statements into graph writing

### 2. Mem0 Add/Search Graph Controls

Current Mem0 execution lives in [operations.py](/c:/Users/jackj/Github/AquiLLM/aquillm/lib/memory/mem0/operations.py).

This layer already supports graph toggles and fail-open behavior. It should gain:

- clearer logging around what was attempted
- better separation between "nothing to write" and "filtered before write"
- optional counters or structured fields that help explain why a candidate memory was dropped

### 3. Memgraph Compatibility Graph Writer

Current Memgraph graph writing is handled in [memgraph_compat.py](/c:/Users/jackj/Github/AquiLLM/aquillm/lib/memory/mem0/memgraph_compat.py).

This layer should become the second quality gate. It should:

- reject obviously bad relations before writing
- normalize entity labels and relation names more consistently
- reduce self-loop and degenerate edges
- preserve useful person-tool-project-preference relations

## Recommended Approach

### Approach A: Balanced Pre-Write and Pre-Graph Filtering

Recommendation: yes

This approach improves quality in two places:

- before facts are handed to Mem0
- before graph edges are written to Memgraph

Advantages:

- best balance of precision and recall
- low deployment risk
- works with current architecture
- easy to test with targeted fixtures

Disadvantages:

- requires careful tuning to avoid over-filtering
- quality improvements depend partly on LLM extraction consistency

### Approach B: Threshold-Only Tuning

Recommendation: no

This approach would mostly adjust environment thresholds and prompts without adding structural filters.

Advantages:

- small code footprint
- easy rollback

Disadvantages:

- unlikely to fix self-referential graph edges by itself
- less predictable
- can reduce recall without meaningfully improving relation quality

### Approach C: Full Custom Graph Writer Redesign

Recommendation: not now

This would replace more of Mem0's graph behavior with custom domain-specific graph logic.

Advantages:

- maximum control
- best long-term path if Mem0 graph quality proves insufficient

Disadvantages:

- much larger implementation and maintenance cost
- unnecessary before we validate simpler guards

## Proposed Design

### Fact Extraction Rules

Add explicit extraction categories:

- explicit remember directives
- stable user identity/background
- long-lived preferences
- durable project or tooling context
- durable collaboration or operating constraints

Add soft exclusions:

- one-off tactical instructions
- transient status updates
- generic acknowledgements
- facts with weak referents or pronouns that cannot be normalized

Add normalization:

- convert obvious "remember X" instructions into fact-first statements
- collapse common wording variants for project/tooling facts
- strip assistant paraphrase noise when user intent is already explicit

### Graph Write Quality Filters

Before writing a relation, reject candidates with one or more of these properties:

- source and destination normalize to the same semantic entity when relation is weak
- relation type is too generic to be useful in graph retrieval
- relation expresses structure that is better stored as plain vector memory than graph memory
- entity names collapse into placeholders like `user`, `assistant`, or the same token on both sides without a meaningful asymmetric relation

Examples to reject:

- `user -[:name]-> user`
- `user -[:prefers]-> user`
- generic identity or preference loops that do not add retrieval value

Examples to preserve:

- `jack -[:works_on]-> aquillm`
- `jack -[:uses]-> qdrant`
- `jack -[:uses]-> memgraph`
- `jack -[:prefers]-> concise_updates`

### Observability

Add structured logging for:

- extracted fact candidates
- fact candidates filtered before Mem0 add
- graph relations filtered before Memgraph write
- graph relations written
- retry and fail-open behavior

Avoid logging secrets or raw embeddings.

### Configuration Surface

Add quality-tuning env variables only where they materially help operators. Do not create a large tuning matrix in the first pass.

Candidate envs:

- `MEM0_GRAPH_MIN_RELATION_CONFIDENCE` or equivalent only if truly needed
- `MEM0_GRAPH_FILTER_SELF_REFERENTIAL=1` default on
- `MEM0_STABLE_FACTS_STRICTNESS=balanced` if a simple mode abstraction proves useful

Prefer hardcoded balanced defaults first unless operator-level tuning becomes necessary.

## Testing Strategy

Add focused tests for realistic conversation turns:

1. Explicit remember directive writes
2. Durable project/tooling fact writes
3. Transient one-off request is ignored
4. Self-referential graph edges are filtered
5. Useful graph relations still survive filtering
6. Search behavior still works with graph enabled and fail-open paths unchanged

Test fixtures should include both:

- high-confidence user profile examples
- ambiguous/noisy turns that previously generated bad edges

## Rollout Strategy

1. Land tuning with current graph infrastructure unchanged.
2. Verify locally using controlled fact prompts.
3. Deploy to development with balanced defaults.
4. Inspect Memgraph contents and app retrieval behavior.
5. Only add new operator-facing tuning env vars if balanced defaults are insufficient.

Rollback remains simple:

- revert tuning commit, or
- temporarily disable graph mode through existing env flags if needed

## Risks

### Over-Filtering

Risk:

- the system may stop writing useful memories

Mitigation:

- keep explicit remember directives high-priority
- add tests for recall-sensitive examples
- start with soft filters targeting only obvious graph junk

### Under-Filtering

Risk:

- graph remains noisy and retrieval quality does not improve enough

Mitigation:

- log filtered and written relations separately
- inspect live Memgraph samples during development rollout

### Divergence Between Vector and Graph Memory

Risk:

- vector memory and graph memory may drift in usefulness

Mitigation:

- keep the same stable-fact candidate source for both
- treat graph filtering as a stricter second stage, not a different extraction pipeline

## Success Criteria

The tuning pass is successful when:

1. Explicit remember directives still reliably persist.
2. Durable project/tooling facts are remembered in both vector and graph paths.
3. Memgraph samples show fewer degenerate edges.
4. Retrieval remains helpful and does not noticeably regress in recall.
5. Logs explain why candidate memories were written or filtered.

## References

- [stable_facts.py](/c:/Users/jackj/Github/AquiLLM/aquillm/lib/memory/extraction/stable_facts.py)
- [operations.py](/c:/Users/jackj/Github/AquiLLM/aquillm/lib/memory/mem0/operations.py)
- [memgraph_compat.py](/c:/Users/jackj/Github/AquiLLM/aquillm/lib/memory/mem0/memgraph_compat.py)
- [Self-Hosted Mem0 Graph Memory Design](/c:/Users/jackj/Github/AquiLLM/docs/specs/2026-03-30-self-hosted-mem0-graph-memory-design.md)
