# Graph capacity implementation plan

**Goal:** Remove redundant acronym initialism-collision audits while preserving complete partitions, accepted edges, cannot-link decisions, provenance, and resource caps.

**Architecture:** Keep the existing resolver and cap. In the sparse acronym fallback, enumerate pairs with an acronym/pronoun/local lowercase definition endpoint, and full/full pairs sharing a name, carrying identifiers, or sharing a base with different versions. Only full/full normalized-name mismatches are omitted. Pair sorting and merge precedence stay unchanged.

**Tech stack:** Python, pytest. **Spec:** `docs/superpowers/specs/2026-10-09-retrieval-gap-program-design.md` in the integration worktree.

## Constraints

- Development only; no production contact, deployment, service mutations, or automatic artifact rebuilds.
- Preserve complete supported partitions, cannot-link/provenance rules, canonical results and hard caps.
- Preserve graph budgets: extractor 3000 ms, direct and extended 4500 ms, overall 5000 ms.
- No DTO change. Coordinator approved a resolver-version bump because audit/checksum changes.

## Evidence and exact implementation

Reproduced: 1,025 distinct `Rapid a{index} generation` full forms plus undefined `RAG` exhaust `MAX_DOCUMENT_DECISIONS = 524_288` while adding full/full pairs in `add_complete(unique_group)`. The clique has 525,825 pairs; only 1,025 acronym/full pairs are meaningful. The intended result is 1,026 singletons with those 1,025 explicit undefined-acronym decisions.

### Task 1: Bound acronym fallback enumeration

Files: `resolution/coreference.py`, `resolution/__init__.py`, `tests/test_coreference_capacity.py`, `tests/test_coreference_sparse_partition.py` under `aquillm/apps/knowledge_graph`.

Interfaces: `resolve_document_mentions(mentions, ontology) -> ResolutionResult` stays fixed; only sparse audit contents omit irrelevant negative decisions. `DOCUMENT_RESOLVER_VERSION` becomes `document-coreference-v3-sparse-acronym`. This constant already feeds extraction `_build_identity`, build submission identity/config checksum, cluster keys, and result checksum. No acceptance bypass or migration.

- [x] Write/run failing actual-cap reproduction described above; observed typed entity capacity failure in candidate enumeration.
- [x] Add a small exhaustive comparison with collision, aliases, shared/conflicting identifiers, versions, ambiguous/lowercase/pre-definition/source-separated acronyms. Compare complete clusters/memberships and every meaningful decision in order; verify reverse-input canonical determinism.
- [x] Replace only `add_complete(unique_group)` fallback using these buckets:

```python
for mention in unique_group:
    if mention.is_acronym or mention.is_pronoun or has_local_acronym_definition(mention):
        for other in unique_group:
            add_pair(mention, other)
    else:
        full_names[mention.normalized_label].append(mention)
        full_bases[mention.base_key].append(mention)
        if mention.identifier:
            identified_full.append(mention)
for group in full_names.values():
    add_complete(group)
add_complete(identified_full)
for group in full_bases.values():
    if len({mention.version_signature for mention in group}) > 1:
        add_complete(group)
```

- [x] Verify true ambiguous/identifier conflict cliques still fail closed at the unchanged cap; check exact budget boundary with a reduced cap, and compare complete partitions at the real cap.
- [x] Run coreference/capacity/sparse/coordinate suites locally; coordinator runs Django persistence/build identity suites in isolated Linux.

### Task 2: Investigate topology diagnostics

- [x] Trace current `topology_invalid` mapping from Memgraph adapter/loader and test valid and malformed V1/V2 snapshots, branch isolation, cap/deadline mappings, payload-free diagnostics.
- [x] Change diagnostics only for a reproduced development gap; otherwise report current observed coverage and leave historical production incident unresolved.
- [x] Run topology suites and record exact selection and output. Commit owned files after fresh verification; write full evidence report with limitations.

Local test invocation uses ignored `artifacts/graph-capacity/run_pure.py` (minimal Django settings solely for existing import-time loopback reachability check), task-local pytest config preserving project config except unavailable Django setting and local marker registration, and `-p no:django`. Database tests remain skipped, not claimed as integration evidence.

## Completion evidence

Local resolver suites: 182 passed, 1 PostgreSQL skip, 7 Django-dependent cases deselected after observed AppRegistryNotReady failures. Selected topology suites: 74 passed. A separate 200-case deterministic comparison against the original sparse implementation preserved complete clusters/memberships/provenance and retained decision order; 19,704 omitted decisions were all normalized-name mismatches. Ruff and git diff --check passed. No topology behavioral defect reproduced; the historical production incident remains unresolved. Coordinator Linux persistence/build-identity verification and independent review remain pending.
