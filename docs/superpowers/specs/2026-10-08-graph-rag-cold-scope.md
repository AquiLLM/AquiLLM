# Cold combined-scope graph deadline design

## Goal and scope
Make the observed combined-scope development retrieval cases complete both graph branches after service-cache reset, under the existing direct/extended 4500 ms and overall 5000 ms budgets. Deploy only to development after verification. Production promotion and the unreproduced production topology-invalid report are outside this change.

## Evidence
Cold paired topology reads spend time in Bolt decoding and repeated Python string validation. One-transaction reuse and opaque-key validation alone did not resolve the failure. The matching Neo4j Rust extension alone passed isolated topology replays but still failed full cold HTTP retrieval. Combining the extension with equivalent compiled opaque-key and gateway control-text validators passed the initial full cold replays. Complete measurements are held privately outside the repository; no corpus identifiers or questions belong here.

## Design
Keep the current four-family gateway protocol, all Cypher, pagination, caps, authorization/readiness/provenance checks, deadline values, retry policy, canonical bytes and cache keys. Replace only the two measured Python character loops with compiled regular expressions implementing exactly the same accepted language and exception precedence. Add neo4j-rust-ext==5.28.4.0 to the knowledge-graph-local optional dependency set, matching existing neo4j==5.28.4. The KG Docker build must verify that the compiled PackStream encoder and decoder actually load, preventing a silently inactive accelerator.

Opaque keys remain exact built-in strings consisting of exactly 64 lowercase ASCII hexadecimal characters. Gateway text remains exact built-in str, obeys its existing code-point length limit, and rejects C0, DEL and every surrogate. A size failure still precedes forbidden-text failure, with the supplied size_error class. No normalization, permissive decoding or change to exception text.

## Validation and release gate
- Preserve acceptance/rejection, exact exception classes, limit precedence, canonical bytes and the existing pinned gateway schema checksum.
- Regression fixtures include every C0 character, DEL, all surrogates, boundary Unicode values, combining/astral characters, string subclasses, nonstring types, uppercase/length/whitespace key defects and valid boundary keys.
- Record a before/after benchmark for large valid text and opaque-key batches; keep wall-clock performance assertions outside the ordinary test suite.
- Run relevant projection/contract/gateway/deadline/retrieval tests and real disposable Memgraph regressions. Assert compiled extension loaded in the built KG image and rerun driver/cap/failure tests with it.
- Confirm full canonical snapshots match baseline references for the authorized development cases using private evidence.
- Repeat full retrieval-only combined-scope cases with fresh gateway caches, test parent-only scope and graph-off/restoration, and verify both branches rather than accepting vector fallback as success.
- Keep prior deployment images/env rollback material and preserve unrelated primary-worktree drafts.

