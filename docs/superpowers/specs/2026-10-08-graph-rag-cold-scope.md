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

## Follow-up design: one bounded snapshot exchange
Clean-image verification of the initial optimization passed five of six cold combined-scope cases; one extended branch exhausted its deadline on the last family HTTP call. This supersedes treating the initial optimization as sufficient. Keep it, and remove three repeated gateway exchanges for complete snapshots that fit the current wire limit.

Add POST /v2/topology/snapshot with a separately pinned canonical descriptor/checksum. Preserve V1 route, enum, descriptor/checksum, all Cypher/source caps and existing four-family behavior. The V2 request carries the existing bounded scalar parameters and absolute deadline once. The admitted worker decodes once, always checks fresh generation manifests (including snapshot-cache hits), builds the existing complete bounded snapshot and serializes it once. No cross-branch cache sharing or DB concurrency change.

The V2 successful response is a closed union: complete snapshot_json plus fresh manifest rows, or an explicit use_family_transport discriminator plus fresh manifest rows. The latter is selected only after a valid complete snapshot is built and found too large for the unchanged configured/hard wire response ceiling. It instructs the loader to perform only the existing three remaining family reads under the same deadline and caps. It is a transport choice, never an error-triggered retry or a partial result. This preserves V1-valid split snapshots that exceed the combined 1 MiB wire envelope. The independent 2,000,000-byte decoded-snapshot bound remains enforced. An invalid snapshot/source overflow does not select family transport.

The loader independently matches returned manifests and applies existing canonical snapshot decoding, authorized document/collection equality, node/edge/depth caps, audit/evidence/provenance validation and failure mappings. Check absolute deadlines after encoding and decoding; late work cannot become success. Keep worker occupancy until actual completion and retry0.

Add strict KG_TOPOLOGY_GATEWAY_SNAPSHOT_ENABLED configuration, default disabled. V1 remains selected when disabled and when a driver does not advertise the optional snapshot capability. This development rollout explicitly enables it after the gateway supports both routes; production configuration is unchanged. Unknown route, auth/schema/deadline or other error never triggers a V1 retry. Graph-off remains network-free.

Test one exchange and one parameter decode, byte equality with V1, fresh manifests on warm hits, both delivery alternatives and exact wire/snapshot size boundaries, canonical malformed envelopes, fixed error mappings, authorization before runtime access, deadline expiry at every stage, bounded worker ownership, V1 compatibility, strict opt-in configuration, and graph-off/restoration. Repeat clean-image and deployed cold replays before claiming resolution.
