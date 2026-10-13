# RAG remediation execution decisions

The user authorized implementation of the remediation program and a review PR in AquiLLM after the planning deliverable. Code targets development at `6e756e252999ee6ccf604f88ea5326f49ddaa6d4`. Production deployment and private-corpus activation are separate.

Tracking: [adversarial audit issue](https://github.com/AquiLLM/AquiLLM-internal/issues/6) and [existing internal retrieval issue](https://github.com/AquiLLM/AquiLLM-internal/issues/4). Private study content stays out of the public repository.

These amendments settle shared implementation contracts from the independent execution preflight. The original design remains binding except for the explicit clarifications below. None records completed implementation or passed runtime validation.

- G4 depends on reviewed G2 and G5.
- G5 application-path acceptance depends on I4's legacy representation store; it does not depend on I5 activation.
- I1 owns the source-artifact schema proposal; ingestion migration 0003 depends on 0002_expand_source_file_max_length. The coordinator integrates registry and migration changes serially.
- R1 includes existing window HTTP and selection-provider routes and the cache/pair-worker changes necessary for one shared weighted admission mechanism.
- Source/container bytes carried from parser to worker are transient and must not enter logs, frontend/history payloads or generic persisted metadata.
- A coordinator plus at most three implementers/reviewers may work concurrently in isolated worktrees. Each task receives independent review before integration.

### D3: approved source occurrence transport

I1 owns these types in `lib/parsers/source_contracts.py` and their validated serializer:

```python
@dataclass(frozen=True)
class ArchiveMemberStepV1:
    container_sha256: str
    member_path: str
    archive_ordinal: int

@dataclass(frozen=True)
class SourceArtifactOccurrenceRefV1:
    occurrence_id: int
    occurrence_key: str
    artifact: SourceArtifactRefV1
    archive_lineage: tuple[ArchiveMemberStepV1, ...]
```

Add `ExtractedTextPayload.source_occurrence: SourceArtifactOccurrenceRefV1 | None = None` and `extraction_revision_id: int | None = None`, beside the planned optional `source_artifact` and `extraction`. In-process extraction can precede persistence, but I3 manifest registration requires a resolved occurrence and extraction revision belonging to that occurrence. Raw untyped metadata is not an alternate identity authority.

Lineage is root→leaf. Ordinals are exact nonnegative integers from actual archive member enumeration, including duplicate names; booleans are rejected. Top-level files use an empty tuple. Each member path is normalized/validated with existing path-safety rules. The final step must match the artifact's member_path/container_sha256. Persist `archive_lineage` JSON plus indexed unique `occurrence_key` in I1's allocated ingestion0003 along with the planned fields. Registration verifies an existing key's complete fields instead of overwriting them.

Occurrence key payload is exactly `{'version':'source-occurrence-v1','artifact_sha256':artifact.sha256,'archive_lineage':[{'container_sha256':s.container_sha256,'member_path':s.member_path,'archive_ordinal':s.archive_ordinal} for s in lineage],'original_filename':original_filename,'content_type':content_type}`. Hash using the canonical JSON convention below. The database occurrence PK is a locator, never part of the deterministic key. The output-key payload uses version `ingestion-output-v1` with keys `item_id`, `occurrence_key`, `parser_key`, `output_kind`, `element_key`; values are exact persisted item identity and canonical nonempty strings. This preserves identical bytes at two ordinal positions and resumes the same upload without conflating different uploads. Authorization still comes from output/document bindings, never these digests.

Recommended explicit persistence API, owned by I1:

```python
def register_source_occurrence(
    artifact: SourceArtifactRefV1, *,
    archive_lineage: tuple[ArchiveMemberStepV1, ...],
    original_filename: str, content_type: str, using: str,
) -> SourceArtifactOccurrenceRefV1: ...
```

ExtractionRevision retains the plan's occurrence/parser/manifest uniqueness. Its `ordered_elements` serializer uses exact source-contract field names, arrays for tuples, explicit nulls, no unknown keys and validated roundtrip; codepoint offsets do not become byte offsets. Same-path identical-byte ordinal duplicates, nested duplicate containers, and retry identity are required regression cases.

### D4: V2 fusion transport

Baseline `chunk_search_fusion.py` already has strict `BaselineCandidate`, `GraphCandidate`, `FusedCandidate`, and `FusionResult`. Its V1 result enforces a baseline prefix, so keep it unchanged. Add in R3's `candidate_rank_fusion.py`:

```python
ChannelV2 = Literal['dense', 'exact', 'trigram', 'full_text', 'graph_direct', 'graph_extended']

@dataclass(frozen=True, slots=True)
class ChannelRankV2:
    channel: ChannelV2
    rank: int

@dataclass(frozen=True, slots=True)
class FusedCandidateV2:
    chunk_id: int
    channel_ranks: tuple[ChannelRankV2, ...]
    rrf_score: float
    protected_exact: bool

@dataclass(frozen=True, slots=True)
class FusionResultV2:
    candidates: tuple[FusedCandidateV2, ...]
    policy_key: Literal['rrf60-v2', 'rrf60-protected-exact-v2']
    scope_key: str
    order_checksum: str

def fuse_channel_ranks_v2(
    channels: Mapping[str, tuple[int, ...]], *, limit: int,
    scope_key: str, protected_exact_ids: tuple[int, ...] = (),
) -> FusionResultV2: ...
```

Keep the plan's `fuse_channel_ids` as the pure ID-only projection of the same algorithm. Channel enum order above is canonical; ranks are one-based after first-occurrence deduplication within each channel. Require positive exact chunk IDs, finite exact floats and bounded exact integer limits. Score is `math.fsum(1/(60+rank) for ranks in canonical channel order)`. Sort by descending score, descending channel count, ascending best rank, ascending chunk ID. At most one validated exact ID can be protected and moved first; reject requests with multiple protected IDs or a protected ID absent from exact ranks. Empty input yields an empty result; no graph ID enters before existing authority validation. Scope key binds the full retrieval identity; private graph authority remains in the runtime envelope, never in a public fusion DTO.

Checksum payload has version `fusion-result-v2`, `policy_key`, `scope_key`, and ordered `candidates` with exact keys `chunk_id`, `channel_ranks` (ordered objects with channel/rank), `rrf_score` (float.hex string, following existing graph candidate checksum convention), and `protected_exact`. Cache hydration must reauthorize identities from R2, not trust integer IDs alone. Raw reranker scores remain separate metadata.

### D4: frontier page and accounting

Use existing exact row types from `projection/records.py`, not untyped dictionaries or a truncated `CollectionGraphProjectionBundleV1` falsely claiming its full manifest. Minimal page declaration:

```python
@dataclass(frozen=True, slots=True)
class FrontierCursorV2:
    request_digest: str
    generation_key: str
    depth: int
    after_key: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class FrontierPageV2:
    nodes: tuple[ProjectedEntityV1, ...]
    memberships: tuple[AutomaticCanonicalMembershipV1, ...]
    relations: tuple[ProjectedPhysicalRelationV1, ...]
    relation_semantics: tuple[ProjectedRelationSemanticsV1, ...]
    documents: tuple[ProjectedDocumentMembershipV1, ...]
    chunks: tuple[ProjectedChunkMembershipV1, ...]
    relation_evidence: tuple[ProjectedRelationEvidenceV1, ...]
    mention_evidence: tuple[ProjectedEntityMentionEvidenceV1, ...]
    provenance: tuple[ProjectedArtifactProvenanceV1, ...]
    evidence_refs: tuple[GraphEvidenceRefV2, ...]
    next_cursor: FrontierCursorV2 | None
    scanned_rows: int
    complete: bool
```

Projection row records preserve their current validators; refs are derived only after validation against the captured ready generation and representation. `complete` means this requested page sequence has no next page, never corpus exhaustiveness. Exactly one of complete and a non-null next_cursor holds. Exact nonnegative scanned_rows counts physical rows examined including duplicates/overflow rows, not merely admitted rows. Cursor tokens are internal, bounded, and rejected if request_digest, generation, depth or authorization differs. Query ordering keys include relation priority, deterministic quality represented canonically, and opaque ties; `after_key` stores the exact lexical keyset used by the producer. No offset-pagination interpretation is allowed.

`FrontierBudgetV2` is mutable request-local accounting, not a wire DTO. Constructor fields: `turn_budget: TurnBudget`, `deadline: float`, `max_identities: int`, `max_edges: int`, `max_source_rows: int`, `max_queries: int`, `max_wire_bytes: int`, `clock: Callable[[],float]`. Counters initialize once: `identities`, `edges`, `source_rows`, `queries`, `wire_bytes` all zero; use one lock across groups. Exact methods `check_active() -> None`, `reserve_query() -> bool`, `charge_page(*, scanned_rows:int, wire_bytes:int) -> bool`, `admit_identity(key:str) -> bool`, `admit_edge(key:tuple[str,str,str]) -> bool`. Identity/edge admission deduplicates; physical reads/queries/wire bytes charge every time including later discarded work. Reserve query before I/O, refuse dispatch when exhausted, and count actual bounded wire bytes at transport rather than serialized in-process object size. Source body/token/pair charges remain with the original TurnBudget/SourceRuntime; this object does not replace or refund them. Caps come from captured effective configuration, at most existing 200 identities/1000 edges/depth2, never multiplied by group count. Relative transport timeout derives from the smaller original deadline and turn remaining time; no serialized monotonic timestamp.

### D5: canonical hash ownership and exact payloads

One convention for new V1/V2 digests: `json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('utf-8')`, then lowercase SHA-256. Enum values are strings; UUIDs use exact canonical lowercase hyphenated `str(UUID)`; null is JSON null; tuples become arrays. Reject duplicate document mappings, noncanonical input, NaN/infinity and bool-as-int. Existing hashes/receipts retain their established conventions and are not rehashed into a false equivalence.

I owns `representation_snapshot_digest(snapshot: RepresentationSnapshotV1) -> str` and `dense_generation_snapshot_digest(snapshot: RepresentationSnapshotV1, selections: tuple[DenseGenerationSelectionV1,...], *, space_key:str) -> str` in its stores/contracts. Representation payload: `version='representation-snapshot-v1'`, `selections` sorted by `(model_label,pkid,id)`; each selection has `document` with all five DocumentRef fields and the three remaining RepresentationSelection fields by their exact names. Dense payload: `version='dense-generation-snapshot-v1'`, `space_key`, `representation_digest`, `entries` in representation order. Every captured representation has one entry with full document reference, representation_key, and either `generation_key`/`manifest_sha256` both populated or both null. Ready mapping is the non-null subset and unique sorted generation_keys must agree exactly. Absent entries are explicit; no later pointer is selected. The snapshot's manifest_sha256 is this digest, not a best-effort cached summary.

G owns `graph_authority_group_checksum_v2`, `graph_selection_digest_v2`, and `graph_algorithm_signature_v2` beside its respective DTOs. Group hash payload keys: `version='graph-authority-group-v2'`, `authorization_context_signature`, `ontology_checksum`, `ready_bundle_checksum`, `representation_digest`, `selected_document_ids` sorted canonical UUIDs, `generation_keys` sorted opaque keys. These bind the original authorization and exact subset rather than a fabricated narrowed context. The selected-scope digest includes `version='graph-selection-v2'`, original authorization signature, full representation digest, canonically ordered group checksums, and coverage exact fields. The algorithm payload includes version, resolver policy/version, frontier/cap policy digest, canonically ordered `(group_checksum,relation_request_digest)` pairs, and selection digest. Group refs remain exact and current-authority checks are still mandatory.

R2's outer nine-field serializer remains exactly the lane contract. Its query_digest for a prepared query is the I-owned actual prepared-input digest; preparation key is included in embedding space identity. Original-query/proposal policy identities additionally enter ranking_policy/vocabulary binding or a versioned composite query digest owned by R, never an independently concatenated ambiguous string. Graph-disabled identities use explicit versioned disabled markers.

### D6: request-local receipt and sink verifier

Baseline `TurnContext` is mutable request-local data; `SynthesisLease` already stores the same budget, exact payload, authority, identities and finite synthesis deadline. `execution_context.execution_publication()` delegates to the current application execution owner's guard. Reuse these boundaries rather than encoding a second execution model.

Coordinator adds `TurnContext.answer_publication: AnswerPublicationContextV1 | None = None`. Library-owned protocol:

```python
class AnswerPublicationContextV1(Protocol):
    async def finalize(self, text: str, *, limited_reason: str | None = None) -> ValidatedAnswer: ...
    async def verify(self, answer: ValidatedAnswer, *, content: str,
                     sink: Literal['stream', 'refresh', 'persist']) -> None: ...
```

The app binds one implementation around final packet, bounded authorize callback, original TurnBudget, original execution owner/write epoch and the existing synthesis lease where applicable. `ValidatedAnswer` gets a private `_receipt: object` field (`repr=False, compare=False`) with no deserializer. Finalization registers a fresh opaque object in that exact context with content/evidence/scope digests and original identity-bound budget/execution/lease. Merely constructing the dataclass or copying a token to another turn cannot mint authority. Verification requires registry membership by object identity, exact byte hash/digests and unchanged binding, rechecks current authority with bounded work, and invokes original execution/lease checks. Failure raises `AnswerPublicationRejected` with a closed reason (`unbound_receipt`, `content_changed`, `evidence_changed`, `scope_changed`, `authority_unavailable`, `execution_inactive`, `lease_expired`); it does not reopen retrieval or mint another lease.

The receipt's registry is request-only and retained through all final sinks, then cleared on completion/cancellation. All fields remain excluded from SDK/model/tool/frontend/history serialization. Source inventory/image markup are part of `content` before finalize. The actual persistence write also stays under existing `execution_publication()` and transcript transaction fences; the verifier is not permission to bypass them. Each stream/refresh/persist uses the same verification API, so no local sink approximates the predicate. If authority/evidence cannot be established while the execution is still active, finalization may register only the exact fact-free limited template with empty source payloads; if execution/lease is closed, publish nothing. A historical message is not retroactively given a receipt.


### D7: extraction and chunk digest boundaries

`SourceChunkV1.source_sha256` is the lowercase SHA-256 of `source_text.encode('utf-8')`, where `source_text == extraction.canonical_text[start:end]`. It binds the exact chunk slice, including Unicode and newlines. `ExtractionV1.canonical_sha256` binds the complete canonical text. `SourceChunkV1.input_sha256` remains the projection's `input_digest(embedding_text)`; profile-prepared wire input has its own receipt digest and is never substituted for either source digest.

I2 owns strict `elements_to_json`/`elements_from_json`, `extraction_manifest`, and extraction revision binding. The extraction manifest uses version `source-extraction-v1`, `parser_key`, the artifact's full fields, `canonical_sha256`, `ordered_elements`, `quality`, and `issue_codes`, serialized using D5. `chunker_key_for(tokenizer, max_tokens, overlap_tokens)` binds the exact local tokenizer adapter revision, both budgets, and projection policy; supplied mismatching policy keys are rejected.

I2 supplies `persist_extraction_revision(occurrence, extraction, *, using: str) -> int` and `bind_payload_extraction(payload, *, using: str) -> int` in ingestion services. The binder retains the verified source occurrence and assigns `payload.extraction_revision_id`; it verifies persisted artifact/occurrence identity, source digest, canonical spans, manifest and collision fields. I3 wires this binder before durable manifest registration. Figure outputs retain the parent's extraction revision and separate media artifact bytes. Existing formats without structural extraction are explicitly partial with `structure_unavailable`, never fabricated structure.

### D6 clarification: default retrieval publication

Valid default-mode direct, manual and document-tool answers receive the same app-owned publication binding around their current selected packet, exact user/scope/source, and existing R1 TurnContext. This does not activate source/window feature routing or planning. A private SourceRuntime carrier retains the original authority, snapshot and budget without binding the public source-mode context. Historical/unbound evidence cannot mint a receipt. Bounded metadata checks may compute SHA-256 in PostgreSQL without returning source bodies; a current hash alone never proves that arbitrary delivered text came from that source. Delivered slices and media require captured exact provenance.

### D7 clarification: bounded nonmonotone token fitting

Binary prefix/suffix probing may find a checked valid fit, but cannot prove no fit exists for a BPE tokenizer. I2 adds exhaustive fallback only when the heuristic finds no nonempty fit. Across one chunk_extraction call, fallback is bounded to 4096 tokenizer calls and 1,000,000 full-input codepoints (context plus slice). These are offline chunker-work limits, not retrieval allowances. Exhaustion raises SourceContractError('chunker_work_limit'); only a completed no-fit search may report unrepresentable_source_element. Fitting policy and both limits are part of the chunker fingerprint. The function returns no partial tuple on failure. I5 must retain immutable input and mark the staged generation failed/limited without activating partial chunks; a capacity failure does not mean source text is impossible to represent.

### D6 clarification: active messages after database reload

The active context tracks its issued message UUIDs and write epoch across database reloads, which discard private receipt attributes. A newly generated active-epoch message without its matching receipt is rejected; it cannot silently become an ordinary historical message. Delta refresh verifies exact serialized assistant content immediately before send using the shared verifier. Existing historical messages are not given retroactive receipts.

### D7 clarification: changed parser policies

The corrected DOCX traversal and XLSX unsupported-chart classification use `native-docx-ordered-v2` and `native-xlsx-ordered-v2`. Changed canonical content or quality cannot reuse an older immutable extraction identity. No persisted revision is overwritten.

### D5 clarification: source and dense revalidation APIs

I4 owns `validate_representation_snapshot(snapshot: RepresentationSnapshotV1, *, using: str) -> RepresentationSnapshotV1`. It returns the same input object after re-reading all concrete owners for each original UUID and matching exact model/PK/UUID/collection/source hash and canonical text digest. Invalid, ambiguous or changed source raises `SourceScopeUnavailable('source_scope_unavailable')`; it never freezes replacement selections.

I4 owns `validate_dense_snapshot(snapshot: RepresentationSnapshotV1, dense_snapshot: DenseGenerationSnapshotV1, *, using: str) -> tuple[UUID, ...]`. It revalidates original source scope and canonical digest/mapping, then returns only captured generation keys whose current active pointer, ready state, source, profile and manifest still match. Pointer loss/swap removes a captured key and can yield an empty tuple (lexical-only); invalid source or mapping raises the closed source-scope error. It never selects a newer generation.

R2/C2 and later representation/graph consumers must retain current authorization and call the same applicable guards immediately before body/cache/publication. Rank-time checks also verify frozen membership, receipt compatibility and current chunk content digest, then recheck pointer/authorization before returning. Validation does not grant authority or create a new budget/scope.


### D5 clarification: strict serving binding and query cache hydration

I4 adds `lib.embeddings.document_binding` with:
- `validate_query_observation(observation: QueryEmbeddingObservationV1, *, space: VerifiedEmbeddingSpaceV1) -> str | None`: returns the validated compatibility key; None only for actual unknown observed model. Malformed, unbound or conflicting envelopes raise `EmbeddingContractError('serving_binding_mismatch')` or the existing receipt contract error.
- `serialize_query_observation(observation, *, space) -> dict`: validates and emits exact internal cache payload keys `version='query-embedding-observation-v1'`, `vector` (array), detached `receipt`, `prepared_input_sha256`, `compatibility_key`, and `serving_binding`.
- `restore_query_observation(payload: dict, *, space) -> QueryEmbeddingObservationV1`: validates exact payload keys and binding/receipt/vector/profile, returning the exact base DTO with its private immutable envelope. Ordinary dataclass reconstruction is unbound and fails closed.

Separate immutable `ServingBindingV1` has exact fields `space_key`, `verification_sha256` (canonical registered endpoint/deployment/engine evidence), `prepared_input_sha256`, `vector_sha256`, `role`, `modality`, `media_sha256` (null for query). The public query DTO and existing v1 receipt schema remain unchanged; the envelope is attached outside dataclass fields. I4 adds separate `ChunkEmbedding.serving_binding` JSON in unshipped documents0009 and revalidates it at staging, activation and ranking. Receipt model echo does not attest engine precision, quantization, pooling or templates; absent observed revision remains null.

R2 cache hydration uses these helpers, never reconstructs a compatible DTO by copying public fields or a config model name. The internal payload is not a public authority input. Reject extra keys, noncanonical/bool/NaN vectors, conflicting compatibility key, incorrect role/query modality/media and changed verification evidence. Serialization alone grants no document authorization; original source/snapshot/current-authority checks still apply. I4 tests same-model incompatible engine binding, positive roundtrip, unbound generic reconstruction and payload relabeling.


### D6 clarification: classify every new final answer before emission

A1 final classification applies to every new assistant answer, including a new ordinary UserMessage after document history and clean-history turns. A provider can emit document markup late in any generated answer; history-based buffering alone cannot guarantee zero unbound output. Defer all provider final-answer callbacks until complete document-markup classification and publication validation through the existing final_stream path. Uncited ordinary content remains receipt-free and receives one nonempty final callback under both LLM_STREAM_FINAL_ANSWER_ONLY settings. Tool/progress events and non-answer behavior remain unchanged.

This deliberately removes incremental ordinary-answer token streaming when `LLM_STREAM_FINAL_ANSWER_ONLY=0`; default final-only behavior is unchanged. No new context, scope, retrieval ledger or history reauthorization is introduced. Tests cover uncited ordinary positives, clean-history unbound document output, historical UserMessage output under both stream settings, refresh and PostgreSQL persistence. The tradeoff is final-answer latency to protect exact final bytes and current source authority.

### D8: V2 endpoint fanout admission

Preserve numerical fanout/admission ceilings and unchanged V1 behavior. In V2, apply query-weighted endpoint reduction before bounded destination fanout selection. Suppressed predicate labels on one directed pair must not consume separate fanout slots and evict another endpoint; otherwise adding irrelevant labels can increase the first endpoint's normalized mass. Preserve closed snapshot validation, confidence/support/utility/direction factors, total topology work limits and the original deadline. This is an explicit V2 ranking treatment and belongs in its algorithm signature; it creates no additional database/materialization allowance. Test the closed graph at and beyond the actual fanout cap. Its answer-quality and activation gates remain separate.

### D9: Provider memory receipt coverage

A vector-style receipt cannot substantiate deletion of graph-side writes. Any graph add attempt therefore makes provider receipt coverage incomplete, including a graph failure followed by a vector fallback. Retain and reconcile all validated known vector IDs, but never mark external reconciliation complete while graph or other provider copies remain unidentified. An additive optional private coverage observer may carry this attempt state without changing the existing result observer's one-argument contract. Actual graph erasure requires a verified provider receipt/delete contract; no such contract or real-provider erasure is claimed by synthetic tests.

### D10: Representation activation lock order

G5 preparation runs inside the caller's outer atomic transaction and acquires the existing graph-aware lifecycle phases: collection scopes/rows/graph rows, then document scope/graph rows/exact source row. I5 must call it before entering any leaf-only document lock context, validate the source and expected old representation after lock acquisition, and finish representation activation in that same transaction. Preserve complete affected-collection discovery/rechecks and canonical lock order. Never acquire graph artifact/run locks after a leaf-only document row is held. All parsing, embedding and pure build work is staged before this short activation transaction; queue publication remains on commit. This clarifies the original under-lifecycle-lock wording without changing the public helper signature or granting another budget.

### D11: Graph representation identity and schema rollback

Representation input identity is separate from resolver policy identity. Document and collection build identities may carry an optional representation digest. Absent bindings retain the exact existing V1 serialization and hash; present bindings use a distinct V2 namespace. Do not put a document representation digest in `resolution_config_checksum`: that field describes resolver policy and participates in compatible collection grouping.

Graph artifacts and collection projections capture a nullable representation snapshot together with its nullable digest. Both must be absent for legacy records or present together with a canonical lowercase SHA-256 digest. Runtime validation checks the exact versioned payload, document ownership, source, membership and artifact association; a digest alone grants no authority. Captured values are immutable, and a new input requires a replacement generation. Build runs remain bound through their artifact association and build key without a parallel identity registry.

The additive migration refuses reversal while representation-bound rows exist, since dropping their bindings could make V2 records appear to be legacy records. Legacy-only reversal remains supported. Operational rollback retains the additive fields and uses the existing activation gates; it does not erase captured provenance. The cost is additional identity plumbing and retained schema during rollback, preserving reproducibility and preventing silent artifact reuse.

### D12: Query serving-space and shared search lifetime

`DOCUMENT_EMBEDDING_SPACE_KEY` selects one explicitly registered, verified document embedding space for the query route. It is not a model-name alias or evidence that historical vectors are compatible. Missing, malformed, unknown or unverified configuration leaves dense retrieval unavailable and preserves authorized lexical retrieval. Resolve the canonical key and registry record on the captured database alias within the original deadline; endpoint binding and response verification still apply. This decision does not activate a deployment setting. I6/P3 own the operator setup and coverage documentation.

Default direct/tool retrieval uses the existing selected-scope authorization builder independently of graph, preservation and selection flags. The prior production builder gated this same permission policy on graph enablement. Preserve supplied-context handling and bound authorization queries by the original TurnContext deadline. R2 may bind a private SourceRuntime carrier around text_chunk_search using those same objects, or reuse an existing runtime. It must not create a second budget/controller, turn on preservation flags, or derive authority from a list of document IDs. A turn captures one original scope and representation/dense mapping across query proposals. A changed authorization signature cannot allocate a replacement scope; explicitly allowed narrowing derives only from captured original membership. Shared holder initialization and snapshots must remain safe under concurrent tool threads. Standalone callers without the original budget and authority remain lexical-only.

The actual ranking result cache lives in chunk_rerank_local_vllm.py, so R2 may add narrow get/set forwarding into its private scoped adapter. Preserve existing APIs and behavior for non-search callers, current-scope checks on cache read/write, and the original reranker failure latch, state.publish fence and deadline. The cost is explicit configuration and a request-local compatibility adapter; neither grants new scope or additional work allowance.

Explicit query narrowing preserves the original representation and dense snapshots. The dense ranker may accept an optional strict frozenset of original document UUIDs and filter eligible generation keys before distance ordering and the existing limit. Filtering after top-k would let excluded documents consume every slot. Lexical retrieval, materialization and ranking-cache identity use that same subset. Widening beyond initially captured membership fails closed. This optional argument preserves existing callers and cannot select a newer pointer or allocate a new budget.

A default single-document tool may supply a freshly built narrower authorization context. Accept it only after current revalidation, with the same principal reference, alias and permission-policy identity, and exact requested membership contained in the original captured scope. Continue using and revalidating the original authority, snapshot, cache authorization digest and budget. A different signature never creates a replacement holder; changed policy/principal/alias, malformed signature, revocation or widened membership fails closed.

### D13: Durable representation rebuild publication

The existing graph build publisher uses the default database. The new representation activation seam must reject unsupported non-default aliases with a closed source-scope error before any locks or mutation; scheduling a callback on another connection does not make implicit-default ORM calls alias-safe. Do not add global router overrides.

Publish the durable rebuild intent only on successful commit. Persist the exact new representation generation as a strict versioned guard in the existing requested_documents JSON, alongside exact source/document ownership. Every durable resume/retry must enforce this guard. A callback closure alone is insufficient because a broker failure would lose its captured expectation. The enqueue helper may add an optional expected guard, but must compare it with the SAME captured context used to derive the emitted build key. Never recapture a newer representation between checking and key generation. Malformed guards fail closed; absent legacy records keep exact existing behavior/hashes. Existing requested build keys fence swaps after dispatch. The optional guard is representation_binding={version: graph-document-representation-v1, generation_key: canonical nonnull UUID}; the surrounding exact document/source fields remain its owner. Guarded requests must not create an automatic resnapshot successor that drops this binding. A changed target requires a new explicit intent. No new schema or activation flag is needed.

Required tests cover commit/rollback, durable publication failure, stale callbacks and resumed failed publication after a same-source representation replacement, plus alias rejection before mutations. Costs: the new seam supports the existing default-only publisher and rejects other aliases; retries preserve their original target and may require an explicit new rebuild request instead of silently upgrading scope.

### D14: Durable ingestion schema and bounded omissions

The ingestion output ledger uses immutable source, extraction, parent, output-key and payload identities with fenced mutable lease/result transitions. A successful result requires the complete exact document reference; registration failures use bounded typed reasons and do not imply parsing or indexing success. Parent links must preserve same-item ownership and reject cycles while permitting deletion of a batch or collection containing both parent and figure outputs; use cascade-safe restriction rather than unconditional parent protection.

The additive ingestion migration refuses reversal after output-ledger use. Unused or legacy-only reversal remains possible; operational rollback retains committed output provenance and resumability. Existing terminal items without a new manifest keep their historical status/document IDs, subject to exact current authorization, without invented source bindings or automatic reingestion. Explicit reset remains source-fenced, and a foreign content-hash collision remains an error.

Archive extraction records empty, unsupported or failed members explicitly within the existing member, byte and depth bounds. When a ceiling prevents further enumeration, record one bounded aggregate omission instead of silently stopping or creating unlimited failure rows. Preserve exact raw archive names, ordinals, byte identities and security checks. The cost is explicit partial/error outcomes and retained additive history; no success or complete coverage is inferred for omitted members.


The claimed-source freeze cannot depend on a resettable attempt counter. Item attempts remain monotonic under the supported claim/recovery transitions, completed manifests independently freeze source identity, and stale parser generations cannot reuse a prior fence. Figure publication locks its derived destination and rejects reuse when direct child grants could widen the source boundary, preserving existing grants and content. Historical figures remain available only through their exact typed parent, expected child relationship and current source/child authorization. Empty nested archives receive member-bound failures just like empty leaves.

### D15: Bounded canonical frontier access and transport

A V2 graph invocation shares one FrontierBudgetV2 across every group, generation and round. Its cumulative ceilings are 5,000 examined source records, 64 datastore queries and 1,048,576 actual topology-gateway HTTP response-body bytes. Stricter operator/request limits win. These are additional restrictions; V1's per-read envelopes did not establish a global query budget. The existing 200 identities, 1,000 edges, depth two and original branch/turn deadlines remain. Manifest, index, provenance, neighborhood and sentinel reads count. Duplicates and invalid or late work are charged. This byte budget does not claim to meter hidden Bolt traffic.

Ordered access uses an additive FrontierIndexV2 equality index and immutable content-addressed directory/successor records under the existing collection-generation authority. New optional generation attestations bind the derived index to the validated projection. Ordinary staged rebuilds populate and independently verify it; this task does not mutate or backfill existing ready generations. Missing index support leaves V2 unavailable while legacy readers remain usable. Partially present or corrupt attestation fails closed. Runtime acquisition cannot sort an unbounded neighborhood or claim global quality order after sorting only an arbitrary read window. Real Memgraph plans and physical-hit observations must demonstrate the bound, and the cross-generation bridge must succeed within the actual 64-query ceiling.

Frontier-index reads use a two-row duplicate boundary. Existing raw records share generation/opaque keys across families, so raw hydration permits the nine known families plus one overflow sentinel, charges every row and validates exact family/digest closure. An inaccessible relation cannot become a transit bridge. The new gateway route uses a relative timeout derived from the original caller deadline; ingress, body reading and queue time consume it. Actual received response bytes are counted before decoding, including bounded invalid/error responses. Unknown remote work after transport failure prevents continuation, and worker occupancy lasts until actual completion.

The existing projected snapshot type accepts one additional complete algorithm tag vector: canonical_frontier_v2, frontier_transition_v2, frontier_evidence_v2 and frontier_seed_v2. Mixed versions are rejected. V1 validation, hashes and legacy consumer behavior remain exact. The V2 signature binds canonical configuration, relation request, captured group checksum, cumulative caps and acquisition/coverage policy. The group checksum is an identity binding, not an authorization capability. A narrow V2 ranking validator checks those inputs and seed/cap closure. Empty V2 snapshots have no seeds, relations, mentions or audits and preserve other provenance checks; their caller reports truthful coverage and skips ranking. No dummy identity is introduced.

The cost is an additional derived access structure, build/validation work and conservative partial results when the shared budget cannot cover every group. Quality and operational gates must measure that tradeoff before activation; a bounded implementation alone proves no recall improvement.

### D16: Lexical projection schema and publication

Allocate documents migration 0011_chunk_search_projection_v1 after 0010_embedding_generation_hnsw_v1, followed by 0012_chunk_search_projection_gin_v1. TextChunk gains nullable search_vector and search_vector_binding, source_kind defaulting to unknown, and section_path defaulting to an empty list. Allowed source kinds are unknown, body, heading, table, formula, figure and bibliography. Legacy structural identity remains unknown rather than inferred from titles or regular expressions. The concurrent GIN index is textchunk_search_vector_gin_v1; installation verifies its actual table, column, method, opclass and valid/ready state and safely handles interrupted invalid-index attempts. No blanket data backfill runs in the migration.

A closed versioned binding covers exact document ownership, source/canonical/revision/representation identities, actual title, body, section path, source kind and projection policy fts-english-titleA-sectionB-bodyD-v1. Non-null vectors alone do not establish readiness. The publication helper validates coordinates and writes vector/binding together under the caller's existing source/generation transaction. Legacy chunk publication invokes it after chunk creation; I5 uses the same helper for staged structural chunks before pointer activation. Title, source, body, section or policy changes invalidate FTS eligibility until a bounded fenced refresh. The original R2 scope, subset, authorization, deadline and final handoff remain mandatory.

Backfill defaults to at most 100 chunk IDs in one batch, with an explicit cursor and strict finite limits. Resolve owners only after selecting the bounded chunk page. FTS eligibility requires a valid index and current coverage of its declared selected modalities; image/caption exclusions cannot be hidden by counting text rows alone. Incomplete FTS retains other authorized lexical channels. Final materialization rechecks bindings, and caches include the ranking/projection identity. Only already-authorized graph ranks may enter fusion; the existing V1 contract remains unchanged.

Deterministic exact matching and pre-limit coverage ordering are the default repair. FTS, reciprocal-rank fusion and protected exact slots remain separately disabled treatments; protected slots require fusion. Failure paths preserve the selected candidate order and unavailable confidence. The fixed disposable query-plan experiment uses 20 authorized documents with 500 chunks each plus 500 unauthorized chunks, one-document and 20-document selected scopes, and fixed rare/common/absent queries. It records actual EXPLAIN ANALYZE BUFFERS without forcing the planner or implying production latency.

Schema reversal of 0011 refuses used projections or structural metadata before dropping fields; unused/legacy-only reversal is allowed. Index-only reversal can disable FTS without erasing provenance. Operational rollback retains the additive schema and history. The cost is stored projection metadata, bounded refresh work, and explicit lexical fallback while coverage is incomplete.


### D14 legacy result identity clarification

Display names are mutable and do not establish source identity. Legacy figure status retains exact historical parent and figure IDs after a parent-title or child-collection rename, provided the uniquely resolved typed parent is among the authorized historical item IDs, belongs to the current source collection, and the figure collection is its current direct child. Current access is required to both. Ambiguous, deleted, revoked, wrong-parent, or incoherently moved references remain unavailable. Historical rows do not acquire invented original child IDs or extraction provenance.

### D15 captured representation mapping and unavailable frontier

The private frontier factory receives the original `SelectedGraphScopeV2` and identifier codec. It derives exact graph-generation/document-key bindings to the captured representation generation, including explicit legacy nulls, from that selection. Missing or ambiguous mappings fail closed. Each requested group must match its captured member and recomputed checksum; callers do not supply a dictionary as authority or select a newer snapshot.

A successful `BoundedGraphTopologyV2` always preserves complete artifact-provenance scope closure. When missing index data or an exhausted budget prevents constructing a closed snapshot, the loader produces a typed unavailable/budget failure with truthful coverage and consumed work. G3 retains that accounting and returns no candidates for the unavailable group. Corrupted authority cannot trigger a legacy retry. An empty V2 success is valid only when required provenance was actually acquired and its identities, seeds, relations, mentions, and audits are empty.


### D15 production index and ranking context

Ordinary projection workers request the additive frontier index when writing new staged generations. Construction uses the existing staging timeout, heartbeat, validation, and ready publication lifecycle. A partial or invalid index cannot publish ready. Existing ready generations remain untouched, and V2 retrieval remains disabled by default. Explicit legacy repository callers can continue omitting the optional index.

V2 ranking receives the loader's exact immutable frontier context, including the captured group checksum, effective cumulative caps, and acquisition policy. Missing or altered context is rejected; stricter operator caps cannot be replaced by hard maxima. The context binds algorithm identity, while the original selected group and current source/ready authorization still establish authority. V1 ranking calls and signatures remain unchanged.

### D17 independent calibration and activation trust

Independent review enrollment uses a versioned administrator-controlled manifest at `/etc/aquillm/evidence-review-trust.json`, separate from evaluated source revisions. Candidate reports and CLI arguments cannot select or populate the trust source. Missing, malformed, or untrusted enrollment keeps policies and activation disabled. The bounded reader validates the exact schema and canonical digests, rejects ambiguous entries, and observes rotation and revocation. Deployment requires administrator-controlled, read-only ownership. This program does not install production trust.

Reusable support-policy calibration and current-run activation have separate artifact pins. Calibration binds the policy, independent reviewer provenance, predeclared numeric criteria, reviewed subjects, runtime, scope, corpus/splits, uncertainty, and expiry. Activation separately binds the exact current report revision and subject set, its independent human reviews, and the accepted calibration-policy digest. This permits enrollment without changing the code revision being evaluated. Calibration does not certify each future answer; trusted human provenance does not mathematically prove a judgment true.

Evaluation captures the original question, actual delivered evidence, support decision, final outbound message, and exact persisted message under the same conversation and message identity. Missing or mismatched observations block eligibility. Synthetic cases and injected trust fixtures test this plumbing only. Missing independent human or live evidence remains pending or inconclusive, with optional policies disabled.

### D18 graph-adoption applicability

Independent source or window comparisons may keep the same recognized legacy graph configuration. They do not require adoption evidence for a graph treatment they did not change. Capture the actual graph policy, resolver version/checksum, algorithm signature and effective runtime identity in each arm's comparison controls, bind them in the configuration snapshot and review subject, and require exact agreement across the arms before declaring the additional graph gate inapplicable. The legacy signature comes from the production graph algorithm configuration; the word `legacy` is not a signature.

Unknown, missing, malformed or inconsistent observations block eligibility. A changed graph identity, configuration or signature, and any contextual identity, shadow or V2 graph treatment, requires the bound graph-quality attachment. Its source, runtime, policy and per-query identities must match the evaluated candidate. The original graph-lane provenance, answer-quality, support and D17 review requirements still apply in both cases. No caller-controlled opt-out or second activation authority is introduced. The cost is additional observation plumbing; a mistaken classification could waive a necessary graph gate, so unchanged and changed configurations require explicit positive and negative tests.

### D15 evidence selector handoff clarification

The private query-aware selection envelope carries an immutable, unique, sorted mapping from captured authority-group checksum to its ready-bundle checksum, populated only from the original selected groups. Evidence references do not contain enough information to reconstruct that binding. Reject missing, extra, repeated or substituted group entries and revalidate final candidates against the original selection; never infer a checksum or select a newer ready bundle. This corrects the selector interface without changing public wire contracts or granting authority.

### D16 channel withdrawal and fallback clarification

Preserve each admitted graph candidate's original one-based channel rank after graph caps remove other rows. Compressing surviving ranks would change reciprocal-rank contributions. At each existing materialization and final authorization boundary, withdraw an expired FTS contribution even when another valid channel retains that chunk. Report current FTS coverage and the actual surviving fusion inputs. A complete provider ranking keeps its still-valid selected rows and order; unavailable or unknown provider status uses the surviving originally admitted channels to recompute deterministic fallback. Status is private per-call metadata, never inferred from row order. These checks keep the original scope, database timeout, candidate ceilings and turn budget; no later provider retry or fresh scope is allowed.

### D19 retained representation schema and activation

Allocate documents migrations 0013 and 0014 after the reviewed lexical projection migrations. The representation state binds the exact concrete owner model, ContentType, primary key and UUID; ContentType supports Django deletion collection and grants no retrieval authority. Immutable generations and legacy source snapshots retain historical chunks and source/media identity. Deferred database guards permit actual owner deletion while rejecting standalone history deletion with the owner retained. History counts against the existing bounded deletion limit.

Damaged legacy offsets do not block an independently valid replacement. Capture canonical validity separately from immutable per-chunk context and artifact support. Preserve the old content, identity and offsets; return explicit unavailable context when its historical source cannot be proved. Never highlight those offsets into current text or substitute a newer PDF/image. Strict new-generation manifests still require exact slices. Used history prevents destructive schema reversal.

Activation acquires graph lifecycle locks before source, state, generations and chunks. A private one-use transition binds the predecessor, target and transaction; its finish step validates the persisted active target and publishes a rebuild intent using the actual post-transition canonical hash. Canonical text, lexical projection, pointer and durable intent commit together. Immutable input revision remains distinct from current canonical hash. Storage and provider work remain outside these locks.

Historical media uses a chunk-qualified URL and current exact permission. New answer publication additionally requires the original selected generation, already charged verified bytes and the existing sealed receipt. A retired source may remain available as a historical citation without becoming fresh evidence for an in-flight answer. The cost of these changes is retained storage and additional database constraints; real migration, deletion, concurrency, API and publication tests are required before release.

### D15 query scope, joint graph channel and cumulative seed work

Capture any requested document subset before seed, topology and evidence admission, while retaining the original full authorization and representation snapshot. An empty requested subset performs no graph work. Revalidation compares that same captured subset; consumers do not reselect a newer authority. Body reads use the original source runtime and representation-aware cache identities.

V2's joint direct/extended handoff contributes one `graph_v2` ranking channel and one original-rank RRF vote. It cannot coexist with nonempty legacy direct/extended channels in the same fusion. Its rank and count cap stays 20 and the total candidate ceiling stays unchanged. The legacy path retains its existing ordering and identity when V2 is absent. Early graph dispatch recognizes the legacy mode explicitly; V2 waits for its frozen baseline and runs once.

Apply the original raw-seed prefix before partitioning incompatible groups and enforce one final seed cap across all groups. Local PPR normalization is rescaled by the group's admitted global seed mass before evidence selection. The policy signature records this rule. Seed database reads and frontier discovery share one cumulative physical-work ledger and original deadline. Explicit known query cardinality and sentinels reserve allowance before dispatch; actual returned records are charged even when late or invalid. These source-row counters do not claim to measure internal database plan tuple scans. Uncertain completion blocks further work and adoption evidence. The cost is truthful partial coverage under pressure; there is no new per-group allowance.

### D18 graph study originals and mode observations

Retain and validate the original effective controls for both graph arms, including exact value types, production bounds and cross-field constraints. Recompute their identities and budget bindings. Treatment declarations name exactly the allowed changed values; they cannot excuse model, extractor, cap, deadline or iteration drift. Every execution must fit its own captured effective limits, including stricter operator values.

A labeled question remains one statistical observation while retaining separate paired executions for baseline, selection, preservation and combined answer modes. Run all quality, uncertainty, safety, completion and latency gates independently for each mode and require all four to pass. A 150-question study therefore needs 600 mode pairs, not 600 independent questions. Each actual invocation has a unique identity; copying one event into four modes is invalid. Current answer-evaluation subjects bind their exact mode, query and execution digest.

The graph event additionally captures the actual request algorithm signature. It is distinct from the static effective configuration signature and binds the kernels and grouped acquisition used by that request. Missing or mismatched observations remain inconclusive. The independent D17 trust source still governs activation; synthetic passing fixtures establish only validation behavior.

### D15 bounded seed reauthorization and allocation

The built-in permission policy reauthorizes the original selected document IDs through a bounded permission query and one union across the supported concrete document models. Preserve duplicate rows so ambiguous identities fail validation. Reserve each query's known returned-row ceiling, including its sentinel, against the same seed/frontier ledger before dispatch. Unsupported policies without the bounded capability return unavailable without invoking an unbounded callback. No fresh authorization or representation context is captured.

The accounting scope covers seed data, seed reauthorization and frontier reads. Database control statements, ready capture and final body validation retain their separate existing bounded contracts; these counters do not claim to measure every database operation. Context-local propagation is limited to the original seed invocation and cleared on exit. A large scope that cannot fit the remaining allowance is unavailable rather than silently narrowed.

Within the unchanged original raw-seed prefix, admit one best seed per nonempty authority group before additional seeds, up to the one global cap. Omitted groups retain partial coverage. The request identity includes this allocation policy and the actual per-group ranking configurations.

### D19 deletion bounds and primary text partitions

Disable Django fast deletion for retained representation history and its cascading chunk/embedding children inside the bounded lifecycle collector. Count distinct collected rows against the unchanged global deletion limit before mutation, including history that the ordinary fast path would omit. Deferred guards allow a real owner deletion in the same transaction while refusing standalone history deletion.

Figure elements may overlay the primary canonical text partition. Lexical section publication selects the single covering non-figure element for a text chunk, while validating and retaining the complete extraction and its figure bindings. An overlay must neither make valid text ambiguous nor grant authority to image bytes.

### D15 selected permission reads outside seed metering

Use the built-in selected-document capability at initial authorization capture and every later revalidation, passing the already normalized or frozen selected IDs. Bound returned rows before materialization with the permission query and selected-ID union sentinel. Existing principal, capability, policy-signature and intersection checks remain mandatory. Unrelated documents or malformed identities elsewhere in a collection no longer exhaust or invalidate the selected scope; duplicate or conflicting selected identities still fail. This intentionally improves selected-scope availability without changing wire identities or ranking.

Only seed calls charge the original frontier ledger. Other source/final checks retain their own deadlines and selected-ID cardinality bounds. A legacy custom policy without the optional selected capability retains its existing unmetered behavior; a required metered call refuses unsupported capability before dispatch. No new scope or budget is created.

### D19 content changes retain citations

Content-change and chunk-replacement cleanup invalidates graph state while retaining published representation and captured legacy chunks. Actual owner deletion still uses the bounded exact-owner cascade. Capture legacy source history before the owner edit; a delayed cleanup callback cannot associate old slices with today's text merely because their offsets fit. Unprovable history remains explicitly unavailable. The private rebuild entry validates the exact live owner, current hash and captured predecessor pointer independently of stale retrieval eligibility, then requires the new persisted active selection at transition completion. This lets an ordinary edit invalidate old evidence immediately while a valid replacement can still be built.

### D19 one current representation scope across tools and publication

The first retrieval or source-tool operation captures all originally selected document representations once, under the existing source runtime lock. Bounded metadata reads use the original selected IDs; missing or ambiguous owners fail validation. Individual operations narrow their requested IDs while retaining that same captured snapshot. Later operations and final publication cannot capture a newer generation. This supports multiple selected documents regardless of tool order while preventing a generation change from refreshing an in-flight answer's authority.

Apply the active-generation predicate before source metadata, cache and body reads, including whole-document, adjacent, figure and default-mode paths. Historical citation endpoints remain the deliberate retained-source lookup. Final answer authorization validates the retained snapshot alongside existing owner locks, permissions, body fingerprints and publication receipts. The snapshot digest and original object identity bind the publication scope. A generation change can therefore produce a limited answer even if the retired chunk's text is unchanged. No new source body is acquired at publication.

A graph build in legacy mode must reject an owner with generation history it cannot bind. Generation-aware builds use the existing V2 gate and durable captured target. This may leave graph retrieval unavailable until a compatible rebuild; it must not enumerate retained chunks as current evidence.

### D19 default and explicit manual acquisition

Default text/media retrieval uses the original private carrier and never captures a replacement scope at publication. An unavailable original selection uses a distinct unavailable-parent value, not an authorized SourceRuntime. Ordinary chat can continue, while automatic source acquisition remains unavailable. Only an actual parsed human `/search` request may create an exact currently VIEW-authorized document child under the original turn budget. Its parent selection is unchanged, and automatic tools cannot reuse that child as expanded authority. Final sinks accept only the privately registered packet/context binding from the same active turn.

Empty or invalid manual evidence retains an ordinary fact-free limited publication context. It receives no manual child capability. Nonnull packet bindings still require the exact original authority and snapshot; malformed bindings are not suppressed as empty evidence.

Auxiliary figures must have both current immutable media provenance and exact selected packet membership. A mutable URL, a related parent or permission alone is insufficient. Omit unsupported auxiliary media with the existing omission notice while preserving usable main text; do not acquire bytes already known ineligible for handoff. Selected standalone figures retain their verified bytes, original budget and final permission/generation checks. Image request position, title and citation must match the actual selected evidence row, including text rows interleaved between images.

Dense preflight freezes scalar generation metadata and validates a unique matching manifest entry. Variable receipt, serving-binding and matched-entry text is charged before payload hydration; the payload query repeats the identity, cardinality, length, digest and eligibility checks. Native schema-fixed 1024-dimensional numeric vectors remain under the existing 45-candidate ceiling. Neither full generation manifests nor hypothetical vector-to-text serialization can substitute for the actual bounded-read contract.

### D15 per-provenance graph evidence bindings

A physical chunk can contain multiple relations and unrelated mentions. Its aggregate entity and predicate sets do not identify a claim. Preserve the actual private association between each admitted provenance item and its mention identity or directed relation endpoints. Every affected mention and transition allowance is charged when that chunk is admitted, with one physical candidate and maximum credit retained. Conflict reservation requires extracted opposing classes on the same actual claim within the same captured authority group. Unrelated mentions cannot change a transition allowance or manufacture opposition. Missing, conflicting or oversized bindings are unavailable; do not infer a Cartesian product from aggregate sets.

### D20 optional vocabulary and query proposals

Vocabulary drafts use checked revisions and publish immutable versions with attributable bounded entries. Attachments pin a particular version explicitly; publishing does not repin them. Draft, publish and attachment changes require current management permission. Selected descendants inherit only through explicit authorized ancestor attachments. Capture exact attachment paths, pins, contents and provenance once, then invalidate frozen use on additions, repins, removal or lost permission. Collection deletion may remove its owned vocabulary history in one bounded cascade; standalone published-history deletion is refused while its owner remains. User deletion may clear the actor reference without changing immutable publication attribution.

Optional vocabulary expansion and one model rewrite share the original three-action acquisition budget and existing planner limits. The original question remains the answer, scoring and publication question. Validate captured vocabulary and resolved source constraints before each optional action; no new scope, retry budget or authority is created. Both treatments are disabled by default. Fresh grammatical pronouns remain ordinary searchable questions, while an ambiguous source reference requires clarification if no validated antecedent is available; never inject the first retrieved title as an assumed referent.

The graph vocabulary treatment is separately disabled by default and supplies negative merge vetoes only. A vocabulary alias cannot authorize a merge. Vetoes remain local to their captured source memberships, including transitive endpoint checks. Actual resolver, artifact and build identity includes the captured vocabulary policy; current attachment or access changes make old graph artifacts ineligible until rebuilt. Bound capture and validation work globally; overflow must not silently drop a veto. Evaluation records the actual scoped policy, so a vocabulary-enabled graph cannot claim the unchanged-legacy exception. Missing independent treatment evidence leaves activation inconclusive and off.
