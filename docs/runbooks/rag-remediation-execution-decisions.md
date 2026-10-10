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

Valid default-mode direct, manual and document-tool answers receive the same app-owned publication binding around their current selected packet, exact user/scope/source, and existing R1 TurnContext. This does not enable preservation, SourceRuntime or planning. Historical/unbound evidence cannot mint a receipt. Bounded metadata checks may compute SHA-256 in PostgreSQL without returning source bodies; a current hash alone never proves that arbitrary delivered text came from that source. Delivered slices and media require captured exact provenance.

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
