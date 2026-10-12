# AquiLLM RAG architecture and retrieval adversarial review

Reviewed on 10 October 2026 against checkout `6e756e252999ee6ccf604f88ea5326f49ddaa6d4`, branch `development`.

AquiLLM's largest architectural weakness is the lack of a continuous contract connecting original source meaning, index identity, retrieved evidence, and published claims. The system has extensive authorization, provenance, budget, and graph consistency checks. Yet source facts can disappear or change meaning during parsing, incompatible embedding spaces can be searched together, relevant passages can be discarded before synthesis, and an answer that fails citation validation can still be published. Additional graph sophistication cannot repair these upstream losses.

This review identifies **19 prioritized findings**. Five are P1 priorities: structured-source fidelity, embedding compatibility, final citation enforcement, reranker outage amplification, and graph identity collapse. The remainder are P2 reliability or retrieval-quality defects and design limitations. These priorities describe product impact and repair urgency; they are not claims of unauthenticated exploits. No permission bypass or live data disclosure was established.

## Scope and evidence

The review covered upload and archive handling, parsers, chunk publication, embeddings, dense and lexical acquisition, reranking, graph extraction and identity resolution, graph projection and traversal, query preparation, evidence selection, synthesis, citations, memory, and evaluation gates. Production code and existing documentation drafts were preserved. Only this audit directory was added. No application database, model service, or deployed server was contacted.

Findings distinguish executable offline reproductions from source-derived counterexamples and deployment-dependent risks. The repository contains newer protections behind switches: windowed reranking, source evidence, budgeted document capacity, follow-up continuity, and iterative acquisition. They are meaningful mitigations, but the checked-in defaults remain `legacy` or disabled for these modes. The actual deployed settings were not inspected. See [preservation configuration](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/apps/chat/services/rag_preservation_config.py#L64) and [example settings](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/.env.example#L492). Graph-specific findings apply when graph retrieval is enabled.

The offline probes execute current functions or structurally selected blocks from current source, with in-memory documents and mocked database/provider boundaries. They establish the described behavior, not production failure rates or model hallucination rates. The ORM double does not establish PostgreSQL execution plans.

| Stage | Principal failure |
| --- | --- |
| Source extraction | Tables, empty cells, formulas, and archive members lose their original meaning or identity. |
| Index publication | Text hash substitutes for a complete parser, chunker, and embedding generation identity. |
| Candidate acquisition | Exact hits are capped in arbitrary document order; dense retrieval has no embedding compatibility predicate. |
| Reranking | Default local adapters see prefixes; outages trigger repeated protocol attempts and vector-first fallback. |
| Graph discovery | Labels become identities; query-independent graph and evidence pruning limit relevant additions. |
| Evidence selection | Relative ranking and feasible token packing do not establish answerability. |
| Synthesis | A failed citation repair can pass publication; valid citation syntax does not establish support. |
| Memory and evaluation | Negated memory requests can persist; some graph quality gates are too weak. |

## P1 findings

### F01 Structured sources can produce wrong or missing facts before indexing

**Confirmed with in-memory files.** XLSX extraction removes blank cells before serializing a row. With headers `Product, Domestic, Foreign`, both `[Widget, 7, None]` and `[Widget, None, 7]` become `Widget, 7`. These distinct facts then have identical extracted text and can share a document hash. Formula cells use cached values only; an uncached `=50+20` disappears. DOCX extraction reads body paragraphs and omits native tables: a document containing a table value `SecretRevenue=93017` yielded only its introductory paragraph. PPTX's text path similarly lacks native table-cell traversal.

The flawed choice is treating plain concatenated text as a sufficient canonical representation for every supported format. Once column or table identity is erased, neither embedding similarity, graph extraction, nor a larger language model can reconstruct the original fact reliably. Character-sliced chunks also lack structural guarantees for headers, qualifiers, tables, and equations.

**Repair direction:** introduce a versioned extraction representation preserving ordered elements, cell coordinates, headers, empty values, formulas and cached-value status, and original source locations. Index useful text projections while retaining that structure. Reject or flag incomplete extraction instead of treating it as complete evidence. Reindex affected documents under a new parser version.

Evidence: [XLSX extraction](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/lib/parsers/spreadsheets/xlsx.py#L14), [DOCX extraction](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/lib/parsers/documents/docx.py#L15), [PPTX extraction](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/lib/parsers/presentations/pptx.py#L19), [character chunking](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/apps/documents/tasks/chunking.py#L153). Reproduction: `ingestion_probes.py`.

### F02 Embedding provenance is recorded but compatibility is not enforced at search

**Confirmed in the retrieval predicate and duplicate-reuse policy.** Dense search filters authorized document IDs and non-null vectors, then orders by L2 distance. It does not require stored embeddings to match the query's provider, model, revision, or preparation policy. A fixed 1024-dimensional column establishes shape, not a common semantic coordinate system.

Switching models while retaining the database, or using legacy Cohere fallback and later returning to local embeddings, can therefore compare incompatible vectors. Duplicate reuse compounds the issue: it selects an existing document using only text hash and completion status, copies its vectors and receipts, and skips text embedding. Changing chunk configuration also does not invalidate this reuse. The probe reproduced that lookup and preservation of an old-model receipt; it deliberately mocked receipt validity and did not claim a live mixed-index incident.

**Repair direction:** define an index-generation identity covering source revision, extraction and chunking policy, embedding model/revision, input transformations, dimensions, and modality. Search only compatible generations. Stage reindexing and explicitly activate a complete generation. Include this identity in duplicate reuse.

`local-only` prevents new cross-provider fallback but does not repair an existing mixed index or govern model migrations. The strict KG embedding signature is a separate protection. Evidence: [dense search](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/apps/documents/services/chunk_search_candidates.py#L199), [duplicate selection and copying](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/apps/documents/tasks/chunking.py#L120), [embedding bypass](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/apps/documents/tasks/chunking.py#L391), [fallback policy](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/lib/embeddings/config.py#L20).

### F03 Failed citation repair can still publish an invalid answer

**Confirmed by executing the real enforcement block and validator.** After one repair attempt, the final guard runs only if the answer contains unknown citation tokens. A fully uncited answer has no unknown tokens. An answer with one allowed citation plus an uncited numeric claim also has no unknown tokens. Both can reach final answer construction despite failing the actual citation validator.

The probe used an initial answer `The capacity is 999 litres.` and two repair outputs: that same uncited sentence, and `A supported statement [doc:a chunk:1]. The capacity is 999 litres.` Both survived the real enforcement block; both failed validation. This affects direct synthesis despite its stricter numeric-citation policy and deferred delivery.

**Repair direction:** make the complete citation predicate a final publication invariant. Revalidate after every repair or fallback. If repair fails, publish a grounded limited answer or verified authorized extracts. Unknown-token detection is only one component of that predicate.

This fix would enforce citation syntax and placement, not factual entailment; F10 addresses that separate issue. Evidence: [post-repair guard](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/lib/llm/providers/complete_turn.py#L1007), [final answer construction](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/lib/llm/providers/complete_turn.py#L1073). Reproduction: `citation_probe.py`.

### F04 Reranker outages amplify work instead of bounding it

**Confirmed with mocked HTTP responses.** A 24-candidate local rerank generated **63 HTTP attempts** when every request immediately returned 503. Both the legacy rank-only adapter and scored adapter reproduced this. They try endpoint and payload variants, then individual passage pairs. Pointwise concurrency is local to each invocation. The legacy turn does not create the preservation ledger, and its parallel searches await completion without a shared retrieval deadline.

The flawed choice is discovering protocol compatibility repeatedly on the request path and treating per-call timeouts as sufficient control. A temporary service outage can consume many more requests, threads, and seconds than a successful call. The probe measured request count, not wall-clock outage latency; an immediate 503 and a slow timeout have different elapsed costs.

**Repair direction:** verify one provider capability outside the user request, distinguish unsupported protocol from upstream failure, stop discovery on outage responses, and enforce one shared deadline, attempt budget, and process-wide concurrency budget. Carry bounded cancellation through worker completion. The newer windowed path already has useful budget machinery to build on.

Evidence: [scored discovery loops](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/apps/documents/services/chunk_rerank_local_scored.py#L192), [pointwise executor](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/apps/documents/services/chunk_rerank_local_scored_http.py#L77), [legacy turn lifetime](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/apps/chat/services/rag_turn.py#L52), [parallel search wait](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/apps/chat/services/rag_pipeline.py#L190). Reproduction: `probes.py`.

### F05 Graph identity resolution treats matching labels as proof of identity

**Confirmed with the canonical resolver.** Two unrelated, unanchored entities named `Mercury`, both typed `model` and belonging to different collections, merge into one component with automatic confidence **0.99**. Document and collection resolution have related exact-name rules.

The system does guard conflicting identifiers, different types, explicit versions, ambiguous acronyms, and component conflicts. Those controls do not establish that two ordinary same-type homonyms are the same entity. An absent identifier conflict is not positive identity evidence. Once the entities are merged, query-time ambiguity checks cannot recover their original separation; their relationships can contaminate one another's candidate neighborhoods.

**Repair direction:** retain uncertain name matches as candidate links. Require contextual identity evidence or stable anchors before automatic promotion; make splits and correction propagation first-class operations. Include homonyms from different research groups and papers in held-out identity evaluation.

Evidence: [document name equivalence](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/apps/knowledge_graph/resolution/coreference.py#L1230), [collection union](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/apps/knowledge_graph/resolution/collection.py#L1787), [canonical name edges](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/apps/knowledge_graph/resolution/canonical.py#L810), [automatic confidence](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/apps/knowledge_graph/resolution/canonical.py#L685). Reproduction: `graph_probes.py`. This establishes resolver behavior, not how often the extractor produces such inputs.

## P2 findings

### F06 Default reranking discards the part of a passage that may contain the answer

**Confirmed in captured request payloads.** Legacy local adapters truncate each candidate to its first **900 characters**, before further pair budgeting. Default chunks are **2048 characters** with **384-character overlap**. An answer-bearing region late in a chunk can be outside that chunk's reranker prefix and outside the next chunk entirely. In the probe, an answer marker after character 900 appeared in none of the 63 captured payloads.

This creates a representation mismatch: dense acquisition can select a whole chunk for an answer that the reranker never sees, so the chunk is rejected before synthesis. Conversely, a relevant introduction can cause selection of a passage whose later text contradicts it. Public tool excerpts introduce another truncation boundary.

**Repair direction:** score exact source windows with verified model-token accounting and preserve the selected spans through synthesis. The repository already contains windowed/source modes; validate and activate those through measured coverage gates rather than merely increasing a character cap. Evidence: [scored prefix slicing](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/apps/documents/services/chunk_rerank_local_scored.py#L75), [legacy prefix slicing](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/apps/documents/services/chunk_rerank_local_vllm.py#L135), [defaults](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/aquillm/apps.py#L85). Reproduction: `probes.py`.

### F07 Exact retrieval limits arbitrary document order instead of relevance

**Confirmed with the real candidate collector and an ORM double.** Exact terms are combined with OR, then matches are ordered by document UUID and chunk number before applying the limit. Twelve early chunks mentioning only `ATP` excluded a later chunk containing both `ATP` and `BRCA1` plus the requested answer.

The exact channel therefore favors early documents and repeated partial matches. Adding irrelevant material in an early-sorting document can evict useful exact evidence. The term extractor also ignores short scientific identifiers: `Compare CO and NO at pH 7` produces no exact terms. Whole-passage trigram similarity is not a substitute for relevance-ranked lexical retrieval or structured identifier handling.

**Repair direction:** rank lexical candidates by phrase/identifier coverage and term relevance before limiting them; preserve short scientific symbols deliberately. Use diversity only after protecting strong matches. Evaluate a proper lexical ranker alongside dense search and retain trigram matching for spelling variation. Evidence: [term extraction](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/apps/documents/services/chunk_search_candidates.py#L79), [OR predicate](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/apps/documents/services/chunk_search_candidates.py#L113), [ordering and cap](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/apps/documents/services/chunk_search_candidates.py#L234). Reproduction: `probes.py`.

### F08 Reranker failure turns hybrid search into vector-first search

**Confirmed with the real collector and fallback.** The baseline pool is a deduplicated concatenation of vector, trigram, and exact results. Reranker fallback takes its first `top_k` entries. With twelve vector candidates and a correct exact-only candidate, top ten fallback returns only vector candidates. Graph-only additions appended after the baseline face the same problem.

Consequently the system may complete a nominal hybrid search while losing the channel that found the answer. Upper-level query-list RRF happens after this loss and cannot repair it.

**Repair direction:** preserve individual channel ranks and create a meaningful deterministic fused ordering before reranking. Use that ordering on failure, with degraded-ranking status available to the rest of the turn. Test recall under dense, lexical, graph, and reranker failures independently. Evidence: [concatenation](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/apps/documents/services/chunk_search_candidates.py#L247), [rank-only fallback](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/apps/documents/services/chunk_rerank_parse.py#L21), [scored fallback](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/apps/documents/services/chunk_rerank.py#L221). Reproduction: `probes.py`.

### F09 Conversational query preparation guesses source references from grammar

**Source-confirmed design gap with an optional-mode failure.** Legacy pronoun handling prepends the first title in the latest retrieved-document list; that need not be the document the user meant. The optional LLM rewrite is a placeholder returning unchanged input, and multi-query preparation splits punctuation rather than semantic requirements.

The newer follow-up resolver has a different failure: words such as `their`, `it`, and `that` are treated as historical source references. With follow-up evidence enabled, a fresh question such as `What causes stars to lose their mass?` can become an unresolved paper reference and be stopped before retrieval. It does not need a previous paper reference to contain those grammatical pronouns.

**Repair direction:** preserve the original question and resolve references only when conversation context supplies a plausible source antecedent. Maintain constraints, negation, units, and identifiers in any generated variants. Explicit ordinal citations and document references deserve deterministic handling; incidental pronouns should not block a fresh search. Evidence: [legacy preparation](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/apps/chat/services/rag_query.py#L55), [rewrite placeholder](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/apps/chat/services/rag_query.py#L65), [pronoun recognition](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/apps/chat/services/rag_source_anchor_resolution.py#L20), [unresolved reference handling](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/apps/chat/services/rag_source_continuity_turn.py#L45). The follow-up switch defaults off.

### F10 Ranking and citation presence do not establish evidence sufficiency

**Confirmed algorithmic limitation.** Final scoring normalizes within the candidate pool: `(-100, -99)` becomes `(0, 1)`, and equal scores become `0.5`. The selector fills feasible slots without an absolute support stop condition; even zero-relevance candidates can be selected. Separately, the citation validator checks permitted identifiers and some placement rules, explicitly not entailment.

A valid citation to an irrelevant passage can therefore satisfy syntax. A nonempty result set is not proof that a comparison, exhaustive claim, or negative claim has enough support. Fixed final passage and per-document caps can further prevent complete evidence for multi-part research questions.

**Repair direction:** distinguish retrieval success, relevance, support for each requested aspect, contradiction, and answerability. Preserve raw scores for calibration where meaningful; treat rank-only fallback confidence as unknown. Publish partial answers with explicit unsupported aspects. Existing coverage assessment, revision-bound source references, and negative-claim instructions provide a foundation, but do not establish semantic support merely by validating coordinates.

Evidence: [normalization](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/apps/chat/services/rag_selection_scoring.py#L51), [selection loop](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/apps/chat/services/rag_selection.py#L139), [citation limitation](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/lib/llm/providers/rag_citations.py#L124), [coverage validation](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/apps/chat/services/rag_coverage.py#L102).

### F11 Graph candidate relevance largely ignores the requested relationship

**Source-confirmed limitation with a transition-weight probe.** Direct graph preparation uses entity spans. Projected PageRank passes only source, target, and weight into its kernel. Passage mapping receives no query and scores relation evidence using the stronger endpoint's mass.

For the same entity seeds, `Which dataset does Atlas use?` and `Which paper contradicts Atlas?` can receive the same graph candidates. Predicate-specific weights and direction influence adjacency, but the requested predicate is not matched to the question. Multiple predicate groups between the same endpoints also add transition credit: the probe changed one destination's share from **0.5 to 0.9** by adding eight equal-weight groups.

**Repair direction:** preserve query-to-relation relevance through graph admission and evidence mapping. Distinguish independent supporting evidence from repeated or differently labeled edges. Final reranking is a mitigation only for candidates that survive the earlier caps. Evidence: [direct seeds](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/apps/knowledge_graph/retrieval/production_direct.py#L78), [kernel inputs](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/apps/knowledge_graph/retrieval/projected_ppr.py#L155), [passage mapping](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/apps/knowledge_graph/retrieval/production_runtime_support.py#L94). Reproduction: `graph_probes.py`.

### F12 Query-independent graph evidence sampling creates an early recall ceiling

**Confirmed policy; production impact requires corpus measurement.** Topology preparation retains a selected physical copy of a logical transition and up to three supporting evidence rows from it. It retains two mentions per identity, ordered by extraction confidence and opaque provenance keys. Graph candidate mapping imposes another three-chunks-per-document ceiling per branch.

A third mention can contain the property requested by the user while two higher-confidence introductory mentions consume the allowance. A losing physical relation copy can contain an important qualification or contradiction. Extraction confidence measures confidence in extraction, not usefulness for this question.

**Repair direction:** separate bounded topology discovery from question-aware evidence retrieval, retaining source and claim diversity. Preserve enough coordinates to fetch relevant supporting and contradicting passages after identifying an entity or relation. Baseline search can independently recover omitted text, but that does not remove the graph channel's limitation. Evidence: [transition evidence selection](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/apps/knowledge_graph/projection/topology_selection.py#L38), [mention selection](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/apps/knowledge_graph/projection/topology_snapshot.py#L169), [per-document candidate cap](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/apps/knowledge_graph/retrieval/production_runtime_support.py#L106).

### F13 Graph coverage has abrupt readiness and size cliffs

**Source-confirmed behavior.** Ready-scope construction requires ready authority for every selected collection and exact coverage of selected documents. Adding a new or rebuilding collection can remove graph retrieval value from otherwise ready sources. Direct retrieval additionally requires compatible ontology generations.

Separately, acquisition rejects raw topology families that exceed their caps before the later fanout policy can choose a bounded neighborhood. A growing hub can disable a branch rather than yield its strongest admissible neighborhood. Physical copies may consume the budget before canonical collapse.

**Repair direction:** represent readiness and completeness per source. Admit a bounded authorized frontier during acquisition, and make incomplete coverage explicit. Preserve version and permission guarantees while allowing ready subsets to contribute. This is a graph-availability defect; baseline retrieval survives, and overflow is isolated from sibling branches.

Evidence: [scope-wide readiness](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/apps/knowledge_graph/retrieval/ready_scope.py#L211), [raw family caps](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/apps/knowledge_graph/projection/topology_adapter.py#L224), [overflow handling](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/apps/knowledge_graph/projection/memgraph_records.py#L160), [later fanout](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/apps/knowledge_graph/projection/topology_selection.py#L51).

### F14 Physical acquisition cannot follow every canonical cross-collection path

**Static counterexample, not a live Memgraph reproduction.** Each selected generation is loaded from the original seed list, with paths constrained to that generation. Suppose collection A contains `seed → shared identity` and collection B contains `shared identity → answer`. If the shared identity is discovered in A rather than being an original seed, B is not subsequently reseeded from it. The merged snapshot cannot traverse an edge that was never loaded.

Original canonical seeds correctly activate matching copies in multiple collections. The gap concerns identities discovered during traversal. The architectural mismatch is between physical per-generation acquisition and reasoning over the composed canonical graph.

**Repair direction:** expand a bounded canonical frontier across compatible authorized generations, with explicit depth accounting and provenance. Add this exact bridge scenario to integration evaluation. Evidence: [fixed original seeds per generation](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/apps/knowledge_graph/projection/topology_adapter.py#L211), [generation-constrained paths](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/apps/knowledge_graph/projection/topology_cypher.py#L7).

### F15 Archived PDFs lose their original citation artifact

**Confirmed with isolated ingestion and fake persistence.** A PDF member extracted from a ZIP creates a PDFDocument, but the persistence branch saves the outer upload's bytes and filename. The probe stored `bundle.zip` and ZIP bytes in the PDF file field while marking the item successful.

The searchable text belongs to the member PDF, while the file used for source inspection belongs to the archive. A valid chunk citation consequently does not guarantee a usable original source artifact.

**Repair direction:** carry immutable member bytes, archive path, and source digest in each parser payload, and bind text and citations to that artifact. Evidence: [archive extraction](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/aquillm/ingestion/parser_archive.py#L38), [PDF persistence](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/aquillm/task_ingest_uploaded.py#L158). Reproduction: `ingestion_probes.py`.

### F16 A failed multi-output upload can leave untracked retrievable documents

**Confirmed with isolated ingestion and fake persistence.** Payloads are saved sequentially, while the batch's document IDs are recorded only after all payloads succeed. If the second payload has an integrity collision, the first document remains saved, the batch becomes `error`, and `document_ids` stays empty. A retry starts over and may collide with the earlier output.

Individual document chunk publication has transaction and recovery mechanisms. They do not provide an atomic or resumable ownership ledger for the enclosing archive or multi-output upload.

**Repair direction:** assign durable per-output identities, checkpoint results idempotently, and expose partial success or stage visibility until completion. Ensure retry and crash recovery resume the existing outputs. Evidence: [payload loop](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/aquillm/task_ingest_uploaded.py#L49), [individual save](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/aquillm/task_ingest_uploaded.py#L177), [late output ledger and failure](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/aquillm/task_ingest_uploaded.py#L226). Reproduction: `ingestion_probes.py`.

### F17 Office image fallback is unreachable when conversion is unavailable

**Source-confirmed and separately exercised offline.** `_extract_via_pdf` contains `yield`, so calling it always returns a generator. Callers test whether that object is `None`, then return after consuming it. When LibreOffice is absent or conversion fails, the generator produces nothing; the direct embedded-image fallback is skipped.

This affects DOCX, PPTX, XLSX, and ODS figure extraction. It weakens the optional visual path precisely when conversion support is missing and cannot be relied on to rescue the structured-text omissions in F01.

**Repair direction:** return an explicit conversion outcome, or separate conversion availability from iteration. Test absent conversion, failed conversion, empty successful conversion, and successful direct fallback independently. Evidence: [generator helper](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/aquillm/ingestion/figure_extraction/office.py#L55), [caller](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/aquillm/ingestion/figure_extraction/office.py#L248), [spreadsheet caller](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/aquillm/ingestion/figure_extraction/spreadsheet.py#L216). The saved ingestion probe file covers F01, F02, F15, and F16; this separate probe is not included there.

### F18 Negated remember requests can become durable prompt context

**Source-confirmed with an offline helper probe.** Remember intent is a substring test. If extraction returns no facts, `Do not remember that I use telescope A.` enters the explicit-remember fallback and survives cleaning. Profile facts are then interpolated into the system prompt. The profile block lacks the explicit anti-instruction wording used for episodic memories.

The primary defect is persisting a negated request. The broader concern is treating extracted profile strings as durable high-priority context without a strong distinction between data and instructions. No successful malicious model takeover was demonstrated.

**Repair direction:** require affirmative persistence intent, distinguish quotation and hypothetical discussion, and never persist the entire utterance merely because extraction declined it. Bound and delimit all retrieved memory as data; maintain provenance and user-visible correction/deletion. Evidence: [intent predicate](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/lib/memory/extraction/stable_facts.py#L206), [persistence fallback](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/aquillm/memory.py#L101), [system-context formatting](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/lib/memory/formatting.py#L24).

### F19 Some graph rollout gates can pass without acceptable quality

**Confirmed gate predicates.** The identity-precision gate passes whenever automatic precision exceeds candidate precision; values of 0.10 and 0.01 satisfy it. The retrieval-quality gate needs some improving case covering each required tag. A single improving case carrying all tags can satisfy that gate while many other cases regress. The separate multi-hop gate compares PPR with one-hop, which does not establish non-regression against the dense baseline.

The repository has substantial evaluation infrastructure: live execution, multiple arms, provenance and permission checks, parity, freshness, deterministic replay, and metrics. The weakness is the sufficiency of these particular acceptance predicates, not an absence of evaluation. This does not assert that every newer release or experiment uses only these gates.

**Repair direction:** require absolute identity-precision floors, representative sample counts, aggregate and per-category non-regression, held-out support labels, and uncertainty bounds. Test complete supporting-set recall and final factual claims in addition to top-ten ranking metrics. Evidence: [identity gate](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/apps/knowledge_graph/evals/run_kg_eval.py#L3843), [retrieval gate](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/apps/knowledge_graph/evals/run_kg_eval.py#L3859).

## Risks requiring deployment or corpus measurements

- **Filtered approximate search:** the shared HNSW index and document-scope filter can lose recall when the planner uses approximate search and filtering removes neighbors. No iterative-scan settings were found in the inspected application/deployment code, but database-global settings or an exact plan could change the outcome. Compare filtered ANN against exact neighbors and inspect actual query plans. This is a measurement target, not a confirmed deployed incident. [Local query](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/apps/documents/services/chunk_search_candidates.py#L199), [pgvector filtering documentation](https://github.com/pgvector/pgvector#filtering).
- **Lexical query cost:** `similarity(content, query) > threshold` with a very permissive threshold is not the same query shape as PostgreSQL's documented index-supported similarity operators. The existence of a trigram GIN index alone does not prove this stage is efficient. Inspect generated SQL and `EXPLAIN (ANALYZE, BUFFERS)` on representative scopes. [Local expression](https://github.com/AquiLLM/AquiLLM/blob/6e756e252999ee6ccf604f88ea5326f49ddaa6d4/aquillm/apps/documents/services/chunk_search_candidates.py#L221), [PostgreSQL trigram index support](https://www.postgresql.org/docs/18/pgtrgm.html#PGTRGM-INDEX).
- **Prompt injection and corpus poisoning:** authorized documents are still untrusted content. The selected-evidence instructions and tool allowlist are useful defenses, but citation membership does not prevent a malicious passage from contaminating a claim. Evaluate document instructions, fake citations, repeated high-overlap passages, contradictions, and persistence into memory. This review established deterministic data and enforcement defects; it did not run a live model red-team campaign.
- **Budget interactions:** the default final limit, 15-row per-search ceiling, three-query ceiling, graph evidence sampling, and per-document cap interact. Their combined recall loss cannot be inferred from each unit test passing. Measure answer-bearing evidence at every boundary, especially comparisons requiring multiple passages from one document.

## Defenses worth preserving

The inspected code contains explicit current-scope authorization and revalidation, graph generation/provenance validation, branch-local failure isolation, transactional chunk replacement and recovery, scorer identity checks, whole-pool fallback when scores are incomparable, source-revision checks, and request-local evidence budgets in preservation modes. Earlier tool evidence is removed from the direct synthesis request copy. The synthesis policy instructs the model to use selected evidence, preserve disagreements, and scope negative conclusions. Coverage planning validates permitted actions and exact source coordinates.

These mechanisms prevent important classes of failure and should survive remediation. A bounded source subset with honest completeness status is preferable to silently presenting it as exhaustive. No recommendation here requires weakening authorization or accepting stale graph generations.

## Repair order and acceptance evidence

1. **Repair deterministic integrity defects:** citation publication, spreadsheet/Office extraction, archive source ownership, and upload output tracking. Acceptance requires preserving exact table facts and source bytes, rejecting invalid final answers, and idempotent restart after failure on any payload.
2. **Enforce index compatibility:** introduce versioned indexing and query compatibility, including duplicate reuse. Acceptance requires a deliberate model/chunker change to reject incompatible rows or activate a complete replacement generation, with no silent mixed-space search.
3. **Preserve evidence through ranking:** validate windowed/source modes, relevance-ranked lexical acquisition, and meaningful hybrid fallback. Acceptance requires suffix-answer, short-identifier, distractor-flooding, and provider-outage cases to retain the correct supporting passages.
4. **Bound total retrieval work:** resolve provider capability outside requests; enforce shared deadlines, attempts, and worker limits. Acceptance requires a controlled outage to terminate within the declared stage/turn budget without multiplying work across query variants.
5. **Prove graph contribution:** fix homonym resolution and coverage discontinuities, then compare baseline, direct-only, extended-only, and combined retrieval at equal reranker/evidence budgets. Include cross-generation bridges, high-degree hubs, contradictions, and answer-bearing third mentions.
6. **Gate on supported answers:** use held-out questions checked against original sources. Measure complete support recall, unsupported numeric claims, citation correctness, abstention/partial-answer quality, and p95 latency. Keep adversarial fixtures separate from tuning data.

The architectural target is one explicit chain: immutable source artifact and structured extraction → compatible index generation → authorized candidates with meaningful ranks → exact selected evidence with completeness status → supported claims that pass final publication checks.

## Reproduction and validation record

Run the audit scripts from the repository root with system Python and the existing dependencies:

```powershell
rtk proxy python -B docs/audits/2026-10-10-rag-adversarial/probes.py
rtk proxy python -B docs/audits/2026-10-10-rag-adversarial/ingestion_probes.py
rtk proxy python -B docs/audits/2026-10-10-rag-adversarial/citation_probe.py
rtk proxy python -B docs/audits/2026-10-10-rag-adversarial/graph_probes.py
```

All four were executed successfully. `probes.py` reproduced exact-match starvation, short-identifier omission, vector-first fallback, missing suffix evidence, and 63 outage attempts in both local adapters. `ingestion_probes.py` reproduced table/formula loss, wrong archive source bytes, partial untracked output, and unversioned duplicate reuse. `citation_probe.py` reproduced two invalid post-repair outputs. `graph_probes.py` reproduced homonym merging and predicate-group transition inflation. Its additional uniform reverse-weight cancellation example is informative mathematical behavior, not counted as a separate defect.

The focused existing candidate-acquisition, rerank-budget, and provider-score tests produced **27 passed, 1 explicitly deselected** under isolated Django settings. The deselected conversation-search compatibility test requires the full installed-app registry. An initial combined run also could not collect the citation test module because its import reaches environment-dependent application settings (`SECRET_KEY`); this was a harness limitation, not classified as an application defect. The citation behavior was instead exercised by the standalone probe. No full application suite, PostgreSQL integration run, live model benchmark, or production load test is claimed.
