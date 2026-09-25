# Collection Graph Reliability and Visualization Design

**Date:** 2026-08-26
**Status:** Approved

## Objective

Make collection knowledge graphs reliable for documents containing PDF control
characters and add an evidence-oriented visualization surface for both the
published schema and the active collection instance graph.

## Observed failure

Collection 217 ("KG Test") has two documents and a published schema, but it has
no active collection graph. Both collection rebuild requests completed only
partially because one document failed. The first attempt exceeded the raw
provider entity-observation cap. A retry extracted 293 entity mentions and nine
relation mentions, then failed during coreference resolution with
`source text contains an unsafe control character`.

Graph activation is atomic. A terminal document failure prevents creation of an
active collection artifact and its Memgraph projection, so a healthy projection
service still has nothing to display.

## Reliability design

### Length-preserving source sanitization

Graph processing will use a derived, deterministic source representation. C0,
surrogate, and other disallowed control characters will be replaced one-for-one
with ordinary spaces. Tab, newline, carriage return, and permitted Unicode
format characters remain unchanged.

The sanitizer applies consistently to text sent to the extractor, the full text
used for span remapping, and persisted-chunk source context read by resolution.
It does not mutate the stored document or chunk. One input code point always
maps to one output code point, preserving all entity offsets and evidence links.
Strict validation remains in place after sanitization and for explicitly
supplied resolver input.

### Bounded observation handling

Provider observations and unique persisted entity mentions are different
quantities. Overlapping chunk observations can exceed the unique-entity limit
without creating an oversized graph. The extractor will retain a bounded raw
observation guard, deterministically map and deduplicate observations, and then
enforce the existing persisted entity and relation caps. It will never silently
truncate unique evidence. Genuine final-cap overflow remains a terminal,
specific failure.

### Build visibility

The visualization response will report one of `building`, `partial`, `failed`,
`empty`, or `ready`, together with bounded counts, the latest safe error code,
and timestamps. Authorized users can request a rebuild. An absent active graph
will therefore be explained rather than rendered as an unexplained blank area.

## User experience

Collections gain a third top-level mode: `Visualization`. It contains two
subviews.

### Schema

The schema view derives nodes and edges from the existing collection schema
workspace API:

- entity types are nodes;
- relation types are directed edges;
- generated/manual origin and draft status affect styling;
- selecting an item opens a details panel and can take editors to the existing
  schema editing surface.

The view supports pan, zoom, fit, search, and type/status filtering.

### Instance Graph

The instance view renders active `CollectionEntity` and `CollectionRelation`
rows only. It starts with a bounded, connected subgraph selected by support,
confidence, and retrieval utility. The initial response is capped at 150 nodes
and 300 edges. Search can seed a focused subgraph, and selecting a node can load
its bounded neighborhood.

Selecting a node or edge opens an inspector containing:

- label and entity/relation type;
- confidence, support, and provenance counts;
- contributing documents and chunks;
- expandable evidence text using the same document/chunk identity used by chat
  citations.

The browser never connects directly to Memgraph and never downloads the entire
collection graph. Cytoscape.js renders the bounded response and supplies
selection, pan/zoom, and layout behavior.

## Backend API

A permission-checked collection endpoint returns visualization data from the
authoritative active PostgreSQL graph. It uses the collection's existing view
permission and returns no graph data for unauthorized callers.

The response includes:

- collection and active-artifact identity;
- build/projection status;
- bounded node and edge records;
- truncation and continuation/focus metadata;
- safe evidence descriptors, with evidence text loaded only when requested.

The endpoint uses stable opaque client identifiers. Ordering and cap behavior
are deterministic, and queries avoid per-node database access.

## Frontend architecture

`CollectionViewMode` gains `visualization`, with URL state preserved in the
existing collection view query parameter. A visualization feature module owns:

- schema-to-graph mapping;
- instance graph fetching and status polling;
- Cytoscape lifecycle and layout;
- graph filters and selection state;
- the node/edge evidence inspector;
- accessible list fallbacks and empty/error states.

The visualization remains read-only. Schema mutations continue through the
existing draft editor and publishing workflow.

## Testing

Implementation follows test-driven development.

Backend tests cover sanitizer safety and offset preservation, explicit unsafe
input rejection, observation deduplication and caps, permissions, deterministic
ordering, bounded payloads, build states, evidence scoping, and query counts.

Frontend tests cover deep-link navigation, Schema/Instance switching, schema
mapping, graph selection, filters, status/empty/error rendering, rebuild
permissions, evidence inspection, and Cytoscape cleanup. Existing collection
files and schema editor behavior must remain unchanged.

## Deployment and verification

Changes are committed locally on `development`, pushed to
`origin/development`, then pulled and deployed on `aquillm-dev2`. Remote `.env`
contents are never committed or printed. Transcription services are excluded.

After deployment, collection 217 is rebuilt. Verification requires both
documents to complete, an active collection artifact, nonzero active entity and
relation counts, a ready projection, a populated visualization, and evidence
links that resolve to the original document chunks.
