import type { CollectionSchemaEnvelope } from "../knowledgeGraph/schemaTypes";
import type { CollectionGraphEnvelope } from "./collectionGraphTypes";

export const schema = {
  collection_id: "7",
  published: {
    version: 1,
    checksum: "checksum",
    entities: [
      {
        key: "paper",
        origin: "generated",
        change_state: "unchanged",
        capabilities: {
          editable_fields: [],
          removable: false,
          renameable: false,
        },
        values: {
          name: "paper",
          description: "A paper.",
          aliases: [],
          default_retrieval_weight: 1,
          default_suppression_policy: "never",
          default_suppression_threshold: 0,
        },
      },
      {
        key: "author",
        origin: "generated",
        change_state: "unchanged",
        capabilities: {
          editable_fields: [],
          removable: false,
          renameable: false,
        },
        values: {
          name: "author",
          description: "An author.",
          aliases: [],
          default_retrieval_weight: 1,
          default_suppression_policy: "never",
          default_suppression_threshold: 0,
        },
      },
    ],
    relations: [
      {
        key: "authored_by",
        origin: "generated",
        change_state: "unchanged",
        capabilities: {
          editable_fields: [],
          removable: false,
          renameable: false,
        },
        values: {
          name: "authored_by",
          description: "Authorship.",
          direction: "directed",
          allowed_head_types: ["paper"],
          allowed_tail_types: ["author"],
        },
      },
    ],
  },
  draft: null,
  permissions: {
    level: "VIEW",
    can_create_draft: false,
    can_edit_definitions: false,
    can_validate: false,
    can_publish: false,
    can_discard_draft: false,
    can_restore: false,
    can_view_history: true,
  },
  constraints: { entity_fields: {}, relation_fields: {} },
} as CollectionSchemaEnvelope;

export const readyGraph: CollectionGraphEnvelope = {
  collection_id: "7",
  artifact_id: "12",
  status: {
    state: "ready",
    error_code: null,
    request_id: null,
    updated_at: null,
  },
  permissions: { can_rebuild: true },
  nodes: [
    {
      id: "entity:1",
      label: "Aquilla",
      entity_type: "model",
      confidence: 0.9,
      retrieval_utility: 0.7,
      evidence: [
        {
          document_id: "document-1",
          chunk_id: 22,
          start: 0,
          end: 7,
          excerpt: "Aquilla evaluates MMLU.",
        },
      ],
    },
    {
      id: "entity:2",
      label: "MMLU",
      entity_type: "benchmark",
      confidence: 0.8,
      retrieval_utility: 0.6,
      evidence: [],
    },
  ],
  edges: [
    {
      id: "relation:1",
      source: "entity:1",
      target: "entity:2",
      relation_type: "evaluates_on",
      confidence: 0.85,
      support_count: 1,
      evidence: [
        {
          document_id: "document-1",
          chunk_id: 22,
          start: 0,
          end: 22,
          excerpt: "Aquilla evaluates MMLU.",
        },
      ],
    },
  ],
  truncated: { nodes: false, edges: false },
};

export const graphWithIsolatedEntity: CollectionGraphEnvelope = {
  ...readyGraph,
  nodes: [
    ...readyGraph.nodes,
    {
      id: "entity:3",
      label: "Standalone concept",
      entity_type: "concept",
      confidence: 0.95,
      retrieval_utility: 0.95,
      evidence: [],
    },
  ],
};
