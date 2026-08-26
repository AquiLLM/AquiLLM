import { describe, expect, it } from "vitest";
import type { CollectionSchemaEnvelope } from "../knowledgeGraph/schemaTypes";
import { mapSchemaEnvelopeToGraph } from "./schemaGraphMapper";

const capabilities = {
  editable_fields: [],
  removable: false,
  renameable: false,
};

const schema = {
  collection_id: "7",
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
  published: {
    version: 1,
    checksum: "checksum",
    entities: [
      {
        key: "paper",
        origin: "generated",
        change_state: "unchanged",
        capabilities,
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
        origin: "collection",
        change_state: "unchanged",
        capabilities,
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
        capabilities,
        values: {
          name: "authored_by",
          description: "Connects papers and authors.",
          direction: "directed",
          allowed_head_types: ["paper"],
          allowed_tail_types: ["author"],
        },
      },
    ],
  },
  draft: null,
  constraints: { entity_fields: {}, relation_fields: {} },
} satisfies CollectionSchemaEnvelope;

describe("mapSchemaEnvelopeToGraph", () => {
  it("maps published entity types and allowed relation endpoints deterministically", () => {
    const graph = mapSchemaEnvelopeToGraph(schema);

    expect(graph.nodes.map((node) => [node.id, node.label, node.kind])).toEqual(
      [
        ["schema-entity:author", "author", "schema-entity"],
        ["schema-entity:paper", "paper", "schema-entity"],
      ],
    );
    expect(graph.edges).toEqual([
      expect.objectContaining({
        id: "schema-relation:authored_by:paper:author",
        source: "schema-entity:paper",
        target: "schema-entity:author",
        label: "authored_by",
        kind: "schema-relation",
      }),
    ]);
  });
});
