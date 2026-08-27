// @vitest-environment jsdom

import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { CollectionSchemaEnvelope } from "../knowledgeGraph/schemaTypes";
import type { CollectionGraphEnvelope } from "./collectionGraphTypes";
import CollectionGraphVisualization from "./CollectionGraphVisualization";

vi.mock("./CollectionGraphCanvas", () => ({
  default: ({ graph, onSelect }: any) => (
    <div data-testid="mock-graph-canvas">
      {graph.nodes.map((node: any) => (
        <button key={node.id} onClick={() => onSelect(node)}>
          {node.label}
        </button>
      ))}
      {graph.edges.map((edge: any) => (
        <button key={edge.id} onClick={() => onSelect(edge)}>
          {edge.label}
        </button>
      ))}
    </div>
  ),
}));

const schema = {
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

const readyGraph: CollectionGraphEnvelope = {
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

const graphWithIsolatedEntity: CollectionGraphEnvelope = {
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

afterEach(() => cleanup());

describe("CollectionGraphVisualization", () => {
  it("loads and renders the schema graph by default", async () => {
    render(
      <CollectionGraphVisualization
        collectionId="7"
        loadSchema={vi.fn().mockResolvedValue(schema)}
        loadInstance={vi.fn().mockResolvedValue(readyGraph)}
        requestRebuild={vi.fn()}
      />,
    );

    expect(await screen.findByRole("button", { name: "paper" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "authored_by" })).toBeTruthy();
    expect(
      screen
        .getByRole("button", { name: "Schema" })
        .getAttribute("aria-pressed"),
    ).toBe("true");
  });

  it("explains a partial instance graph and allows authorized rebuilds", async () => {
    const requestRebuild = vi
      .fn()
      .mockResolvedValue({ request_id: "r2", status: "queued" });
    const partial = {
      ...readyGraph,
      artifact_id: null,
      nodes: [],
      edges: [],
      status: {
        state: "partial" as const,
        error_code: "task_terminal_failure",
        request_id: "r1",
        updated_at: null,
      },
    };
    render(
      <CollectionGraphVisualization
        collectionId="7"
        loadSchema={vi.fn().mockResolvedValue(schema)}
        loadInstance={vi.fn().mockResolvedValue(partial)}
        requestRebuild={requestRebuild}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Instance Graph" }));
    expect(await screen.findByText(/partially/i)).toBeTruthy();
    expect(screen.getByText(/task_terminal_failure/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Rebuild graph" }));
    await waitFor(() => expect(requestRebuild).toHaveBeenCalledWith("7"));
  });

  it("shows clickable relation evidence for a ready instance graph", async () => {
    render(
      <CollectionGraphVisualization
        collectionId="7"
        loadSchema={vi.fn().mockResolvedValue(schema)}
        loadInstance={vi.fn().mockResolvedValue(readyGraph)}
        requestRebuild={vi.fn()}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Instance Graph" }));
    fireEvent.click(
      await screen.findByRole("button", { name: "evaluates_on" }),
    );

    expect(screen.getByText("Aquilla evaluates MMLU.")).toBeTruthy();
    expect(screen.getByText(/document-1/)).toBeTruthy();
  });

  it("shows source evidence when an instance entity is selected", async () => {
    render(
      <CollectionGraphVisualization
        collectionId="7"
        loadSchema={vi.fn().mockResolvedValue(schema)}
        loadInstance={vi.fn().mockResolvedValue(readyGraph)}
        requestRebuild={vi.fn()}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Instance Graph" }));
    fireEvent.click(await screen.findByRole("button", { name: "Aquilla" }));

    expect(screen.getByText("Aquilla evaluates MMLU.")).toBeTruthy();
  });

  it("shows connected entities by default and exposes isolated entities on demand", async () => {
    render(
      <CollectionGraphVisualization
        collectionId="7"
        loadSchema={vi.fn().mockResolvedValue(schema)}
        loadInstance={vi.fn().mockResolvedValue(graphWithIsolatedEntity)}
        requestRebuild={vi.fn()}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Instance Graph" }));

    expect(await screen.findByRole("button", { name: "Aquilla" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "MMLU" })).toBeTruthy();
    expect(
      screen.queryByRole("button", { name: "Standalone concept" }),
    ).toBeNull();
    expect(screen.getByText(/2 connected · 1 unconnected/i)).toBeTruthy();

    fireEvent.click(
      screen.getByRole("button", { name: "Include unconnected" }),
    );

    expect(
      screen.getByRole("button", { name: "Standalone concept" }),
    ).toBeTruthy();
  });
});
