// @vitest-environment jsdom

import {
  cleanup,
  fireEvent,
  render,
  screen,
} from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { schema, readyGraph, graphWithIsolatedEntity } from "./collectionGraphTestFixtures";
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

afterEach(() => { cleanup(); vi.useRealTimers(); });

describe("CollectionGraphVisualization evidence and scope", () => {
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
