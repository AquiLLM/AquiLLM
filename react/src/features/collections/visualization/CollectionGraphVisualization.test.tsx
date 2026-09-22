// @vitest-environment jsdom

import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { schema, readyGraph } from "./collectionGraphTestFixtures";
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

describe("CollectionGraphVisualization", () => {
  it("shows automatic document progress and a useful capacity failure", async () => {
    const loadInstance = vi.fn().mockResolvedValue({
      ...readyGraph, nodes: [], edges: [], artifact_id: null,
      status: { ...readyGraph.status, state: "partial", error_code: "document_builds_failed" },
      progress: { total: 31, active: 7, failed: 24, pending: 0, building: 0, ingesting: 0,
        failures: [{ code: "extraction_entity_limit", count: 24 }] },
    });
    render(<CollectionGraphVisualization collectionId="7"
      loadSchema={vi.fn().mockResolvedValue(schema)} loadInstance={loadInstance} />);
    fireEvent.click(screen.getByRole("button", { name: "Instance Graph" }));
    expect(await screen.findByText(/7 of 31 document graphs complete/)).toBeTruthy();
    expect(screen.getByText(/24 failed/)).toBeTruthy();
    expect(screen.getByText(/graph processing limit/)).toBeTruthy();
  });

  it("keeps polling after an unchanged building response and stops when ready", async () => {
    vi.useFakeTimers();
    let calls = 0;
    const loadInstance = vi.fn(async () => ({
      ...readyGraph,
      status: { ...readyGraph.status, state: ++calls >= 3 ? "ready" as const : "building" as const },
    }));
    render(<CollectionGraphVisualization collectionId="7"
      loadSchema={vi.fn().mockResolvedValue(schema)} loadInstance={loadInstance} />);
    fireEvent.click(screen.getByRole("button", { name: "Instance Graph" }));
    await act(async () => {});
    expect(loadInstance).toHaveBeenCalledTimes(1);
    await act(async () => { await vi.advanceTimersByTimeAsync(5000); });
    expect(loadInstance).toHaveBeenCalledTimes(2);
    await act(async () => { await vi.advanceTimersByTimeAsync(5000); });
    expect(loadInstance).toHaveBeenCalledTimes(3);
    await act(async () => { await vi.advanceTimersByTimeAsync(10000); });
    expect(loadInstance).toHaveBeenCalledTimes(3);
  });

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
});
