// @vitest-environment jsdom

import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  fetchCollectionGraph,
  requestCollectionGraphRebuild,
} from "./collectionGraphApi";

const payload = {
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
      evidence: [],
    },
  ],
  edges: [],
  truncated: { nodes: false, edges: false },
};

beforeEach(() => {
  window.apiUrls = {
    api_collection_graph_visualization:
      "/api/collection/%(col_id)s/graph/visualization/",
    api_collection_graph_rebuild: "/api/collection/%(col_id)s/graph/rebuild/",
  };
});

describe("collectionGraphApi", () => {
  it("formats collection and search parameters and forwards abort signals", async () => {
    const signal = new AbortController().signal;
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(payload), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(
      fetchCollectionGraph("7", { query: "Aquilla", signal }),
    ).resolves.toEqual(payload);
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/collection/7/graph/visualization/?q=Aquilla",
      expect.objectContaining({ credentials: "include", signal }),
    );
  });

  it("rejects malformed graph responses", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({ ...payload, status: { state: "mystery" } }),
          {
            status: 200,
          },
        ),
      ),
    );

    await expect(fetchCollectionGraph("7")).rejects.toThrow(
      "invalid graph response",
    );
  });

  it("posts rebuild requests through the configured route", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({ request_id: "request-1", status: "queued" }),
        {
          status: 202,
        },
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(requestCollectionGraphRebuild("7")).resolves.toEqual({
      request_id: "request-1",
      status: "queued",
    });
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/collection/7/graph/rebuild/",
      expect.objectContaining({ method: "POST", credentials: "include" }),
    );
  });
});
