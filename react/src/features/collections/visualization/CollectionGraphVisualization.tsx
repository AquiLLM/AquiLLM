import React, { useCallback, useEffect, useMemo, useState } from "react";
import { createDefaultCollectionSchemaApi } from "../knowledgeGraph/collectionSchemaEditorHelpers";
import type { CollectionSchemaEnvelope } from "../knowledgeGraph/schemaTypes";
import CollectionGraphCanvas from "./CollectionGraphCanvas";
import {
  fetchCollectionGraph,
  requestCollectionGraphRebuild,
} from "./collectionGraphApi";
import type {
  CollectionGraphEnvelope,
  VisualizationEdge,
  VisualizationGraph,
  VisualizationNode,
} from "./collectionGraphTypes";
import {
  mapInstanceEnvelopeToGraph,
  mapSchemaEnvelopeToGraph,
} from "./schemaGraphMapper";

type VisualizationMode = "schema" | "instance";
type SelectedElement = VisualizationNode | VisualizationEdge | null;

export interface CollectionGraphVisualizationProps {
  collectionId: string;
  loadSchema?: (collectionId: string) => Promise<CollectionSchemaEnvelope>;
  loadInstance?: (collectionId: string) => Promise<CollectionGraphEnvelope>;
  requestRebuild?: (
    collectionId: string,
  ) => Promise<{ request_id: string; status: string }>;
}

async function defaultLoadSchema(collectionId: string) {
  const result =
    await createDefaultCollectionSchemaApi().loadWorkspace(collectionId);
  if (!result.ok) throw new Error(`schema request failed: ${result.kind}`);
  return result.data;
}

function statusMessage(envelope: CollectionGraphEnvelope) {
  switch (envelope.status.state) {
    case "building":
      return "Graph generation is still in progress.";
    case "partial":
      return "Graph generation completed partially; at least one document failed.";
    case "failed":
      return "Graph generation failed before an active graph could be published.";
    case "empty":
      return "No active instance graph has been generated for this collection yet.";
    default:
      return null;
  }
}

function filterGraph(graph: VisualizationGraph, query: string, type: string) {
  const normalized = query.trim().toLocaleLowerCase();
  const matchingEdges = new Set(
    graph.edges
      .filter(
        (edge) =>
          !normalized || edge.label.toLocaleLowerCase().includes(normalized),
      )
      .map((edge) => edge.id),
  );
  const edgeNodeIds = new Set(
    graph.edges
      .filter((edge) => matchingEdges.has(edge.id))
      .flatMap((edge) => [edge.source, edge.target]),
  );
  const nodes = graph.nodes.filter(
    (node) =>
      (!type || node.type === type) &&
      (!normalized ||
        node.label.toLocaleLowerCase().includes(normalized) ||
        node.type.toLocaleLowerCase().includes(normalized) ||
        edgeNodeIds.has(node.id)),
  );
  const nodeIds = new Set(nodes.map((node) => node.id));
  return {
    nodes,
    edges: graph.edges.filter(
      (edge) =>
        nodeIds.has(edge.source) &&
        nodeIds.has(edge.target) &&
        (!normalized ||
          matchingEdges.has(edge.id) ||
          nodes.some(
            (node) =>
              [edge.source, edge.target].includes(node.id) &&
              node.label.toLocaleLowerCase().includes(normalized),
          )),
    ),
  };
}

const CollectionGraphVisualization: React.FC<
  CollectionGraphVisualizationProps
> = ({
  collectionId,
  loadSchema = defaultLoadSchema,
  loadInstance = fetchCollectionGraph,
  requestRebuild = requestCollectionGraphRebuild,
}) => {
  const [mode, setMode] = useState<VisualizationMode>("schema");
  const [schema, setSchema] = useState<CollectionSchemaEnvelope | null>(null);
  const [instance, setInstance] = useState<CollectionGraphEnvelope | null>(
    null,
  );
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [type, setType] = useState("");
  const [selected, setSelected] = useState<SelectedElement>(null);
  const [rebuilding, setRebuilding] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      if ((mode === "schema" && schema) || (mode === "instance" && instance))
        return;
      setLoading(true);
      setError(null);
      try {
        if (mode === "schema") {
          const value = await loadSchema(collectionId);
          if (!cancelled) setSchema(value);
        } else {
          const value = await loadInstance(collectionId);
          if (!cancelled) setInstance(value);
        }
      } catch (loadError) {
        if (!cancelled) {
          setError(
            loadError instanceof Error
              ? loadError.message
              : "Unable to load graph",
          );
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    void load();
    return () => {
      cancelled = true;
    };
  }, [collectionId, instance, loadInstance, loadSchema, mode, schema]);

  useEffect(() => {
    if (mode !== "instance" || instance?.status.state !== "building") return;
    let cancelled = false;
    const timeout = window.setTimeout(() => {
      void loadInstance(collectionId)
        .then((value) => {
          if (!cancelled) setInstance(value);
        })
        .catch((pollError: unknown) => {
          if (!cancelled) {
            setError(
              pollError instanceof Error
                ? pollError.message
                : "Unable to refresh graph status",
            );
          }
        });
    }, 5000);
    return () => {
      cancelled = true;
      window.clearTimeout(timeout);
    };
  }, [collectionId, instance?.status.state, loadInstance, mode]);

  const graph = useMemo(() => {
    if (mode === "schema")
      return schema ? mapSchemaEnvelopeToGraph(schema) : null;
    return instance ? mapInstanceEnvelopeToGraph(instance) : null;
  }, [instance, mode, schema]);
  const types = useMemo(
    () => [...new Set((graph?.nodes ?? []).map((node) => node.type))].sort(),
    [graph],
  );
  const visibleGraph = useMemo(
    () => (graph ? filterGraph(graph, query, type) : null),
    [graph, query, type],
  );
  const onSelect = useCallback(
    (element: VisualizationNode | VisualizationEdge) => {
      setSelected(element);
    },
    [],
  );

  const changeMode = (next: VisualizationMode) => {
    setMode(next);
    setQuery("");
    setType("");
    setSelected(null);
    setError(null);
  };

  const rebuild = async () => {
    setRebuilding(true);
    setError(null);
    try {
      await requestRebuild(collectionId);
      setInstance((current) =>
        current
          ? {
              ...current,
              status: {
                ...current.status,
                state: "building",
                error_code: null,
              },
            }
          : current,
      );
    } catch (rebuildError) {
      setError(
        rebuildError instanceof Error
          ? rebuildError.message
          : "Unable to rebuild graph",
      );
    } finally {
      setRebuilding(false);
    }
  };

  const instanceMessage = instance ? statusMessage(instance) : null;

  return (
    <section
      className="space-y-[14px]"
      aria-label="Collection graph visualization"
    >
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-[18px] border border-border-low_contrast bg-scheme-shade_4 p-[10px]">
        <div className="flex gap-2" aria-label="Visualization kind">
          <button
            type="button"
            aria-pressed={mode === "schema"}
            onClick={() => changeMode("schema")}
            className={`h-[36px] rounded-[18px] border px-4 ${mode === "schema" ? "bg-accent text-white border-accent" : "border-border-mid_contrast"}`}
          >
            Schema
          </button>
          <button
            type="button"
            aria-pressed={mode === "instance"}
            onClick={() => changeMode("instance")}
            className={`h-[36px] rounded-[18px] border px-4 ${mode === "instance" ? "bg-accent text-white border-accent" : "border-border-mid_contrast"}`}
          >
            Instance Graph
          </button>
        </div>
        <div className="flex flex-wrap gap-2">
          <input
            aria-label="Search graph"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search nodes or relations"
            className="h-[36px] min-w-[230px] rounded-[18px] border border-border-mid_contrast bg-scheme-shade_2 px-3"
          />
          <select
            aria-label="Filter by type"
            value={type}
            onChange={(event) => setType(event.target.value)}
            className="h-[36px] rounded-[18px] border border-border-mid_contrast bg-scheme-shade_2 px-3"
          >
            <option value="">All types</option>
            {types.map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        </div>
      </div>

      {loading && (
        <p className="rounded-[18px] bg-scheme-shade_4 p-4">Loading graph…</p>
      )}
      {error && (
        <p role="alert" className="rounded-[18px] border border-red-400 p-4">
          {error}
        </p>
      )}
      {mode === "instance" && instanceMessage && (
        <div className="rounded-[18px] border border-border-mid_contrast bg-scheme-shade_4 p-4">
          <p>{instanceMessage}</p>
          {instance?.status.error_code && (
            <p className="mt-1 text-sm text-text-lower_contrast">
              Build error: <code>{instance.status.error_code}</code>
            </p>
          )}
          {instance?.permissions.can_rebuild && (
            <button
              type="button"
              onClick={() => void rebuild()}
              disabled={rebuilding || instance.status.state === "building"}
              className="mt-3 h-[36px] rounded-[18px] border border-border-mid_contrast px-4 disabled:opacity-50"
            >
              {rebuilding ? "Starting rebuild…" : "Rebuild graph"}
            </button>
          )}
        </div>
      )}

      {visibleGraph && visibleGraph.nodes.length > 0 && (
        <div className="grid gap-[14px] xl:grid-cols-[minmax(0,1fr)_330px]">
          <div>
            <div className="mb-2 flex items-center justify-between text-sm text-text-lower_contrast">
              <span>
                {visibleGraph.nodes.length} nodes · {visibleGraph.edges.length}{" "}
                relations
              </span>
              {mode === "instance" && instance?.truncated.nodes && (
                <span>Showing a bounded subgraph</span>
              )}
            </div>
            <CollectionGraphCanvas graph={visibleGraph} onSelect={onSelect} />
            <label className="mt-2 block text-sm">
              Select graph element
              <select
                className="ml-2 rounded border border-border-mid_contrast bg-scheme-shade_2 px-2 py-1"
                value={selected?.id ?? ""}
                onChange={(event) => {
                  const element = [
                    ...visibleGraph.nodes,
                    ...visibleGraph.edges,
                  ].find((item) => item.id === event.target.value);
                  if (element) setSelected(element);
                }}
              >
                <option value="">Choose…</option>
                {[...visibleGraph.nodes, ...visibleGraph.edges].map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.label}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <aside className="rounded-[18px] border border-border-low_contrast bg-scheme-shade_4 p-4">
            {!selected ? (
              <p>
                Select a node or relation to inspect its details and evidence.
              </p>
            ) : (
              <>
                <h2 className="text-lg font-semibold">{selected.label}</h2>
                <p className="text-sm text-text-lower_contrast">
                  {selected.kind}
                </p>
                {"type" in selected && (
                  <p className="mt-2">Type: {selected.type}</p>
                )}
                {selected.description && (
                  <p className="mt-2">{selected.description}</p>
                )}
                {selected.confidence !== undefined && (
                  <p className="mt-2">
                    Confidence: {(selected.confidence * 100).toFixed(1)}%
                  </p>
                )}
                {"supportCount" in selected &&
                  selected.supportCount !== undefined && (
                    <p className="mt-1">
                      Supporting passages: {selected.supportCount}
                    </p>
                  )}
                {"evidence" in selected &&
                  selected.evidence &&
                  selected.evidence.length > 0 && (
                    <div className="mt-4 space-y-3">
                      <h3 className="font-semibold">Evidence</h3>
                      {selected.evidence.map((evidence) => (
                        <details
                          key={`${evidence.document_id}:${evidence.chunk_id}:${evidence.start}`}
                          className="rounded border border-border-low_contrast p-2"
                          open
                        >
                          <summary className="cursor-pointer text-sm">
                            Document {evidence.document_id} · chunk{" "}
                            {evidence.chunk_id}
                          </summary>
                          <p className="mt-2 whitespace-pre-wrap text-sm">
                            {evidence.excerpt}
                          </p>
                        </details>
                      ))}
                    </div>
                  )}
              </>
            )}
          </aside>
        </div>
      )}
    </section>
  );
};

export default CollectionGraphVisualization;
