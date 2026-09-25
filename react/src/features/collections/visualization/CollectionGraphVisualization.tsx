import React, { useCallback, useEffect, useMemo, useState } from "react";
import type { CollectionSchemaEnvelope } from "../knowledgeGraph/schemaTypes";
import CollectionGraphCanvas from "./CollectionGraphCanvas";
import { CollectionGraphDetails } from "./CollectionGraphDetails";
import { CollectionGraphStatus } from "./CollectionGraphStatus";
import { CollectionGraphScope } from "./CollectionGraphScope";
import { connectedGraph, connectedNodeIds, filterGraph, needsProgressPoll, statusMessage } from "./collectionGraphPresentation";
import {
  defaultLoadSchema,
  fetchCollectionGraph,
  requestCollectionGraphRebuild,
} from "./collectionGraphApi";
import type {
  CollectionGraphEnvelope,
  VisualizationEdge,
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
  const [includeUnconnected, setIncludeUnconnected] = useState(false);

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

  const shouldPoll = needsProgressPoll(instance);
  useEffect(() => {
    if (mode !== "instance" || !shouldPoll) return;
    let cancelled = false;
    let timeout: number;
    const poll = () => {
      void loadInstance(collectionId)
        .then((value) => {
          if (!cancelled) {
            setInstance(value);
            if (needsProgressPoll(value)) timeout = window.setTimeout(poll, 5000);
          }
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
    };
    timeout = window.setTimeout(poll, 5000);
    return () => {
      cancelled = true;
      window.clearTimeout(timeout);
    };
  }, [collectionId, shouldPoll, loadInstance, mode]);

  const graph = useMemo(() => {
    if (mode === "schema")
      return schema ? mapSchemaEnvelopeToGraph(schema) : null;
    return instance ? mapInstanceEnvelopeToGraph(instance) : null;
  }, [instance, mode, schema]);
  const types = useMemo(
    () => [...new Set((graph?.nodes ?? []).map((node) => node.type))].sort(),
    [graph],
  );
  const instanceCounts = useMemo(() => {
    if (!graph || mode !== "instance") return null;
    const connected = connectedNodeIds(graph).size;
    return { connected, unconnected: graph.nodes.length - connected };
  }, [graph, mode]);
  const scopedGraph = useMemo(() => {
    if (!graph || mode !== "instance" || includeUnconnected) return graph;
    return connectedGraph(graph);
  }, [graph, includeUnconnected, mode]);
  const visibleGraph = useMemo(
    () => (scopedGraph ? filterGraph(scopedGraph, query, type) : null),
    [query, scopedGraph, type],
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
    setIncludeUnconnected(false);
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

      {mode === "instance" && graph && instanceCounts && (
        <CollectionGraphScope
          connected={instanceCounts.connected}
          unconnected={instanceCounts.unconnected}
          bounded={Boolean(instance?.truncated.nodes)}
          includeUnconnected={includeUnconnected}
          onChange={(next) => { setIncludeUnconnected(next); setSelected(null); }}
        />
      )}

      {loading && (
        <p className="rounded-[18px] bg-scheme-shade_4 p-4">Loading graph…</p>
      )}
      {error && (
        <p role="alert" className="rounded-[18px] border border-red-400 p-4">
          {error}
        </p>
      )}
      {mode === "instance" && (
        <CollectionGraphStatus instance={instance} message={instanceMessage} rebuilding={rebuilding} onRebuild={() => void rebuild()} />
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
          <CollectionGraphDetails selected={selected} />
        </div>
      )}
    </section>
  );
};

export default CollectionGraphVisualization;
