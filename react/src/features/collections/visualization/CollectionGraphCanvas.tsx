import React, { useEffect, useRef } from "react";
import cytoscape, {
  type ElementDefinition,
  type StylesheetJson,
} from "cytoscape";
import type {
  VisualizationEdge,
  VisualizationGraph,
  VisualizationNode,
} from "./collectionGraphTypes";

export interface CollectionGraphCanvasProps {
  graph: VisualizationGraph;
  onSelect: (element: VisualizationNode | VisualizationEdge) => void;
}

const styles: StylesheetJson = [
  {
    selector: "node",
    style: {
      label: "data(label)",
      "background-color": "#7892bf",
      color: "#172033",
      "font-size": "11px",
      "text-wrap": "ellipsis",
      "text-max-width": "120px",
      "text-valign": "bottom",
      "text-margin-y": 7,
      width: 34,
      height: 34,
      "border-width": 2,
      "border-color": "#40577f",
    },
  },
  {
    selector: 'node[kind = "schema-entity"]',
    style: { shape: "round-rectangle", "background-color": "#9baed0" },
  },
  {
    selector: "edge",
    style: {
      label: "data(label)",
      width: 2,
      "line-color": "#7084a8",
      "target-arrow-color": "#7084a8",
      "target-arrow-shape": "triangle",
      "curve-style": "bezier",
      "font-size": "9px",
      color: "#33415c",
      "text-background-color": "#eef2f8",
      "text-background-opacity": 0.9,
      "text-background-padding": "2px",
    },
  },
  {
    selector: ":selected",
    style: {
      "border-color": "#345ea8",
      "border-width": 4,
      "line-color": "#345ea8",
      "target-arrow-color": "#345ea8",
    },
  },
];

const CollectionGraphCanvas: React.FC<CollectionGraphCanvasProps> = ({
  graph,
  onSelect,
}) => {
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!containerRef.current || graph.nodes.length === 0) return undefined;
    const byId = new Map<string, VisualizationNode | VisualizationEdge>([
      ...graph.nodes.map((node) => [node.id, node] as const),
      ...graph.edges.map((edge) => [edge.id, edge] as const),
    ]);
    const elements: ElementDefinition[] = [
      ...graph.nodes.map((node) => ({
        data: {
          id: node.id,
          label: node.label,
          kind: node.kind,
          type: node.type,
        },
      })),
      ...graph.edges.map((edge) => ({
        data: {
          id: edge.id,
          source: edge.source,
          target: edge.target,
          label: edge.label,
          kind: edge.kind,
        },
      })),
    ];
    const instance = cytoscape({
      container: containerRef.current,
      elements,
      style: styles,
      layout: {
        name: "cose",
        animate: false,
        fit: true,
        padding: 28,
        nodeRepulsion: () => 5000,
        idealEdgeLength: () => 90,
      },
      minZoom: 0.2,
      maxZoom: 3,
    });
    instance.on("tap", "node, edge", (event) => {
      const selected = byId.get(event.target.id());
      if (selected) onSelect(selected);
    });
    return () => instance.destroy();
  }, [graph, onSelect]);

  return (
    <div
      ref={containerRef}
      role="img"
      aria-label={`Knowledge graph with ${graph.nodes.length} nodes and ${graph.edges.length} relations`}
      className="h-[620px] min-h-[420px] w-full rounded-[18px] border border-border-low_contrast bg-scheme-shade_2"
      data-testid="collection-graph-canvas"
    />
  );
};

export default CollectionGraphCanvas;
