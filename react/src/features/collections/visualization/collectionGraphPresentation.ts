import type { CollectionGraphEnvelope, VisualizationGraph } from './collectionGraphTypes';

export function statusMessage(envelope: CollectionGraphEnvelope) {
  switch (envelope.status.state) {
    case 'building':
      return 'Graph generation is still in progress.';
    case 'partial':
      return 'Graph generation completed partially; at least one document failed.';
    case 'failed':
      return 'Graph generation failed before an active graph could be published.';
    case 'empty':
      return 'No active instance graph has been generated for this collection yet.';
    default:
      return null;
  }
}

export function needsProgressPoll(envelope: CollectionGraphEnvelope | null) {
  return envelope?.status.state === 'building' || Boolean(envelope?.progress && (
    envelope.progress.ingesting + envelope.progress.pending + envelope.progress.building > 0
  ));
}

export function failureMessage(code: string) {
  switch (code) {
    case 'extraction_chunk_limit':
    case 'extraction_character_limit':
    case 'extraction_entity_limit':
    case 'extraction_relation_limit':
    case 'extraction_observation_limit':
      return 'The document exceeds the current graph processing limit.';
    default:
      return 'Document graph processing failed. A rebuild can retry it.';
  }
}

export function filterGraph(graph: VisualizationGraph, query: string, type: string) {
  const normalized = query.trim().toLocaleLowerCase();
  const matchingEdges = new Set(
    graph.edges
      .filter((edge) => !normalized || edge.label.toLocaleLowerCase().includes(normalized))
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

export function connectedNodeIds(graph: VisualizationGraph) {
  return new Set(graph.edges.flatMap((edge) => [edge.source, edge.target]));
}

export function connectedGraph(graph: VisualizationGraph): VisualizationGraph {
  const connected = connectedNodeIds(graph);
  return {
    nodes: graph.nodes.filter((node) => connected.has(node.id)),
    edges: graph.edges,
  };
}
