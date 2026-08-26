import type { CollectionSchemaEnvelope } from "../knowledgeGraph/schemaTypes";
import type { VisualizationGraph } from "./collectionGraphTypes";

export function mapSchemaEnvelopeToGraph(
  envelope: CollectionSchemaEnvelope,
): VisualizationGraph {
  const entities = [...envelope.published.entities].sort((left, right) =>
    left.key.localeCompare(right.key),
  );
  const relations = [...envelope.published.relations].sort((left, right) =>
    left.key.localeCompare(right.key),
  );
  return {
    nodes: entities.map((entity) => ({
      id: `schema-entity:${entity.key}`,
      label: entity.values.name,
      kind: "schema-entity" as const,
      type: entity.key,
      description: entity.values.description,
      origin: entity.origin,
    })),
    edges: relations.flatMap((relation) =>
      [...relation.values.allowed_head_types].sort().flatMap((head) =>
        [...relation.values.allowed_tail_types].sort().map((tail) => ({
          id: `schema-relation:${relation.key}:${head}:${tail}`,
          source: `schema-entity:${head}`,
          target: `schema-entity:${tail}`,
          label: relation.values.name,
          kind: "schema-relation" as const,
          description: relation.values.description,
        })),
      ),
    ),
  };
}

export function mapInstanceEnvelopeToGraph(
  envelope: import("./collectionGraphTypes").CollectionGraphEnvelope,
): VisualizationGraph {
  return {
    nodes: envelope.nodes.map((node) => ({
      id: node.id,
      label: node.label,
      kind: "instance-entity" as const,
      type: node.entity_type,
      confidence: node.confidence,
      retrievalUtility: node.retrieval_utility,
      evidence: node.evidence,
    })),
    edges: envelope.edges.map((edge) => ({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      label: edge.relation_type,
      kind: "instance-relation" as const,
      confidence: edge.confidence,
      supportCount: edge.support_count,
      evidence: edge.evidence,
    })),
  };
}
