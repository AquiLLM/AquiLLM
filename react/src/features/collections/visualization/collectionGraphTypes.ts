export type CollectionGraphState =
  "building" | "partial" | "failed" | "empty" | "ready";

export interface CollectionGraphStatus {
  state: CollectionGraphState;
  error_code: string | null;
  request_id: string | null;
  updated_at: string | null;
}

export interface CollectionGraphEvidence {
  document_id: string;
  chunk_id: number;
  start: number;
  end: number;
  excerpt: string;
}

export interface CollectionGraphNode {
  id: string;
  label: string;
  entity_type: string;
  confidence: number;
  retrieval_utility: number;
  evidence: CollectionGraphEvidence[];
}

export interface CollectionGraphEdge {
  id: string;
  source: string;
  target: string;
  relation_type: string;
  confidence: number;
  support_count: number;
  evidence: CollectionGraphEvidence[];
}

export interface CollectionGraphEnvelope {
  collection_id: string;
  artifact_id: string | null;
  status: CollectionGraphStatus;
  permissions: { can_rebuild: boolean };
  nodes: CollectionGraphNode[];
  edges: CollectionGraphEdge[];
  truncated: { nodes: boolean; edges: boolean };
}

export interface VisualizationNode {
  id: string;
  label: string;
  kind: "schema-entity" | "instance-entity";
  type: string;
  description?: string;
  origin?: string;
  confidence?: number;
  retrievalUtility?: number;
  evidence?: CollectionGraphEvidence[];
}

export interface VisualizationEdge {
  id: string;
  source: string;
  target: string;
  label: string;
  kind: "schema-relation" | "instance-relation";
  description?: string;
  confidence?: number;
  supportCount?: number;
  evidence?: CollectionGraphEvidence[];
}

export interface VisualizationGraph {
  nodes: VisualizationNode[];
  edges: VisualizationEdge[];
}
