import { getCookie } from "../../../utils/csrf";
import { formatCollectionSchemaRoute } from "../knowledgeGraph/schemaApiRoutes";
import type {
  CollectionGraphEdge,
  CollectionGraphEnvelope,
  CollectionGraphEvidence,
  CollectionGraphNode,
  CollectionGraphState,
} from "./collectionGraphTypes";

const STATES = new Set<CollectionGraphState>([
  "building",
  "partial",
  "failed",
  "empty",
  "ready",
]);

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isNullableString(value: unknown): value is string | null {
  return value === null || typeof value === "string";
}

function isEvidence(value: unknown): value is CollectionGraphEvidence {
  return (
    isRecord(value) &&
    typeof value.document_id === "string" &&
    Number.isInteger(value.chunk_id) &&
    Number.isInteger(value.start) &&
    Number.isInteger(value.end) &&
    typeof value.excerpt === "string"
  );
}

function isNode(value: unknown): value is CollectionGraphNode {
  return (
    isRecord(value) &&
    typeof value.id === "string" &&
    typeof value.label === "string" &&
    typeof value.entity_type === "string" &&
    isNumber(value.confidence) &&
    isNumber(value.retrieval_utility) &&
    Array.isArray(value.evidence) &&
    value.evidence.every(isEvidence)
  );
}

function isEdge(value: unknown): value is CollectionGraphEdge {
  return (
    isRecord(value) &&
    typeof value.id === "string" &&
    typeof value.source === "string" &&
    typeof value.target === "string" &&
    typeof value.relation_type === "string" &&
    isNumber(value.confidence) &&
    Number.isInteger(value.support_count) &&
    Array.isArray(value.evidence) &&
    value.evidence.every(isEvidence)
  );
}

function parseEnvelope(value: unknown): CollectionGraphEnvelope {
  if (
    !isRecord(value) ||
    !isRecord(value.status) ||
    !isRecord(value.permissions)
  ) {
    throw new Error("invalid graph response");
  }
  const statusState = value.status.state;
  if (
    typeof statusState !== "string" ||
    !STATES.has(statusState as CollectionGraphState) ||
    typeof value.collection_id !== "string" ||
    !isNullableString(value.artifact_id) ||
    !isNullableString(value.status.error_code) ||
    !isNullableString(value.status.request_id) ||
    !isNullableString(value.status.updated_at) ||
    typeof value.permissions.can_rebuild !== "boolean" ||
    !Array.isArray(value.nodes) ||
    !value.nodes.every(isNode) ||
    !Array.isArray(value.edges) ||
    !value.edges.every(isEdge) ||
    !isRecord(value.truncated) ||
    typeof value.truncated.nodes !== "boolean" ||
    typeof value.truncated.edges !== "boolean"
  ) {
    throw new Error("invalid graph response");
  }
  return value as unknown as CollectionGraphEnvelope;
}

function route(key: string, collectionId: string): string {
  const pattern = window.apiUrls?.[key];
  if (!pattern) throw new Error(`missing graph route: ${key}`);
  return formatCollectionSchemaRoute(pattern, { col_id: collectionId });
}

export async function fetchCollectionGraph(
  collectionId: string,
  options: { query?: string; signal?: AbortSignal } = {},
): Promise<CollectionGraphEnvelope> {
  const url = new URL(
    route("api_collection_graph_visualization", collectionId),
    window.location.origin,
  );
  if (options.query) url.searchParams.set("q", options.query);
  const response = await fetch(`${url.pathname}${url.search}`, {
    credentials: "include",
    signal: options.signal,
    headers: { Accept: "application/json" },
  });
  if (!response.ok) throw new Error(`graph request failed: ${response.status}`);
  return parseEnvelope(await response.json());
}

export async function requestCollectionGraphRebuild(collectionId: string) {
  const response = await fetch(
    route("api_collection_graph_rebuild", collectionId),
    {
      method: "POST",
      credentials: "include",
      headers: {
        Accept: "application/json",
        "X-CSRFToken": getCookie("csrftoken") ?? "",
      },
    },
  );
  const value: unknown = await response.json();
  if (
    !response.ok ||
    !isRecord(value) ||
    typeof value.request_id !== "string" ||
    typeof value.status !== "string"
  ) {
    throw new Error(`graph rebuild failed: ${response.status}`);
  }
  return { request_id: value.request_id, status: value.status };
}
