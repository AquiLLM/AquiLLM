export interface CollectionContent {
  id: number;
  type: string;
  name: string;
  created_at: string;
  document_count?: number;
  parent_document_id?: string | null;
}

export interface CollectionViewProps {
  collectionId: string;
  onBack?: () => void;
}

export interface CollectionBreadcrumb {
  name: string;
  id: number | null;
  path: string;
  fullPath: string;
}

export type CollectionViewMode = "files" | "knowledge-graph" | "visualization";

export const COLLECTION_VIEW_QUERY_PARAM = "view";

export type CollectionViewNavigationIntent =
  | { type: "mode"; mode: CollectionViewMode }
  | { type: "browser"; mode: CollectionViewMode };

export type CollectionViewNavigationGuard = (
  intent: CollectionViewNavigationIntent,
) => boolean;

export function parseCollectionViewMode(search: string): CollectionViewMode {
  const view = new URLSearchParams(search).get(COLLECTION_VIEW_QUERY_PARAM);
  return view === "knowledge-graph" || view === "visualization"
    ? view
    : "files";
}

export function buildCollectionViewSearch(
  currentSearch: string,
  mode: CollectionViewMode,
): string {
  const params = new URLSearchParams(currentSearch);
  if (mode === "files") {
    params.delete(COLLECTION_VIEW_QUERY_PARAM);
  } else {
    params.set(COLLECTION_VIEW_QUERY_PARAM, mode);
  }
  const serialized = params.toString();
  return serialized ? `?${serialized}` : "";
}

export function buildCollectionViewUrl(
  pathname: string,
  currentSearch: string,
  hash: string,
  mode: CollectionViewMode,
): string {
  return `${pathname}${buildCollectionViewSearch(currentSearch, mode)}${hash}`;
}
