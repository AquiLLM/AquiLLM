import { Collection } from './CollectionsTree';

/** Normalize one collection object from GET /api/collections JSON. */
export function mapCollectionFromApi(col: {
  id: number;
  name: string;
  parent: number | null;
  path?: string;
  document_count?: number;
  children_count?: number;
  created_at?: string;
  updated_at?: string;
  children?: unknown[];
}): Collection {
  return {
    id: col.id,
    name: col.name,
    parent: col.parent,
    collection: col.id,
    path: col.path ?? '',
    children: Array.isArray(col.children) ? (col.children as Collection[]) : [],
    document_count: col.document_count ?? 0,
    children_count: col.children_count ?? 0,
    created_at: new Date(col.created_at || new Date()).toLocaleString(),
    updated_at: new Date(col.updated_at || new Date()).toISOString(),
  };
}

export function rootCollectionsFromParsed(parsed: Collection[]): Collection[] {
  return parsed.filter((col) => col.parent === null);
}
