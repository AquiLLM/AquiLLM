
/**
 * Represents a collection/folder in the system
 * @interface Folder
 */
export interface Collection {
    id: number;
    name: string;
    parent: number | null;  // Parent collection ID, null if root-level
    collection: number;     // Reference to the collection this folder belongs to
    path: string;          // Full path in format: "parent/child/grandchild"
    children: Collection[];    // Nested collections
    document_count: number;
    children_count: number;
    created_at: string;
    updated_at: string;
  }