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
