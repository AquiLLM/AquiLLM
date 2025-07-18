export interface FileSystemItem {
    id: number;
    type: 'collection' | 'pdf' | 'audio' | 'arxiv' | 'transcript' | 'document' | 'texdocument' | string;
    name: string;
    created_at?: string;
    updated_at?: string;
    // Add more fields if needed (e.g. parentId, updated_at, etc.)
  }
  