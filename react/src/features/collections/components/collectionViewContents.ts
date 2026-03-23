import type { CollectionContent } from './collectionViewTypes';

export function buildOrderedCollectionContents(
  documents: any[],
  children: any[]
): CollectionContent[] {
  const transformedDocuments = (documents || []).map((doc: any) => ({
    id: doc.id,
    type: doc.type || 'document',
    name: doc.title || 'Untitled',
    created_at: doc.ingestion_date
      ? new Date(doc.ingestion_date).toLocaleString()
      : doc.created_at
        ? new Date(doc.created_at).toLocaleString()
        : new Date().toLocaleString(),
    document_count: 0,
  }));
  const transformedChildren = (children || []).map((child: any) => ({
    id: child.id,
    type: 'collection',
    name: child.name,
    created_at: new Date(child.created_at || new Date()).toLocaleString(),
    document_count: child.document_count,
    parent_document_id: child.parent_document_id || null,
  }));
  const docsById = new Map<string, CollectionContent>(
    transformedDocuments.map((doc: CollectionContent) => [String(doc.id), doc])
  );
  const anchoredChildrenByDoc = new Map<string, CollectionContent[]>();
  const unanchoredChildren: CollectionContent[] = [];

  for (const child of transformedChildren as CollectionContent[]) {
    const parentDocumentId = child.parent_document_id ? String(child.parent_document_id) : '';
    if (parentDocumentId && docsById.has(parentDocumentId)) {
      const existing = anchoredChildrenByDoc.get(parentDocumentId) || [];
      existing.push(child);
      anchoredChildrenByDoc.set(parentDocumentId, existing);
    } else {
      unanchoredChildren.push(child);
    }
  }

  const orderedContents: CollectionContent[] = [];
  for (const doc of transformedDocuments as CollectionContent[]) {
    orderedContents.push(doc);
    const anchoredChildren = anchoredChildrenByDoc.get(String(doc.id)) || [];
    anchoredChildren.sort((a, b) => a.name.localeCompare(b.name));
    orderedContents.push(...anchoredChildren);
  }
  unanchoredChildren.sort((a, b) => a.name.localeCompare(b.name));
  orderedContents.push(...unanchoredChildren);

  return orderedContents;
}
