import type { Collection } from '../../../components/CollectionsTree';
import type { CollectionBreadcrumb } from './collectionViewTypes';

export function buildCollectionBreadcrumbs(
  collection: Collection,
  allCollections: Collection[]
): CollectionBreadcrumb[] {
  if (!collection.path) return [];

  const segments = collection.path.split('/').filter((segment) => segment.trim() !== '');
  const breadcrumbs: CollectionBreadcrumb[] = [];

  breadcrumbs.push({
    name: 'Root',
    id: null,
    path: '',
    fullPath: '',
  });

  let currentFullPath = '';
  const pathToCollectionMap = new Map<string, number>();

  allCollections.forEach((col) => {
    if (col.path) {
      pathToCollectionMap.set(col.path, col.id);
    }
  });

  for (let i = 0; i < segments.length; i++) {
    const segmentName = segments[i];
    currentFullPath = currentFullPath ? `${currentFullPath}/${segmentName}` : segmentName;

    const collectionId = pathToCollectionMap.get(currentFullPath);

    let matchingCollection: Collection | undefined;
    if (!collectionId) {
      matchingCollection = allCollections.find(
        (col) => col.name === segmentName && col.path && col.path.endsWith(currentFullPath)
      );
    }

    if (collectionId || matchingCollection) {
      breadcrumbs.push({
        name: segmentName,
        id: collectionId ?? matchingCollection?.id ?? null,
        path: segmentName,
        fullPath: currentFullPath,
      });
    } else {
      breadcrumbs.push({
        name: segmentName,
        id: null,
        path: segmentName,
        fullPath: currentFullPath,
      });
    }
  }

  return breadcrumbs;
}
