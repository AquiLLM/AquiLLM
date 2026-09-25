import { useCallback, useEffect, useState } from 'react';
import type { Collection } from '../../../components/CollectionsTree';
import { mapCollectionFromApi } from '../../../components/collectionsPageMap';
import formatUrl from '../../../utils/formatUrl';
import { buildOrderedCollectionContents } from './collectionViewContents';
import type { CollectionContent } from './collectionViewTypes';

export interface UseCollectionViewDataResult {
  collection: Collection | null;
  contents: CollectionContent[];
  loading: boolean;
  error: string | null;
  permissionSource: {
    direct: boolean;
    source_collection_id: number | null;
    source_collection_name: string | null;
    permission_level: string | null;
  } | null;
  allCollections: Collection[];
  /** Loading affordance only; schema envelope overrides in KG workspace. */
  initialCanEdit: boolean;
  /** Loading affordance only; schema envelope overrides in KG workspace. */
  initialCanManage: boolean;
  fetchCollectionData: () => void;
  refreshAllCollections: () => void;
  setContents: React.Dispatch<React.SetStateAction<CollectionContent[]>>;
}

export function useCollectionViewData(collectionId: string): UseCollectionViewDataResult {
  const [collection, setCollection] = useState<Collection | null>(null);
  const [contents, setContents] = useState<CollectionContent[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [permissionSource, setPermissionSource] = useState<
    UseCollectionViewDataResult['permissionSource']
  >(null);
  const [allCollections, setAllCollections] = useState<Collection[]>([]);
  const [initialCanEdit, setInitialCanEdit] = useState(false);
  const [initialCanManage, setInitialCanManage] = useState(false);

  const fetchCollectionData = useCallback(() => {
    setLoading(true);
    setError(null);
    fetch(formatUrl(window.apiUrls.api_collection, { col_id: collectionId }), {
      headers: { Accept: 'application/json' },
    })
      .then((res) => {
        if (!res.ok) {
          return res.json().then((err) => {
            throw new Error(err.error || 'Failed to fetch collection');
          });
        }
        return res.json();
      })
      .then((data) => {
        if (!data.collection) throw new Error('Invalid response format');
        if (data.permission_source) setPermissionSource(data.permission_source);
        else setPermissionSource(null);
        setInitialCanEdit(Boolean(data.can_edit));
        setInitialCanManage(Boolean(data.can_manage));
        setCollection({
          id: data.collection.id,
          name: data.collection.name,
          parent: data.collection.parent,
          collection: data.collection.id,
          path: data.collection.path,
          children: data.children || [],
          document_count: data.documents?.length || 0,
          children_count: data.children?.length || 0,
          created_at: data.collection.created_at
            ? new Date(data.collection.created_at).toLocaleString()
            : new Date().toLocaleString(),
          updated_at: data.collection.updated_at
            ? new Date(data.collection.updated_at).toISOString()
            : new Date().toISOString(),
        });
        setContents(buildOrderedCollectionContents(data.documents || [], data.children || []));
        setLoading(false);
      })
      .catch((err: Error) => {
        console.error('Error refetching collection:', err);
        setError(err.message);
        setCollection(null);
        setContents([]);
        setPermissionSource(null);
        setInitialCanEdit(false);
        setInitialCanManage(false);
        setLoading(false);
      });
  }, [collectionId]);

  const refreshAllCollections = useCallback(() => {
    fetch(window.apiUrls.api_collections, {
      headers: { Accept: 'application/json' },
      credentials: 'include',
    })
      .then((res) => {
        if (!res.ok) throw new Error('Failed to fetch available collections');
        return res.json();
      })
      .then((data) => {
        const collectionsData = data.collections || [];
        setAllCollections(collectionsData.map((col: any) => mapCollectionFromApi(col)));
      })
      .catch((err) => console.error('Error fetching all collections:', err));
  }, []);

  useEffect(() => {
    setCollection(null);
    setContents([]);
    setPermissionSource(null);
    setInitialCanEdit(false);
    setInitialCanManage(false);
    setError(null);
    fetchCollectionData();
  }, [fetchCollectionData]);

  useEffect(() => {
    refreshAllCollections();
  }, [refreshAllCollections]);

  return {
    collection,
    contents,
    loading,
    error,
    permissionSource,
    allCollections,
    initialCanEdit,
    initialCanManage,
    fetchCollectionData,
    refreshAllCollections,
    setContents,
  };
}
