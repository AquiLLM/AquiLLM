import { Folder } from '../components/CollectionsTree';

/**
 * Shows a temporary message then clears it after a delay
 * 
 * @param setter - The state setter function
 * @param message - The message to display
 * @param duration - How long to show the message (in ms)
 */
export const showTempMessage = (
  setter: React.Dispatch<React.SetStateAction<string | null>>,
  message: string,
  duration = 5000
) => {
  setter(message);
  setTimeout(() => setter(null), duration);
};

/**
 * Parses collection data from API responses
 * 
 * @param collectionsData - Raw collections data from API
 * @returns Parsed collection objects
 */
export const parseCollections = (collectionsData: any[]): Folder[] => {
  return collectionsData.map((col: any) => ({
    id: col.id,
    name: col.name,
    parent: col.parent,
    collection: col.id,
    path: col.path,
    children: col.children || [],
    document_count: col.document_count,
    children_count: col.children_count,
    created_at: new Date(col.created_at || new Date()).toLocaleString(),
    updated_at: new Date(col.updated_at || new Date()).toISOString(),
  }));
}; 