import { getCookie } from './csrf';

/**
 * Moves a collection to a new parent collection
 * 
 * @param collectionId - ID of the collection to move
 * @param newParentId - ID of the new parent collection (null for root)
 * @returns Promise with the response data
 */
export const moveCollection = async (collectionId: number, newParentId: number | null): Promise<any> => {
  try {
    const res = await fetch(`/collection/move/${collectionId}/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': getCookie('csrftoken'),
      },
      body: JSON.stringify({ new_parent_id: newParentId }),
      credentials: 'include',
    });

    if (!res.ok) {
      const data = await res.json();
      throw new Error(data.error || `Failed to move collection: ${res.status} ${res.statusText}`);
    }

    return await res.json();
  } catch (err) {
    console.error('Error moving collection:', err);
    throw err;
  }
};

/**
 * Moves a document to a new collection
 * 
 * @param documentId - ID of the document to move
 * @param newCollectionId - ID of the destination collection (null for root)
 * @returns Promise with the response data
 */
export const moveDocument = async (documentId: number, newCollectionId: number | null): Promise<any> => {
  try {
    const res = await fetch(`/document/move/${documentId}/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': getCookie('csrftoken'),
      },
      body: JSON.stringify({ new_collection_id: newCollectionId }),
      credentials: 'include',
    });

    if (!res.ok) {
      const data = await res.json();
      throw new Error(data.error || `Failed to move document: ${res.status} ${res.statusText}`);
    }

    return await res.json();
  } catch (err) {
    console.error('Error moving document:', err);
    throw err;
  }
};

/**
 * Deletes a collection
 * 
 * @param collectionId - ID of the collection to delete
 * @returns Promise with the response data
 */
export const deleteCollection = async (collectionId: number): Promise<any> => {
  try {
    const res = await fetch(`/collection/delete/${collectionId}/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': getCookie('csrftoken'),
      },
      credentials: 'include',
    });

    if (!res.ok) {
      const data = await res.json();
      throw new Error(data.error || `Failed to delete collection: ${res.status} ${res.statusText}`);
    }

    return await res.json();
  } catch (err) {
    console.error('Error deleting collection:', err);
    throw err;
  }
};

/**
 * Creates a new collection
 * 
 * @param name - Name of the collection to create
 * @param parentId - Optional parent collection ID (null for root)
 * @returns Promise with the response data
 */
export const createCollection = async (name: string, parentId: number | null = null): Promise<any> => {
  try {
    const res = await fetch('/api/collections/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': getCookie('csrftoken'),
      },
      body: JSON.stringify({ 
        name,
        parent_id: parentId
      }),
      credentials: 'include',
    });

    if (!res.ok) {
      const data = await res.json();
      throw new Error(data.error || `Failed to create collection: ${res.status} ${res.statusText}`);
    }

    return await res.json();
  } catch (err) {
    console.error('Error creating collection:', err);
    throw err;
  }
}; 