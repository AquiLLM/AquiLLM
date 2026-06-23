import { getCookie } from '../../../utils/csrf';

function csrfToken(): string {
  return getCookie('csrftoken');
}

async function jsonOrThrow<T>(r: Response): Promise<T> {
  if (!r.ok) {
    let detail = '';
    try {
      const data = await r.json();
      detail = data?.error ? `: ${data.error}` : '';
    } catch {
      // ignore
    }
    throw new Error(`Request failed (${r.status})${detail}`);
  }
  return r.json() as Promise<T>;
}

export interface CollectionNote {
  collection_id: number;
  collection_name: string;
  body: string;
  updated_at: string | null;
  updated_by: string | null;
  exists: boolean;
  max_body_length: number;
}

export async function getCollectionNote(collectionId: number): Promise<CollectionNote> {
  const r = await fetch(`/api/collections/${collectionId}/note/`, { credentials: 'include' });
  return jsonOrThrow<CollectionNote>(r);
}

export async function saveCollectionNote(
  collectionId: number,
  body: string,
): Promise<CollectionNote> {
  const r = await fetch(`/api/collections/${collectionId}/note/`, {
    method: 'PUT',
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
      'X-CSRFToken': csrfToken(),
    },
    body: JSON.stringify({ body }),
  });
  return jsonOrThrow<CollectionNote>(r);
}
