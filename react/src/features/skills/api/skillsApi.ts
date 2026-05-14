import { getCookie } from '../../../utils/csrf';
import type { Skill, SkillsListResponse } from '../types';

function csrfToken(): string {
  return getCookie('csrftoken');
}

const BASE = '/api/skills/';

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

export async function listSkills(): Promise<SkillsListResponse> {
  const r = await fetch(BASE, { credentials: 'include' });
  return jsonOrThrow<SkillsListResponse>(r);
}

export async function createSkill(payload: {
  name: string;
  body: string;
  enabled?: boolean;
}): Promise<Skill> {
  const r = await fetch(BASE, {
    method: 'POST',
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
      'X-CSRFToken': csrfToken(),
    },
    body: JSON.stringify(payload),
  });
  return jsonOrThrow<Skill>(r);
}

export async function updateSkill(
  id: number,
  payload: Partial<Pick<Skill, 'name' | 'body' | 'enabled'>>,
): Promise<Skill> {
  const r = await fetch(`${BASE}${id}/`, {
    method: 'PUT',
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
      'X-CSRFToken': csrfToken(),
    },
    body: JSON.stringify(payload),
  });
  return jsonOrThrow<Skill>(r);
}

export async function deleteSkill(id: number): Promise<void> {
  const r = await fetch(`${BASE}${id}/`, {
    method: 'DELETE',
    credentials: 'include',
    headers: { 'X-CSRFToken': csrfToken() },
  });
  if (!r.ok) {
    throw new Error(`Delete failed (${r.status})`);
  }
}

export interface CollectionSkill {
  collection_id: number;
  collection_name: string;
  body: string;
  updated_at: string | null;
  updated_by: string | null;
  exists: boolean;
  max_body_length: number;
}

export async function getCollectionSkill(collectionId: number): Promise<CollectionSkill> {
  const r = await fetch(`/api/collections/${collectionId}/skill/`, { credentials: 'include' });
  return jsonOrThrow<CollectionSkill>(r);
}

export async function saveCollectionSkill(
  collectionId: number,
  body: string,
): Promise<CollectionSkill> {
  const r = await fetch(`/api/collections/${collectionId}/skill/`, {
    method: 'PUT',
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
      'X-CSRFToken': csrfToken(),
    },
    body: JSON.stringify({ body }),
  });
  return jsonOrThrow<CollectionSkill>(r);
}
