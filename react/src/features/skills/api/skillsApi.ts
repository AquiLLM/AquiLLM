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

// ----- Phase 2: feedback → suggestion flow ---------------------------------

export interface PendingFeedback {
  message_id: number;
  message_uuid: string;
  conversation_id: number;
  conversation_name: string | null;
  rating: number | null;
  feedback_text: string;
  feedback_submitted_at: string | null;
  model: string | null;
  content_preview: string;
}

export interface SkillEditSuggestion {
  id: number;
  collection_id: number;
  source_message_id: number;
  notes_body_at_generation: string;
  proposed_body: string;
  status: 'pending' | 'accepted' | 'dismissed';
  generated_by: string | null;
  resolved_by: string | null;
  resolved_at: string | null;
  created_at: string | null;
}

export async function listPendingFeedback(collectionId: number): Promise<PendingFeedback[]> {
  const r = await fetch(`/api/collections/${collectionId}/pending-feedback/`, {
    credentials: 'include',
  });
  const data = await jsonOrThrow<{ items: PendingFeedback[] }>(r);
  return data.items;
}

export async function listSuggestions(collectionId: number): Promise<SkillEditSuggestion[]> {
  const r = await fetch(`/api/collections/${collectionId}/suggestions/`, {
    credentials: 'include',
  });
  const data = await jsonOrThrow<{ items: SkillEditSuggestion[] }>(r);
  return data.items;
}

export async function generateSuggestion(
  collectionId: number,
  messageId: number,
): Promise<SkillEditSuggestion> {
  const r = await fetch(`/api/collections/${collectionId}/suggestions/generate/`, {
    method: 'POST',
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
      'X-CSRFToken': csrfToken(),
    },
    body: JSON.stringify({ message_id: messageId }),
  });
  return jsonOrThrow<SkillEditSuggestion>(r);
}

export async function acceptSuggestion(
  suggestionId: number,
  bodyOverride?: string,
): Promise<SkillEditSuggestion> {
  const payload: Record<string, unknown> = {};
  if (bodyOverride !== undefined) payload.body = bodyOverride;
  const r = await fetch(`/api/suggestions/${suggestionId}/accept/`, {
    method: 'POST',
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
      'X-CSRFToken': csrfToken(),
    },
    body: JSON.stringify(payload),
  });
  return jsonOrThrow<SkillEditSuggestion>(r);
}

export async function dismissSuggestion(suggestionId: number): Promise<SkillEditSuggestion> {
  const r = await fetch(`/api/suggestions/${suggestionId}/dismiss/`, {
    method: 'POST',
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
      'X-CSRFToken': csrfToken(),
    },
  });
  return jsonOrThrow<SkillEditSuggestion>(r);
}

export async function dismissPendingFeedback(
  collectionId: number,
  messageId: number,
): Promise<void> {
  const r = await fetch(
    `/api/collections/${collectionId}/pending-feedback/${messageId}/dismiss/`,
    {
      method: 'POST',
      credentials: 'include',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': csrfToken(),
      },
    },
  );
  if (!r.ok) {
    let detail = '';
    try {
      detail = `: ${(await r.json())?.error || ''}`;
    } catch {
      // ignore
    }
    throw new Error(`Dismiss feedback failed (${r.status})${detail}`);
  }
}
