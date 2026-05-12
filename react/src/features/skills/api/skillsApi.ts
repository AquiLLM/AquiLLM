import { getCsrfCookie } from '../../../main';
import type { Skill, SkillsListResponse } from '../types';

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
      'X-CSRFToken': getCsrfCookie(),
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
      'X-CSRFToken': getCsrfCookie(),
    },
    body: JSON.stringify(payload),
  });
  return jsonOrThrow<Skill>(r);
}

export async function deleteSkill(id: number): Promise<void> {
  const r = await fetch(`${BASE}${id}/`, {
    method: 'DELETE',
    credentials: 'include',
    headers: { 'X-CSRFToken': getCsrfCookie() },
  });
  if (!r.ok) {
    throw new Error(`Delete failed (${r.status})`);
  }
}
