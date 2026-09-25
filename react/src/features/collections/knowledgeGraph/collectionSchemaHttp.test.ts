// @vitest-environment jsdom

import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  MAX_SCHEMA_RESPONSE_BYTES,
  createCollectionSchemaHttp,
  readBoundedJson,
  schemaHttpUnavailable,
} from './collectionSchemaHttp';

afterEach(() => {
  vi.restoreAllMocks();
});

function jsonResponse(body: unknown, init: ResponseInit = {}): Response {
  return new Response(JSON.stringify(body), {
    status: init.status ?? 200,
    headers: { 'content-type': 'application/json', ...(init.headers as Record<string, string> | undefined) },
    ...init,
  });
}

describe('collectionSchemaHttp', () => {
  it('sends credentials and Accept on GET requests', async () => {
    const fetchFn = vi.fn().mockResolvedValue(jsonResponse({ ok: true }));
    const client = createCollectionSchemaHttp({ fetchFn, getCsrfToken: () => 'csrf-token' });
    await client.requestJson('/schema');
    const init = fetchFn.mock.calls[0]?.[1];
    expect(init.credentials).toBe('include');
    expect(new Headers(init.headers).get('Accept')).toBe('application/json');
    expect(new Headers(init.headers).get('X-CSRFToken')).toBeNull();
  });

  it('sends CSRF and JSON content type on mutations', async () => {
    const fetchFn = vi.fn().mockResolvedValue(jsonResponse({ ok: true }));
    const client = createCollectionSchemaHttp({ fetchFn, getCsrfToken: () => 'csrf-token' });
    await client.requestJson('/schema', { method: 'POST', body: { draft_id: 'd1' } });
    const headers = new Headers(fetchFn.mock.calls[0]?.[1].headers);
    expect(headers.get('Content-Type')).toBe('application/json');
    expect(headers.get('X-CSRFToken')).toBe('csrf-token');
  });

  it('sends If-Match revision header when provided', async () => {
    const fetchFn = vi.fn().mockResolvedValue(jsonResponse({ ok: true }));
    const client = createCollectionSchemaHttp({ fetchFn });
    await client.requestJson('/schema', { method: 'PUT', revision: 7, body: {} });
    expect(new Headers(fetchFn.mock.calls[0]?.[1].headers).get('If-Match')).toBe('7');
  });

  it('maps login redirects and 401 to session_expired', async () => {
    const fetchFn = vi.fn().mockResolvedValue(
      jsonResponse({ error: 'login' }, { status: 401, headers: { 'content-type': 'application/json' } }),
    );
    const client = createCollectionSchemaHttp({ fetchFn });
    const result = await client.requestJson('/schema');
    expect(result).toEqual({ ok: false, kind: 'session_expired' });
  });

  it('maps HTML responses to invalid_response', async () => {
    const fetchFn = vi.fn().mockResolvedValue(new Response('<html></html>', { headers: { 'content-type': 'text/html' } }));
    const client = createCollectionSchemaHttp({ fetchFn });
    const result = await client.requestJson('/schema');
    expect(result).toEqual({ ok: false, kind: 'invalid_response' });
  });

  it('maps 409 conflict payloads', async () => {
    const conflict = { attempted_revision: 4, current_revision: 6, draft_id: 'd1', definitions: [] };
    const fetchFn = vi.fn().mockResolvedValue(jsonResponse(conflict, { status: 409 }));
    const client = createCollectionSchemaHttp({ fetchFn });
    const result = await client.requestJson('/schema', { method: 'PUT', body: {} });
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.kind).toBe('revision_conflict');
      expect(result.conflict).toEqual(conflict);
    }
  });

  it('rejects oversized JSON responses', async () => {
    const huge = 'x'.repeat(MAX_SCHEMA_RESPONSE_BYTES + 1);
    const response = new Response(JSON.stringify({ blob: huge }), {
      headers: { 'content-type': 'application/json' },
    });
    const parsed = await readBoundedJson(response);
    expect(parsed).toBeNull();
  });

  it('propagates abort errors as network_error', async () => {
    const fetchFn = vi.fn().mockRejectedValue(new DOMException('Aborted', 'AbortError'));
    const client = createCollectionSchemaHttp({ fetchFn });
    const result = await client.requestJson('/schema', { signal: AbortSignal.abort() });
    expect(result).toEqual({ ok: false, kind: 'network_error' });
  });

  it('returns schema_unavailable helper without logging bodies', async () => {
    const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => undefined);
    expect(schemaHttpUnavailable()).toEqual({ ok: false, kind: 'schema_unavailable' });
    expect(consoleSpy).not.toHaveBeenCalled();
  });
});
