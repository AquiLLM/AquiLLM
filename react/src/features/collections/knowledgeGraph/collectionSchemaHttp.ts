import type { CollectionSchemaClientErrorKind, SchemaConflictInfo } from './schemaTypes';

export const MAX_SCHEMA_RESPONSE_BYTES = 512 * 1024;
export const MAX_SCHEMA_JSON_DEPTH = 32;
export const MAX_SCHEMA_ARRAY_ITEMS = 500;

export type CollectionSchemaHttpResult<T> =
  | { ok: true; data: T }
  | { ok: false; kind: CollectionSchemaClientErrorKind; conflict?: SchemaConflictInfo };

export interface CollectionSchemaRequestOptions {
  method?: string;
  body?: unknown;
  revision?: number;
  headers?: HeadersInit;
  signal?: AbortSignal;
}

export interface CollectionSchemaHttpClient {
  requestJson<T>(
    url: string,
    init?: CollectionSchemaRequestOptions,
  ): Promise<CollectionSchemaHttpResult<T>>;
}

export interface CollectionSchemaHttpOptions {
  fetchFn?: typeof fetch;
  getCsrfToken?: () => string;
}

const MUTATION_METHODS = new Set(['POST', 'PUT', 'PATCH', 'DELETE']);

function isJsonContentType(contentType: string | null): boolean {
  if (!contentType) return false;
  const normalized = contentType.split(';', 1)[0]?.trim().toLowerCase();
  return normalized === 'application/json';
}

function mapStatusToKind(status: number): CollectionSchemaClientErrorKind {
  if (status === 401) return 'session_expired';
  if (status === 403) return 'forbidden';
  if (status === 404) return 'not_found';
  if (status === 409) return 'revision_conflict';
  if (status === 422) return 'validation_failed';
  if (status === 429) return 'rate_limited';
  if (status >= 500) return 'server_error';
  return 'invalid_response';
}

function exceedsDepth(value: unknown, depth = 0): boolean {
  if (depth > MAX_SCHEMA_JSON_DEPTH) return true;
  if (value === null || typeof value !== 'object') return false;
  if (Array.isArray(value)) {
    if (value.length > MAX_SCHEMA_ARRAY_ITEMS) return true;
    return value.some((item) => exceedsDepth(item, depth + 1));
  }
  return Object.values(value as Record<string, unknown>).some((item) => exceedsDepth(item, depth + 1));
}

export async function readBoundedJson(response: Response): Promise<unknown | null> {
  const reader = response.body?.getReader();
  if (!reader) {
    const text = await response.text();
    if (text.length > MAX_SCHEMA_RESPONSE_BYTES) return null;
    try {
      const parsed = JSON.parse(text);
      return exceedsDepth(parsed) ? null : parsed;
    } catch {
      return null;
    }
  }

  const chunks: Uint8Array[] = [];
  let total = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (!value) continue;
    total += value.byteLength;
    if (total > MAX_SCHEMA_RESPONSE_BYTES) return null;
    chunks.push(value);
  }

  const merged = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.byteLength;
  }

  try {
    const parsed = JSON.parse(new TextDecoder().decode(merged));
    return exceedsDepth(parsed) ? null : parsed;
  } catch {
    return null;
  }
}

function buildHeaders(
  method: string,
  getCsrfToken: () => string,
  revision?: number,
  extra?: HeadersInit,
): Headers {
  const headers = new Headers(extra);
  headers.set('Accept', 'application/json');
  if (MUTATION_METHODS.has(method)) {
    headers.set('Content-Type', 'application/json');
    headers.set('X-CSRFToken', getCsrfToken());
  }
  if (revision !== undefined) {
    headers.set('If-Match', String(revision));
  }
  return headers;
}

export function createCollectionSchemaHttp(
  options: CollectionSchemaHttpOptions = {},
): CollectionSchemaHttpClient {
  const fetchFn = options.fetchFn ?? fetch;
  const getCsrfToken = options.getCsrfToken ?? (() => '');

  return {
    async requestJson<T>(url: string, init: CollectionSchemaRequestOptions = {}) {
      const method = (init.method ?? 'GET').toUpperCase();
      const headers = buildHeaders(method, getCsrfToken, init.revision, init.headers);
      let response: Response;
      try {
        response = await fetchFn(url, {
          method,
          headers,
          credentials: 'include',
          signal: init.signal,
          body:
            init.body === undefined
              ? undefined
              : typeof init.body === 'string'
                ? init.body
                : JSON.stringify(init.body),
        });
      } catch {
        return { ok: false, kind: 'network_error' };
      }

      if (response.redirected && response.url.includes('/login')) {
        return { ok: false, kind: 'session_expired' };
      }

      const contentType = response.headers.get('content-type');
      if (!isJsonContentType(contentType)) {
        return { ok: false, kind: response.status === 401 ? 'session_expired' : 'invalid_response' };
      }

      const payload = await readBoundedJson(response);
      if (payload === null) {
        return { ok: false, kind: 'invalid_response' };
      }

      if (!response.ok) {
        const kind = mapStatusToKind(response.status);
        if (kind === 'revision_conflict' && payload && typeof payload === 'object') {
          return { ok: false, kind, conflict: payload as SchemaConflictInfo };
        }
        return { ok: false, kind };
      }

      return { ok: true, data: payload as T };
    },
  };
}

export function schemaHttpUnavailable<T>(): CollectionSchemaHttpResult<T> {
  return { ok: false, kind: 'schema_unavailable' };
}
