// @vitest-environment jsdom

import { describe, expect, it, vi } from 'vitest';

import { createCollectionSchemaApi } from './collectionSchemaApi';
import type { CollectionSchemaHttpClient, CollectionSchemaRequestOptions } from './collectionSchemaHttp';
import type { CollectionSchemaApiRouteMap } from './schemaApiRoutes';
import {
  editDraftEnvelope,
  manageDraftEnvelope,
  validationResultFixture,
  viewPublishedEnvelope,
} from './schemaTestFixtures';

const routes: CollectionSchemaApiRouteMap = {
  workspace: '/api/collection/%(col_id)s/schema/',
  createDraft: '/api/collection/%(col_id)s/schema/draft/',
  entity: '/api/collection/%(col_id)s/schema/entity/%(entity_key)s/',
  relation: '/api/collection/%(col_id)s/schema/relation/%(relation_key)s/',
  validate: '/api/collection/%(col_id)s/schema/validate/',
  diff: '/api/collection/%(col_id)s/schema/diff/',
  publish: '/api/collection/%(col_id)s/schema/publish/',
  discard: '/api/collection/%(col_id)s/schema/discard/',
  versions: '/api/collection/%(col_id)s/schema/versions/',
  versionDiff: '/api/collection/%(col_id)s/schema/versions/%(version_id)s/diff/',
  restore: '/api/collection/%(col_id)s/schema/versions/%(version_id)s/restore/',
  restoreReplace: '/api/collection/%(col_id)s/schema/restore-replace/',
  generate: '/api/collection/%(col_id)s/schema/generate/',
  generationStatus: '/api/collection/%(col_id)s/schema/generation/%(run_id)s/',
};

function createMockClient(): CollectionSchemaHttpClient & {
  calls: Array<{ url: string; init?: CollectionSchemaRequestOptions }>;
  requestJson: CollectionSchemaHttpClient['requestJson'];
} {
  const calls: Array<{ url: string; init?: CollectionSchemaRequestOptions }> = [];
  const requestJson = vi.fn(async (url: string, init?: CollectionSchemaRequestOptions) => {
    calls.push({ url, init });
    return { ok: true as const, data: manageDraftEnvelope as never };
  }) as CollectionSchemaHttpClient['requestJson'];
  return { calls, requestJson };
}

describe('createCollectionSchemaApi', () => {
  it('returns schema_unavailable before fetch when routes are missing', async () => {
    const client = createMockClient();
    const api = createCollectionSchemaApi(null, client);
    const result = await api.loadWorkspace('42');
    expect(result).toEqual({ ok: false, kind: 'schema_unavailable' });
    expect(client.requestJson).not.toHaveBeenCalled();
  });

  it('formats collection placeholders for workspace load', async () => {
    const client = createMockClient();
    const api = createCollectionSchemaApi(routes, client);
    await api.loadWorkspace('42');
    expect(client.calls[0]?.url).toBe('/api/collection/42/schema/');
  });

  it('sends validate draft identity in POST body', async () => {
    const client = createMockClient();
    client.requestJson = vi.fn(async (url: string, init?: CollectionSchemaRequestOptions) => {
      client.calls.push({ url, init });
      return { ok: true as const, data: validationResultFixture as never };
    }) as CollectionSchemaHttpClient['requestJson'];
    const api = createCollectionSchemaApi(routes, client);
    await api.validate('7', 'draft-manage-1', 5);
    expect(client.calls[0]?.init).toMatchObject({
      method: 'POST',
      body: { draft_id: 'draft-manage-1', revision: 5 },
    });
  });

  it('sends publish operation with If-Match revision', async () => {
    const client = createMockClient();
    const api = createCollectionSchemaApi(routes, client);
    await api.publish(
      '7',
      {
        draft_id: 'draft-manage-1',
        revision: 5,
        candidate_checksum: 'candidate-checksum-v5',
        validation_result_id: 'validation-result-1',
      },
      5,
    );
    expect(client.calls[0]?.url).toBe('/api/collection/7/schema/publish/');
    expect(client.calls[0]?.init).toMatchObject({
      method: 'POST',
      revision: 5,
      body: {
        draft_id: 'draft-manage-1',
        revision: 5,
        candidate_checksum: 'candidate-checksum-v5',
        validation_result_id: 'validation-result-1',
      },
    });
  });

  it('sends restore replace challenge token and existing draft revision', async () => {
    const client = createMockClient();
    const api = createCollectionSchemaApi(routes, client);
    await api.restoreReplace('7', 4, 'restore-challenge-token', 5);
    expect(client.calls[0]?.init).toMatchObject({
      method: 'POST',
      revision: 5,
      body: {
        version_id: 4,
        challenge_token: 'restore-challenge-token',
        existing_draft_revision: 5,
      },
    });
  });

  it('normalizes VIEW envelopes without draft content', async () => {
    const client = createMockClient();
    client.requestJson = vi.fn(async () => ({ ok: true as const, data: viewPublishedEnvelope as never })) as CollectionSchemaHttpClient['requestJson'];
    const api = createCollectionSchemaApi(routes, client);
    const result = await api.loadWorkspace('1');
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.data.draft).toBeNull();
    }
  });

  it('maps entity upsert to formatted entity route with revision', async () => {
    const client = createMockClient();
    const api = createCollectionSchemaApi(routes, client);
    await api.upsertEntity('9', 'person', 3, { description: 'Updated' });
    expect(client.calls[0]?.url).toBe('/api/collection/9/schema/entity/person/');
    expect(client.calls[0]?.init).toMatchObject({ method: 'PUT', revision: 3, body: { values: { description: 'Updated' } } });
  });

  it('returns complete normalized envelopes from mutation methods', async () => {
    const client = createMockClient();
    client.requestJson = vi.fn(async () => ({ ok: true as const, data: editDraftEnvelope as never })) as CollectionSchemaHttpClient['requestJson'];
    const api = createCollectionSchemaApi(routes, client);
    const result = await api.createDraft('col-edit');
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.data.draft?.draft_id).toBe('draft-edit-1');
    }
  });

  it('starts collection schema generation with a formatted POST request', async () => {
    const client = createMockClient();
    client.requestJson = vi.fn(async (url: string, init?: CollectionSchemaRequestOptions) => {
      client.calls.push({ url, init });
      return {
        ok: true as const,
        data: { run_id: 'run-1', status: 'queued', status_url: '/api/collection/7/schema/generation/run-1/' },
      } as never;
    }) as CollectionSchemaHttpClient['requestJson'];
    const api = createCollectionSchemaApi(routes, client);

    const result = await api.startGeneration('7');

    expect(result).toEqual({
      ok: true,
      data: { run_id: 'run-1', status: 'queued', status_url: '/api/collection/7/schema/generation/run-1/' },
    });
    expect(client.calls[0]).toMatchObject({
      url: '/api/collection/7/schema/generate/',
      init: { method: 'POST', body: {} },
    });
  });

  it('gets a UUID generation status from its formatted route', async () => {
    const client = createMockClient();
    client.requestJson = vi.fn(async (url: string, init?: CollectionSchemaRequestOptions) => {
      client.calls.push({ url, init });
      return {
        ok: true as const,
        data: { run_id: '44c4f7a2-50e6-42a2-a45a-45c4ca38e580', status: 'running', error_code: null, statistics: {} },
      } as never;
    }) as CollectionSchemaHttpClient['requestJson'];
    const api = createCollectionSchemaApi(routes, client);

    const result = await api.getGenerationStatus('7', '44c4f7a2-50e6-42a2-a45a-45c4ca38e580');

    expect(result.ok).toBe(true);
    expect(client.calls[0]).toMatchObject({
      url: '/api/collection/7/schema/generation/44c4f7a2-50e6-42a2-a45a-45c4ca38e580/',
      init: { signal: undefined },
    });
  });
});
