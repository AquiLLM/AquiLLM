// @vitest-environment jsdom

import { describe, expect, it } from 'vitest';

import {
  REQUIRED_SCHEMA_API_CAPABILITIES,
  SCHEMA_API_URL_KEYS,
  readCollectionSchemaApiRoutes,
} from './schemaApiRoutes';

function completeApiUrls(): Record<string, string> {
  return {
    [SCHEMA_API_URL_KEYS.workspace]: '/api/collection/%(col_id)s/schema/',
    [SCHEMA_API_URL_KEYS.createDraft]: '/api/collection/%(col_id)s/schema/draft/',
    [SCHEMA_API_URL_KEYS.entity]: '/api/collection/%(col_id)s/schema/entity/%(entity_key)s/',
    [SCHEMA_API_URL_KEYS.relation]: '/api/collection/%(col_id)s/schema/relation/%(relation_key)s/',
    [SCHEMA_API_URL_KEYS.validate]: '/api/collection/%(col_id)s/schema/validate/',
    [SCHEMA_API_URL_KEYS.diff]: '/api/collection/%(col_id)s/schema/diff/',
    [SCHEMA_API_URL_KEYS.publish]: '/api/collection/%(col_id)s/schema/publish/',
    [SCHEMA_API_URL_KEYS.discard]: '/api/collection/%(col_id)s/schema/discard/',
    [SCHEMA_API_URL_KEYS.versions]: '/api/collection/%(col_id)s/schema/versions/',
    [SCHEMA_API_URL_KEYS.versionDiff]: '/api/collection/%(col_id)s/schema/versions/%(version_id)s/diff/',
    [SCHEMA_API_URL_KEYS.restore]: '/api/collection/%(col_id)s/schema/versions/%(version_id)s/restore/',
    [SCHEMA_API_URL_KEYS.restoreReplace]: '/api/collection/%(col_id)s/schema/restore-replace/',
  };
}

describe('readCollectionSchemaApiRoutes complete map', () => {
  it('returns all required capabilities for a complete apiUrls map', () => {
    const result = readCollectionSchemaApiRoutes(completeApiUrls());
    expect(result.available).toBe(true);
    if (result.available) {
      for (const capability of REQUIRED_SCHEMA_API_CAPABILITIES) {
        expect(result.routes[capability]).toContain('%(col_id)s');
      }
    }
  });

  it('lists missing capabilities when a required key is absent', () => {
    const urls = completeApiUrls();
    delete urls[SCHEMA_API_URL_KEYS.validate];
    const result = readCollectionSchemaApiRoutes(urls);
    expect(result).toEqual({ available: false, missing: ['validate'] });
  });

  it('does not require publishStatus for synchronous publish contract', () => {
    const result = readCollectionSchemaApiRoutes(completeApiUrls());
    expect(result.available).toBe(true);
    expect('api_collection_schema_publish_status' in completeApiUrls()).toBe(false);
    if (result.available) {
      expect('publishStatus' in result.routes).toBe(false);
    }
  });
});
