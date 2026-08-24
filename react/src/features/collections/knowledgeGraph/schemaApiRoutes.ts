export type CollectionSchemaApiCapability =
  | 'workspace'
  | 'createDraft'
  | 'entity'
  | 'relation'
  | 'validate'
  | 'diff'
  | 'publish'
  | 'discard'
  | 'versions'
  | 'versionDiff'
  | 'restore'
  | 'restoreReplace'
  | 'generate'
  | 'generationStatus';

export const SCHEMA_API_URL_KEYS: Record<CollectionSchemaApiCapability, string> = {
  workspace: 'api_collection_schema_workspace',
  createDraft: 'api_collection_schema_draft',
  entity: 'api_collection_schema_entity',
  relation: 'api_collection_schema_relation',
  validate: 'api_collection_schema_validate',
  diff: 'api_collection_schema_diff',
  publish: 'api_collection_schema_publish',
  discard: 'api_collection_schema_discard',
  versions: 'api_collection_schema_versions',
  versionDiff: 'api_collection_schema_version_diff',
  restore: 'api_collection_schema_restore',
  restoreReplace: 'api_collection_schema_restore_replace',
  generate: 'api_collection_schema_generate',
  generationStatus: 'api_collection_schema_generation_status',
};

export const REQUIRED_SCHEMA_API_CAPABILITIES: CollectionSchemaApiCapability[] = [
  'workspace',
  'createDraft',
  'entity',
  'relation',
  'validate',
  'diff',
  'publish',
  'discard',
  'versions',
  'versionDiff',
  'restore',
  'restoreReplace',
  'generate',
  'generationStatus',
];

/** Sync publish contract: no publishStatus route is required or exposed. */
export const OPTIONAL_SCHEMA_API_CAPABILITIES = [] as const;

export type CollectionSchemaApiRouteMap = Record<CollectionSchemaApiCapability, string>;

export type ReadCollectionSchemaApiRoutesResult =
  | { available: true; routes: CollectionSchemaApiRouteMap }
  | { available: false; missing: CollectionSchemaApiCapability[] };

export function readCollectionSchemaApiRoutes(
  apiUrls: Record<string, string | undefined>,
): ReadCollectionSchemaApiRoutesResult {
  const missing: CollectionSchemaApiCapability[] = [];
  const routes = {} as CollectionSchemaApiRouteMap;

  for (const capability of REQUIRED_SCHEMA_API_CAPABILITIES) {
    const key = SCHEMA_API_URL_KEYS[capability];
    const url = apiUrls[key];
    if (!url) {
      missing.push(capability);
      continue;
    }
    routes[capability] = url;
  }

  if (missing.length > 0) {
    return { available: false, missing };
  }

  return { available: true, routes };
}

export function formatCollectionSchemaRoute(
  pattern: string,
  params: Record<string, string | number>,
): string {
  return pattern.replace(/%\((\w+)\)s/g, (_, key: string) => {
    const value = params[key];
    if (value === undefined) {
      throw new Error(`Missing parameter: ${key}`);
    }
    return String(value);
  });
}
