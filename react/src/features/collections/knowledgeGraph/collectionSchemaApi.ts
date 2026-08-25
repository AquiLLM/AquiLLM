import type { CollectionSchemaHttpClient, CollectionSchemaHttpResult } from './collectionSchemaHttp';
import { schemaHttpUnavailable } from './collectionSchemaHttp';
import type { CollectionSchemaApiRouteMap } from './schemaApiRoutes';
import { formatCollectionSchemaRoute, readCollectionSchemaApiRoutes } from './schemaApiRoutes';
import type {
  CollectionSchemaEnvelope,
  EntityTypeDefinition,
  PublishOperation,
  RelationTypeDefinition,
  SchemaDiffSummary,
  SchemaGenerationStart,
  SchemaGenerationStatus,
  SchemaHistoryPage,
  ValidationResult,
} from './schemaTypes';

export interface CollectionSchemaApi {
  loadWorkspace(collectionId: string): Promise<CollectionSchemaHttpResult<CollectionSchemaEnvelope>>;
  createDraft(collectionId: string): Promise<CollectionSchemaHttpResult<CollectionSchemaEnvelope>>;
  upsertEntity(
    collectionId: string,
    entityKey: string,
    revision: number,
    values: Record<string, unknown>,
  ): Promise<CollectionSchemaHttpResult<CollectionSchemaEnvelope>>;
  deleteEntity(
    collectionId: string,
    entityKey: string,
    revision: number,
  ): Promise<CollectionSchemaHttpResult<CollectionSchemaEnvelope>>;
  upsertRelation(
    collectionId: string,
    relationKey: string,
    revision: number,
    values: Record<string, unknown>,
  ): Promise<CollectionSchemaHttpResult<CollectionSchemaEnvelope>>;
  deleteRelation(
    collectionId: string,
    relationKey: string,
    revision: number,
  ): Promise<CollectionSchemaHttpResult<CollectionSchemaEnvelope>>;
  validate(
    collectionId: string,
    draftId: string,
    revision: number,
  ): Promise<CollectionSchemaHttpResult<ValidationResult>>;
  fetchDiff(collectionId: string): Promise<CollectionSchemaHttpResult<SchemaDiffSummary>>;
  publish(
    collectionId: string,
    operation: PublishOperation,
    revision: number,
  ): Promise<CollectionSchemaHttpResult<CollectionSchemaEnvelope>>;
  discardDraft(
    collectionId: string,
    draftId: string,
    revision: number,
  ): Promise<CollectionSchemaHttpResult<CollectionSchemaEnvelope>>;
  listVersions(
    collectionId: string,
    cursor?: string | null,
  ): Promise<CollectionSchemaHttpResult<SchemaHistoryPage>>;
  fetchVersionDiff(
    collectionId: string,
    versionId: number,
  ): Promise<CollectionSchemaHttpResult<SchemaDiffSummary>>;
  restoreVersion(
    collectionId: string,
    versionId: number,
  ): Promise<CollectionSchemaHttpResult<CollectionSchemaEnvelope>>;
  restoreReplace(
    collectionId: string,
    versionId: number,
    challengeToken: string,
    existingDraftRevision: number,
  ): Promise<CollectionSchemaHttpResult<CollectionSchemaEnvelope>>;
  startGeneration(collectionId: string): Promise<CollectionSchemaHttpResult<SchemaGenerationStart>>;
  getGenerationStatus(
    collectionId: string,
    runId: string,
    signal?: AbortSignal,
  ): Promise<CollectionSchemaHttpResult<SchemaGenerationStatus>>;
}

function routeUrl(pattern: string, params: Record<string, string | number>): string {
  return formatCollectionSchemaRoute(pattern, params);
}

function normalizeEnvelope(raw: CollectionSchemaEnvelope): CollectionSchemaEnvelope {
  if (raw.permissions.level === 'VIEW') {
    return { ...raw, draft: null };
  }
  return raw;
}

function asEnvelope(result: CollectionSchemaHttpResult<CollectionSchemaEnvelope>) {
  if (!result.ok) return result;
  return { ok: true as const, data: normalizeEnvelope(result.data) };
}

function parseDefinitionList<T>(value: unknown): T[] {
  return Array.isArray(value) ? (value as T[]) : [];
}

function parseEnvelope(raw: unknown): CollectionSchemaEnvelope | null {
  if (!raw || typeof raw !== 'object') return null;
  const envelope = raw as CollectionSchemaEnvelope;
  if (!envelope.permissions || !envelope.published) return null;
  envelope.published.entities = parseDefinitionList<EntityTypeDefinition>(envelope.published.entities);
  envelope.published.relations = parseDefinitionList<RelationTypeDefinition>(envelope.published.relations);
  if (envelope.draft) {
    envelope.draft.entities = parseDefinitionList<EntityTypeDefinition>(envelope.draft.entities);
    envelope.draft.relations = parseDefinitionList<RelationTypeDefinition>(envelope.draft.relations);
  }
  return envelope;
}

async function envelopeRequest(
  client: CollectionSchemaHttpClient,
  url: string,
  init?: Parameters<CollectionSchemaHttpClient['requestJson']>[1],
): Promise<CollectionSchemaHttpResult<CollectionSchemaEnvelope>> {
  const result = await client.requestJson<CollectionSchemaEnvelope>(url, init);
  if (!result.ok) return result;
  const parsed = parseEnvelope(result.data);
  if (!parsed) return { ok: false, kind: 'invalid_response' };
  return { ok: true, data: parsed };
}

export function createCollectionSchemaApi(
  routes: CollectionSchemaApiRouteMap | null,
  httpClient: CollectionSchemaHttpClient,
): CollectionSchemaApi {
  if (!routes) {
    const unavailable = async () => schemaHttpUnavailable<CollectionSchemaEnvelope>();
    return {
      loadWorkspace: unavailable,
      createDraft: unavailable,
      upsertEntity: unavailable,
      deleteEntity: unavailable,
      upsertRelation: unavailable,
      deleteRelation: unavailable,
      validate: unavailable as CollectionSchemaApi['validate'],
      fetchDiff: unavailable as CollectionSchemaApi['fetchDiff'],
      publish: unavailable,
      discardDraft: unavailable,
      listVersions: unavailable as CollectionSchemaApi['listVersions'],
      fetchVersionDiff: unavailable as CollectionSchemaApi['fetchVersionDiff'],
      restoreVersion: unavailable,
      restoreReplace: unavailable,
      startGeneration: unavailable as CollectionSchemaApi['startGeneration'],
      getGenerationStatus: unavailable as CollectionSchemaApi['getGenerationStatus'],
    };
  }

  const col = (collectionId: string) => ({ col_id: collectionId });

  return {
    async loadWorkspace(collectionId) {
      return asEnvelope(await envelopeRequest(httpClient, routeUrl(routes.workspace, col(collectionId))));
    },
    async createDraft(collectionId) {
      return asEnvelope(
        await envelopeRequest(httpClient, routeUrl(routes.createDraft, col(collectionId)), { method: 'POST', body: {} }),
      );
    },
    async upsertEntity(collectionId, entityKey, revision, values) {
      return asEnvelope(
        await envelopeRequest(httpClient, routeUrl(routes.entity, { ...col(collectionId), entity_key: entityKey }), {
          method: 'PUT',
          revision,
          body: { values },
        }),
      );
    },
    async deleteEntity(collectionId, entityKey, revision) {
      return asEnvelope(
        await envelopeRequest(httpClient, routeUrl(routes.entity, { ...col(collectionId), entity_key: entityKey }), {
          method: 'DELETE',
          revision,
        }),
      );
    },
    async upsertRelation(collectionId, relationKey, revision, values) {
      return asEnvelope(
        await envelopeRequest(
          httpClient,
          routeUrl(routes.relation, { ...col(collectionId), relation_key: relationKey }),
          { method: 'PUT', revision, body: { values } },
        ),
      );
    },
    async deleteRelation(collectionId, relationKey, revision) {
      return asEnvelope(
        await envelopeRequest(
          httpClient,
          routeUrl(routes.relation, { ...col(collectionId), relation_key: relationKey }),
          { method: 'DELETE', revision },
        ),
      );
    },
    async validate(collectionId, draftId, revision) {
      return httpClient.requestJson<ValidationResult>(routeUrl(routes.validate, col(collectionId)), {
        method: 'POST',
        body: { draft_id: draftId, revision },
      });
    },
    async fetchDiff(collectionId) {
      return httpClient.requestJson<SchemaDiffSummary>(routeUrl(routes.diff, col(collectionId)));
    },
    async publish(collectionId, operation, revision) {
      return asEnvelope(
        await envelopeRequest(httpClient, routeUrl(routes.publish, col(collectionId)), {
          method: 'POST',
          revision,
          body: operation,
        }),
      );
    },
    async discardDraft(collectionId, draftId, revision) {
      return asEnvelope(
        await envelopeRequest(httpClient, routeUrl(routes.discard, col(collectionId)), {
          method: 'POST',
          revision,
          body: { draft_id: draftId, revision },
        }),
      );
    },
    async listVersions(collectionId, cursor) {
      const base = routeUrl(routes.versions, col(collectionId));
      const url = cursor ? `${base}?cursor=${encodeURIComponent(cursor)}` : base;
      return httpClient.requestJson<SchemaHistoryPage>(url);
    },
    async fetchVersionDiff(collectionId, versionId) {
      return httpClient.requestJson<SchemaDiffSummary>(
        routeUrl(routes.versionDiff, { ...col(collectionId), version_id: versionId }),
      );
    },
    async restoreVersion(collectionId, versionId) {
      return asEnvelope(
        await envelopeRequest(httpClient, routeUrl(routes.restore, { ...col(collectionId), version_id: versionId }), {
          method: 'POST',
          body: {},
        }),
      );
    },
    async restoreReplace(collectionId, versionId, challengeToken, existingDraftRevision) {
      return asEnvelope(
        await envelopeRequest(httpClient, routeUrl(routes.restoreReplace, col(collectionId)), {
          method: 'POST',
          revision: existingDraftRevision,
          body: { version_id: versionId, challenge_token: challengeToken, existing_draft_revision: existingDraftRevision },
        }),
      );
    },
    async startGeneration(collectionId) {
      return httpClient.requestJson<SchemaGenerationStart>(routeUrl(routes.generate, col(collectionId)), {
        method: 'POST',
        body: {},
      });
    },
    async getGenerationStatus(collectionId, runId, signal) {
      return httpClient.requestJson<SchemaGenerationStatus>(
        routeUrl(routes.generationStatus, { ...col(collectionId), run_id: runId }),
        { signal },
      );
    },
  };
}

export function createCollectionSchemaApiFromWindow(
  apiUrls: Record<string, string | undefined>,
  httpClient: CollectionSchemaHttpClient,
): CollectionSchemaApi {
  const result = readCollectionSchemaApiRoutes(apiUrls);
  return createCollectionSchemaApi(result.available ? result.routes : null, httpClient);
}
