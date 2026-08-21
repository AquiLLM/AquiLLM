import { getCookie } from '../../../utils/csrf';
import { createCollectionSchemaApi, type CollectionSchemaApi } from './collectionSchemaApi';
import { createCollectionSchemaHttp } from './collectionSchemaHttp';
import { readCollectionSchemaApiRoutes } from './schemaApiRoutes';
import type { EntityTypeDefinition, RelationTypeDefinition, SchemaDefinitionKind } from './schemaTypes';
import { definitionSource } from './collectionSchemaWorkspaceHelpers';
import type { CollectionSchemaEditorState } from './collectionSchemaReducer';
import {
  applyReviewedRebase,
  previewReviewedRebase,
  type SchemaFormBufferState,
} from './schemaFormBuffer';

export function definitionValues(
  kind: SchemaDefinitionKind,
  definition: EntityTypeDefinition | RelationTypeDefinition,
): Record<string, unknown> {
  void kind;
  return { ...definition.values } as unknown as Record<string, unknown>;
}

export function createDefaultCollectionSchemaApi(): CollectionSchemaApi {
  const routes = readCollectionSchemaApiRoutes(window.apiUrls ?? {});
  const http = createCollectionSchemaHttp({ getCsrfToken: () => getCookie('csrftoken') });
  return createCollectionSchemaApi(routes.available ? routes.routes : null, http);
}

export function buildConflictReapplyUpdate(
  state: CollectionSchemaEditorState,
  buffer: SchemaFormBufferState,
  resolutions: Parameters<typeof applyReviewedRebase>[1],
) {
  const envelope = state.envelope;
  const conflict = state.conflict;
  if (
    !conflict ||
    !envelope?.draft ||
    !buffer.definitionKey ||
    !buffer.definitionKind ||
    !buffer.initialValues ||
    !buffer.currentValues
  ) {
    return null;
  }
  const source = definitionSource(envelope);
  const list = buffer.definitionKind === 'entity' ? source.entities : source.relations;
  const latest = list.find((item) => item.key === buffer.definitionKey);
  if (!latest) return null;
  const preview = previewReviewedRebase(
    buffer.initialValues,
    buffer.currentValues,
    latest.values as unknown as Record<string, unknown>,
  );
  const latestRevision = conflict.current_revision;
  const rebased = applyReviewedRebase(
    preview,
    resolutions,
    latestRevision,
    latest.values as unknown as Record<string, unknown>,
  );
  return {
    formReload: { baseRevision: latestRevision, values: rebased.values },
    envelope: {
      ...envelope,
      draft: { ...envelope.draft, revision: latestRevision },
    },
    requestGeneration: state.requestGeneration,
  };
}
