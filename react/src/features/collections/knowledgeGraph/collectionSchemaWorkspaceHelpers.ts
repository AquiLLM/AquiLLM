import type { CollectionSchemaEditorState } from './collectionSchemaReducer';
import type { CollectionSchemaEnvelope } from './schemaTypes';

export function definitionSource(envelope: CollectionSchemaEnvelope) {
  return envelope.draft ?? envelope.published;
}

export function workspacePhaseMessage(phase: CollectionSchemaEditorState['phase']): string | null {
  switch (phase) {
    case 'loading':
      return 'Loading schema workspace…';
    case 'unavailable':
      return 'Schema editor is unavailable for this collection.';
    case 'session_expired':
      return 'Your session expired. Sign in again to continue editing schema.';
    case 'forbidden':
      return 'You do not have permission to view this schema workspace.';
    default:
      return null;
  }
}
