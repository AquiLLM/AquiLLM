import type {
  CollectionSchemaClientErrorKind,
  CollectionSchemaEnvelope,
  PublishOperation,
  PublishStatus,
  SchemaConflictInfo,
  SchemaDefinitionKind,
  SelectedSchemaDefinition,
  ValidationResult,
} from './schemaTypes';

export interface SchemaSelection {
  kind: SchemaDefinitionKind;
  key: string;
}

export interface CollectionSchemaEditorState {
  phase: 'initial' | 'loading' | 'ready' | 'unavailable' | 'forbidden' | 'session_expired';
  collectionId: string | null;
  envelope: CollectionSchemaEnvelope | null;
  requestGeneration: number;
  pendingOperation: 'load' | 'mutation' | 'validate' | 'publish' | 'discard' | 'restore' | null;
  validation: {
    status: 'idle' | 'pending' | 'valid' | 'invalid';
    result: ValidationResult | null;
    requestGeneration: number;
  };
  publish: { status: PublishStatus; operation: PublishOperation | null; requestGeneration: number };
  conflict: SchemaConflictInfo | null;
  selection: SchemaSelection | null;
}

export type CollectionSchemaAction =
  | { type: 'collection/changed'; collectionId: string; requestGeneration: number }
  | { type: 'load/started'; collectionId: string; requestGeneration: number }
  | { type: 'load/succeeded'; envelope: CollectionSchemaEnvelope; requestGeneration: number }
  | { type: 'load/failed'; kind: CollectionSchemaClientErrorKind; requestGeneration: number }
  | { type: 'mutation/started'; requestGeneration: number }
  | { type: 'mutation/succeeded'; envelope: CollectionSchemaEnvelope; requestGeneration: number }
  | { type: 'mutation/failed'; kind: CollectionSchemaClientErrorKind; requestGeneration: number }
  | { type: 'validate/started'; requestGeneration: number }
  | { type: 'validate/succeeded'; result: ValidationResult; requestGeneration: number }
  | { type: 'validate/failed'; kind: CollectionSchemaClientErrorKind; requestGeneration: number }
  | { type: 'publish/started'; operation: PublishOperation; requestGeneration: number }
  | { type: 'publish/polling'; operation: PublishOperation; requestGeneration: number }
  | { type: 'publish/succeeded'; envelope: CollectionSchemaEnvelope; requestGeneration: number }
  | { type: 'publish/failed'; requestGeneration: number }
  | { type: 'discard/started'; requestGeneration: number }
  | { type: 'discard/succeeded'; envelope: CollectionSchemaEnvelope; requestGeneration: number }
  | { type: 'restore/started'; requestGeneration: number }
  | { type: 'restore/succeeded'; envelope: CollectionSchemaEnvelope; requestGeneration: number }
  | { type: 'conflict/received'; conflict: SchemaConflictInfo; envelope: CollectionSchemaEnvelope; requestGeneration: number }
  | { type: 'selection/changed'; selection: SchemaSelection | null };

export function createInitialCollectionSchemaState(): CollectionSchemaEditorState {
  return {
    phase: 'initial',
    collectionId: null,
    envelope: null,
    requestGeneration: 0,
    pendingOperation: null,
    validation: { status: 'idle', result: null, requestGeneration: 0 },
    publish: { status: 'idle', operation: null, requestGeneration: 0 },
    conflict: null,
    selection: null,
  };
}

const definitionSource = (envelope: CollectionSchemaEnvelope) => envelope.draft ?? envelope.published;

const normalizeEnvelope = (envelope: CollectionSchemaEnvelope): CollectionSchemaEnvelope =>
  envelope.permissions.level === 'VIEW' ? { ...envelope, draft: null } : envelope;

const clearValidation = (state: CollectionSchemaEditorState) => ({
  status: 'idle' as const,
  result: null,
  requestGeneration: state.requestGeneration,
});

const applyEnvelope = (state: CollectionSchemaEditorState, envelope: CollectionSchemaEnvelope) => {
  const normalized = normalizeEnvelope(envelope);
  const source = definitionSource(normalized);
  const selection =
    state.selection &&
    (state.selection.kind === 'entity'
      ? source.entities.some((item) => item.key === state.selection!.key)
      : source.relations.some((item) => item.key === state.selection!.key))
      ? state.selection
      : null;
  return { envelope: normalized, selection, validation: clearValidation(state), conflict: null };
};

const hasValidationErrors = (result: ValidationResult) =>
  result.issues.some((issue) => issue.severity === 'error');

const phaseForError = (kind: CollectionSchemaClientErrorKind): CollectionSchemaEditorState['phase'] => {
  if (kind === 'forbidden') return 'forbidden';
  if (kind === 'session_expired') return 'session_expired';
  if (kind === 'schema_unavailable') return 'unavailable';
  return 'ready';
};

const isStale = (state: CollectionSchemaEditorState, requestGeneration: number) =>
  requestGeneration !== state.requestGeneration;

const startOperation = (
  state: CollectionSchemaEditorState,
  requestGeneration: number,
  pendingOperation: CollectionSchemaEditorState['pendingOperation'],
  extra?: Partial<CollectionSchemaEditorState>,
) => ({ ...state, requestGeneration, pendingOperation, ...extra });

export function collectionSchemaReducer(
  state: CollectionSchemaEditorState,
  action: CollectionSchemaAction,
): CollectionSchemaEditorState {
  switch (action.type) {
    case 'collection/changed':
      return { ...createInitialCollectionSchemaState(), collectionId: action.collectionId, requestGeneration: action.requestGeneration };
    case 'load/started':
      return startOperation({ ...state, phase: 'loading', collectionId: action.collectionId, conflict: null }, action.requestGeneration, 'load');
    case 'load/succeeded':
      if (isStale(state, action.requestGeneration)) return state;
      return { ...state, phase: 'ready', pendingOperation: null, ...applyEnvelope(state, action.envelope) };
    case 'load/failed':
      if (isStale(state, action.requestGeneration)) return state;
      return {
        ...state,
        phase: phaseForError(action.kind),
        pendingOperation: null,
        envelope: action.kind === 'forbidden' || action.kind === 'session_expired' ? state.envelope : null,
      };
    case 'mutation/started':
    case 'discard/started':
    case 'restore/started':
      return startOperation(state, action.requestGeneration, action.type === 'mutation/started' ? 'mutation' : action.type === 'discard/started' ? 'discard' : 'restore', {
        validation: clearValidation(state),
      });
    case 'mutation/succeeded':
    case 'discard/succeeded':
    case 'restore/succeeded':
      if (isStale(state, action.requestGeneration)) return state;
      return { ...state, pendingOperation: null, ...applyEnvelope(state, action.envelope) };
    case 'mutation/failed':
      if (isStale(state, action.requestGeneration)) return state;
      return { ...state, pendingOperation: null };
    case 'validate/started':
      return startOperation(state, action.requestGeneration, 'validate', {
        validation: { status: 'pending', result: null, requestGeneration: action.requestGeneration },
      });
    case 'validate/succeeded':
      if (isStale(state, action.requestGeneration)) return state;
      return {
        ...state,
        pendingOperation: null,
        validation: {
          status: hasValidationErrors(action.result) ? 'invalid' : 'valid',
          result: action.result,
          requestGeneration: action.requestGeneration,
        },
      };
    case 'validate/failed':
      if (isStale(state, action.requestGeneration)) return state;
      return { ...state, pendingOperation: null, validation: clearValidation(state) };
    case 'publish/started':
      return startOperation(state, action.requestGeneration, 'publish', {
        publish: { status: 'pending', operation: action.operation, requestGeneration: action.requestGeneration },
      });
    case 'publish/polling':
      if (isStale(state, action.requestGeneration)) return state;
      return { ...state, publish: { status: 'polling', operation: action.operation, requestGeneration: action.requestGeneration } };
    case 'publish/succeeded':
      if (isStale(state, action.requestGeneration)) return state;
      return {
        ...state,
        pendingOperation: null,
        publish: { status: 'succeeded', operation: null, requestGeneration: action.requestGeneration },
        ...applyEnvelope(state, action.envelope),
      };
    case 'publish/failed':
      if (isStale(state, action.requestGeneration)) return state;
      return {
        ...state,
        pendingOperation: null,
        publish: { status: 'failed', operation: state.publish.operation, requestGeneration: action.requestGeneration },
      };
    case 'conflict/received':
      if (isStale(state, action.requestGeneration)) return state;
      return { ...state, pendingOperation: null, ...applyEnvelope(state, action.envelope), conflict: action.conflict };
    case 'selection/changed':
      return { ...state, selection: action.selection };
    default:
      return state;
  }
}

export function canValidate(state: CollectionSchemaEditorState): boolean {
  return Boolean(
    state.phase === 'ready' &&
      state.envelope?.draft &&
      state.envelope.permissions.can_validate &&
      state.pendingOperation !== 'validate' &&
      state.validation.status !== 'pending',
  );
}

export function canPublish(state: CollectionSchemaEditorState): boolean {
  const { envelope, validation: validationState, pendingOperation } = state;
  const result = validationState.result;
  const draft = envelope?.draft;
  if (!draft || !envelope?.permissions.can_publish || !result || validationState.status !== 'valid') return false;
  if (pendingOperation === 'publish' || hasValidationErrors(result)) return false;
  const identity = result.identity;
  return (
    draft.draft_id === identity.draft_id &&
    draft.revision === identity.revision &&
    identity.candidate_checksum.length > 0
  );
}

export function selectedDefinition(state: CollectionSchemaEditorState): SelectedSchemaDefinition | null {
  const { envelope, selection } = state;
  if (!envelope || !selection) return null;
  const source = definitionSource(envelope);
  if (selection.kind === 'entity') {
    const definition = source.entities.find((item) => item.key === selection.key);
    return definition ? { kind: 'entity', definition } : null;
  }
  const definition = source.relations.find((item) => item.key === selection.key);
  return definition ? { kind: 'relation', definition } : null;
}
