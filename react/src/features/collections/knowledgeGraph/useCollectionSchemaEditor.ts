import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from 'react';
import {
  buildConflictReapplyUpdate,
  createDefaultCollectionSchemaApi,
  definitionValues,
} from './collectionSchemaEditorHelpers';
import type { CollectionSchemaApi } from './collectionSchemaApi';
import { initialGenerationState, useSchemaGeneration } from './useSchemaGeneration';
import { useSchemaHistory } from './useSchemaHistory';
import {
  collectionSchemaReducer,
  createInitialCollectionSchemaState,
  type CollectionSchemaEditorState,
} from './collectionSchemaReducer';
import { definitionSource } from './collectionSchemaWorkspaceHelpers';
import {
  createInitialSchemaFormBufferState,
  previewReviewedRebase,
  schemaFormBufferReducer,
} from './schemaFormBuffer';
import type {
  SchemaDefinitionKind,
  SchemaHistoryVersion,
  ValidationIssue,
  ValidationResult,
} from './schemaTypes';

export interface UseCollectionSchemaEditorOptions {
  collectionId: string;
  collectionName?: string;
  api?: CollectionSchemaApi;
  registerDirtyState?: (registration: { isDirty: boolean; discard: () => void } | null) => void;
  requestSelectionChange?: (next: () => void) => void;
}

export function useCollectionSchemaEditor(options: UseCollectionSchemaEditorOptions) {
  const { collectionId, registerDirtyState, requestSelectionChange } = options;
  const api = useMemo(() => options.api ?? createDefaultCollectionSchemaApi(), [options.api]);
  const [editorState, dispatch] = useReducer(collectionSchemaReducer, undefined, createInitialCollectionSchemaState);
  const [formBuffer, dispatchForm] = useReducer(schemaFormBufferReducer, undefined, createInitialSchemaFormBufferState);
  const { history, historyLoading, historyError, onLoadHistory, onLoadMoreHistory } = useSchemaHistory(api, collectionId);
  const [restoreChallengeToken, setRestoreChallengeToken] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const requestGenerationRef = useRef(0);
  const editorStateRef = useRef(editorState);
  editorStateRef.current = editorState;
  const formBufferRef = useRef(formBuffer);
  formBufferRef.current = formBuffer;
  const validationResultRef = useRef<ValidationResult | null>(null);

  const nextGeneration = useCallback(() => {
    requestGenerationRef.current += 1;
    return requestGenerationRef.current;
  }, []);

  const reloadWorkspace = useCallback(
    async (generation: number) => {
      dispatch({ type: 'load/started', collectionId, requestGeneration: generation });
      const result = await api.loadWorkspace(collectionId);
      if (generation !== requestGenerationRef.current) return;
      if (!result.ok) {
        dispatch({ type: 'load/failed', kind: result.kind, requestGeneration: generation });
        return;
      }
      dispatch({ type: 'load/succeeded', envelope: result.data, requestGeneration: generation });
    },
    [api, collectionId],
  );

  const { generation, setGeneration, generationPollRef, onGenerateSchema } = useSchemaGeneration({
    api, collectionId, editorState, nextGeneration, requestGenerationRef, reloadWorkspace,
  });

  useEffect(() => {
    const generation = nextGeneration();
    generationPollRef.current?.cancel();
    generationPollRef.current = null;
    setGeneration(initialGenerationState);
    dispatch({ type: 'collection/changed', collectionId, requestGeneration: generation });
    void reloadWorkspace(generation);
  }, [collectionId, api, nextGeneration, reloadWorkspace]);

  useEffect(
    () => () => {
      requestGenerationRef.current += 1;
      generationPollRef.current?.cancel();
      generationPollRef.current = null;
    },
    [collectionId],
  );

  useEffect(() => {
    registerDirtyState?.({
      isDirty: formBuffer.open && formBuffer.dirtyFields.length > 0,
      discard: () => dispatchForm({ type: 'form/discard' }),
    });
    return () => registerDirtyState?.(null);
  }, [formBuffer.dirtyFields.length, formBuffer.open, registerDirtyState]);

  const openDefinition = useCallback(
    (kind: SchemaDefinitionKind, key: string) => {
      const apply = () => {
        const envelope = editorStateRef.current.envelope;
        if (!envelope) return;
        const source = definitionSource(envelope);
        const list = kind === 'entity' ? source.entities : source.relations;
        const definition = list.find((item) => item.key === key);
        if (!definition) return;
        dispatch({ type: 'selection/changed', selection: { kind, key } });
        dispatchForm({
          type: 'form/open',
          kind,
          key,
          baseRevision: envelope.draft?.revision ?? null,
          baseDraftId: envelope.draft?.draft_id ?? null,
          values: definitionValues(kind, definition),
        });
      };
      requestSelectionChange ? requestSelectionChange(apply) : apply();
    },
    [requestSelectionChange],
  );

  const runEnvelopeMutation = useCallback(
    async (
      generation: number,
      started: CollectionSchemaEditorState['pendingOperation'],
      perform: () => ReturnType<CollectionSchemaApi['createDraft']>,
    ) => {
      const startType =
        started === 'discard' ? 'discard/started' : started === 'restore' ? 'restore/started' : 'mutation/started';
      dispatch({ type: startType, requestGeneration: generation });
      const result = await perform();
      if (generation !== requestGenerationRef.current) return;
      if (!result.ok) {
        if (result.kind === 'revision_conflict' && result.conflict) {
          const reload = await api.loadWorkspace(collectionId);
          if (reload.ok) {
            dispatch({
              type: 'conflict/received',
              conflict: result.conflict,
              envelope: reload.data,
              requestGeneration: generation,
            });
          }
        } else {
          dispatch({ type: 'mutation/failed', kind: result.kind, requestGeneration: generation });
        }
        return;
      }
      const successType =
        started === 'discard' ? 'discard/succeeded' : started === 'restore' ? 'restore/succeeded' : 'mutation/succeeded';
      dispatch({ type: successType, envelope: result.data, requestGeneration: generation });
    },
    [api, collectionId],
  );

  const onCreateDraft = useCallback(async () => {
    await runEnvelopeMutation(nextGeneration(), 'mutation', () => api.createDraft(collectionId));
  }, [api, collectionId, nextGeneration, runEnvelopeMutation]);

  const onValidate = useCallback(async () => {
    const draft = editorStateRef.current.envelope?.draft;
    if (!draft) return;
    const generation = nextGeneration();
    dispatch({ type: 'validate/started', requestGeneration: generation });
    const result = await api.validate(collectionId, draft.draft_id, draft.revision);
    if (generation !== requestGenerationRef.current) return;
    if (!result.ok) {
      dispatch({ type: 'validate/failed', kind: result.kind, requestGeneration: generation });
      return;
    }
    validationResultRef.current = result.data;
    dispatch({ type: 'validate/succeeded', result: result.data, requestGeneration: generation });
  }, [api, collectionId, nextGeneration]);

  const onPublish = useCallback(async () => {
    const draft = editorStateRef.current.envelope?.draft;
    const validation = validationResultRef.current ?? editorStateRef.current.validation.result;
    if (!draft || !validation) return;
    const generation = nextGeneration();
    const operation = {
      draft_id: validation.identity.draft_id,
      revision: validation.identity.revision,
      candidate_checksum: validation.identity.candidate_checksum,
      validation_result_id: validation.identity.result_id,
    };
    dispatch({ type: 'publish/started', operation, requestGeneration: generation });
    const result = await api.publish(collectionId, operation, draft.revision);
    if (generation !== requestGenerationRef.current) return;
    if (!result.ok) {
      dispatch({ type: 'publish/failed', requestGeneration: generation });
      return;
    }
    dispatch({ type: 'publish/succeeded', envelope: result.data, requestGeneration: generation });
    setStatusMessage('Schema published successfully.');
  }, [api, collectionId, nextGeneration]);

  const onDiscardDraft = useCallback(async () => {
    const draft = editorStateRef.current.envelope?.draft;
    if (!draft) return;
    await runEnvelopeMutation(nextGeneration(), 'discard', () => api.discardDraft(collectionId, draft.draft_id, draft.revision));
  }, [api, collectionId, nextGeneration, runEnvelopeMutation]);

  const onSaveDefinition = useCallback(async () => {
    const draft = editorStateRef.current.envelope?.draft;
    const buffer = formBufferRef.current;
    if (!draft || !buffer.baseDraftId || buffer.baseRevision === null || !buffer.definitionKey || !buffer.definitionKind || !buffer.currentValues) return;
    const generation = nextGeneration();
    dispatchForm({ type: 'form/save/started' });
    const perform =
      buffer.definitionKind === 'entity'
        ? () => api.upsertEntity(collectionId, buffer.definitionKey!, buffer.baseDraftId!, buffer.baseRevision!, buffer.currentValues!)
        : () => api.upsertRelation(collectionId, buffer.definitionKey!, buffer.baseDraftId!, buffer.baseRevision!, buffer.currentValues!);
    dispatch({ type: 'mutation/started', requestGeneration: generation });
    const result = await perform();
    if (generation !== requestGenerationRef.current) return;
    if (!result.ok) {
      dispatchForm({
        type: 'form/save/rejected',
        conflictFields: result.conflict?.definitions.flatMap((item) => item.fields.map((field) => field.field)),
      });
      if (result.kind === 'revision_conflict' && result.conflict) {
        const reload = await api.loadWorkspace(collectionId);
        if (reload.ok) {
          dispatch({ type: 'conflict/received', conflict: result.conflict, envelope: reload.data, requestGeneration: generation });
        }
      } else {
        dispatch({ type: 'mutation/failed', kind: result.kind, requestGeneration: generation });
      }
      return;
    }
    dispatch({ type: 'mutation/succeeded', envelope: result.data, requestGeneration: generation });
    dispatchForm({
      type: 'form/save/succeeded',
      baseRevision: result.data.draft?.revision ?? draft.revision,
      values: buffer.currentValues,
    });
  }, [api, collectionId, nextGeneration]);

  const conflictPreview = useMemo(() => {
    if (!editorState.conflict || !formBuffer.initialValues || !formBuffer.currentValues || !editorState.envelope?.draft || !formBuffer.definitionKey) {
      return null;
    }
    const source = definitionSource(editorState.envelope);
    const list = formBuffer.definitionKind === 'entity' ? source.entities : source.relations;
    const latest = list.find((item) => item.key === formBuffer.definitionKey);
    return latest
      ? previewReviewedRebase(
          formBuffer.initialValues,
          formBuffer.currentValues,
          latest.values as unknown as Record<string, unknown>,
        )
      : null;
  }, [editorState.conflict, editorState.envelope, formBuffer]);

  const onConflictReapply = useCallback((resolutions: Parameters<typeof buildConflictReapplyUpdate>[2]) => {
    const update = buildConflictReapplyUpdate(editorStateRef.current, formBufferRef.current, resolutions);
    if (!update) return;
    dispatchForm({ type: 'form/reload', ...update.formReload });
    dispatch({ type: 'load/succeeded', envelope: update.envelope, requestGeneration: update.requestGeneration });
  }, []);

  return {
    editorState,
    formBuffer,
    history,
    historyLoading,
    historyError,
    conflictPreview,
    restoreChallengeToken,
    statusMessage,
    generation,
    onSelectDefinition: openDefinition,
    onCreateDraft,
    onValidate,
    onPublish,
    onDiscardDraft,
    onGenerateSchema,
    onFieldChange: (field: string, value: unknown) => dispatchForm({ type: 'form/edit', field, value }),
    onSaveDefinition,
    onRevertDefinition: () => dispatchForm({ type: 'form/revert' }),
    onCancelDefinition: () => dispatchForm({ type: 'form/close' }),
    onLoadHistory,
    onLoadMoreHistory,
    onRestoreVersion: (_version: SchemaHistoryVersion) => setRestoreChallengeToken(null),
    onConfirmRestore: async () => runEnvelopeMutation(nextGeneration(), 'restore', () => api.restoreVersion(collectionId, 4)),
    onConflictDiscard: () => dispatchForm({ type: 'form/discard' }),
    onConflictReapply,
    onIssueSelect: (_issue: ValidationIssue) => undefined,
  };
}

export type UseCollectionSchemaEditorResult = ReturnType<typeof useCollectionSchemaEditor>;
