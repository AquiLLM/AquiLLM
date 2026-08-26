import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from 'react';
import {
  buildConflictReapplyUpdate,
  createDefaultCollectionSchemaApi,
  definitionValues,
} from './collectionSchemaEditorHelpers';
import type { CollectionSchemaApi } from './collectionSchemaApi';
import { createGenerationPollController, type GenerationPollController } from './schemaGenerationPolling';
import { isSchemaGenerationEligibleDraft } from './schemaGenerationEligibility';
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
  SchemaGenerationState,
  SchemaHistoryPage,
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

const initialGenerationState: SchemaGenerationState = { status: 'idle' };

export function useCollectionSchemaEditor(options: UseCollectionSchemaEditorOptions) {
  const { collectionId, registerDirtyState, requestSelectionChange } = options;
  const api = useMemo(() => options.api ?? createDefaultCollectionSchemaApi(), [options.api]);
  const [editorState, dispatch] = useReducer(collectionSchemaReducer, undefined, createInitialCollectionSchemaState);
  const [formBuffer, dispatchForm] = useReducer(schemaFormBufferReducer, undefined, createInitialSchemaFormBufferState);
  const [history, setHistory] = useState<SchemaHistoryPage | null>(null);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [restoreChallengeToken, setRestoreChallengeToken] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [generation, setGeneration] = useState<SchemaGenerationState>(initialGenerationState);
  const requestGenerationRef = useRef(0);
  const automaticGenerationCollectionsRef = useRef(new Set<string>());
  const generationPollRef = useRef<GenerationPollController | null>(null);
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

  const onGenerateSchema = useCallback(async () => {
    generationPollRef.current?.cancel();
    const generationRequest = nextGeneration();
    setGeneration({ status: 'starting' });
    const started = await api.startGeneration(collectionId);
    if (generationRequest !== requestGenerationRef.current) return;
    if (!started.ok) {
      setGeneration({ status: 'failed', errorCode: started.kind });
      return;
    }

    const { run_id: runId, status } = started.data;
    setGeneration({ status, runId });
    if (status === 'failed') {
      setGeneration({ status: 'failed', runId });
      return;
    }
    if (status === 'succeeded') {
      setGeneration({ status: 'succeeded', runId });
      await reloadWorkspace(generationRequest);
      return;
    }

    const controller = createGenerationPollController({
      poll: async (signal) => {
        const result = await api.getGenerationStatus(collectionId, runId, signal);
        if (!result.ok) {
          return { run_id: runId, status: 'failed', error_code: result.kind, statistics: {} };
        }
        if (generationRequest === requestGenerationRef.current && !signal?.aborted) {
          setGeneration({
            status: result.data.status,
            runId: result.data.run_id,
            errorCode: result.data.error_code,
            statistics: result.data.statistics,
          });
        }
        return result.data;
      },
    });
    generationPollRef.current = controller;
    const outcome = await controller.promise;
    if (generationRequest !== requestGenerationRef.current) return;
    if (outcome.status === 'cancelled') return;
    if (outcome.status === 'exhausted') {
      setGeneration({ status: 'failed', runId, errorCode: 'polling_exhausted' });
      return;
    }
    if (!('run_id' in outcome)) return;
    setGeneration({
      status: outcome.status,
      runId: outcome.run_id,
      errorCode: outcome.error_code,
      statistics: outcome.statistics,
    });
    if (outcome.status === 'succeeded') {
      await reloadWorkspace(generationRequest);
    }
  }, [api, collectionId, nextGeneration, reloadWorkspace]);

  useEffect(() => {
    const envelope = editorState.envelope;
    const canAutoGenerate =
      editorState.phase === 'ready' &&
      envelope?.collection_id === collectionId &&
      envelope.published.version === 0 &&
      isSchemaGenerationEligibleDraft(envelope.draft) &&
      envelope.permissions.can_edit_definitions;
    if (!canAutoGenerate || automaticGenerationCollectionsRef.current.has(collectionId)) return;
    automaticGenerationCollectionsRef.current.add(collectionId);
    void onGenerateSchema();
  }, [collectionId, editorState.envelope, editorState.phase, onGenerateSchema]);

  const onSaveDefinition = useCallback(async () => {
    const draft = editorStateRef.current.envelope?.draft;
    const buffer = formBufferRef.current;
    if (!draft || !buffer.definitionKey || !buffer.definitionKind || !buffer.currentValues) return;
    const generation = nextGeneration();
    dispatchForm({ type: 'form/save/started' });
    const perform =
      buffer.definitionKind === 'entity'
        ? () => api.upsertEntity(collectionId, buffer.definitionKey!, draft.revision, buffer.currentValues!)
        : () => api.upsertRelation(collectionId, buffer.definitionKey!, draft.revision, buffer.currentValues!);
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

  const onLoadHistory = useCallback(async () => {
    setHistoryLoading(true);
    setHistoryError(null);
    const result = await api.listVersions(collectionId);
    setHistoryLoading(false);
    if (!result.ok) {
      setHistoryError('Unable to load schema history.');
      return;
    }
    setHistory(result.data);
  }, [api, collectionId]);

  const onLoadMoreHistory = useCallback(async () => {
    if (!history?.next_cursor) return;
    setHistoryLoading(true);
    const result = await api.listVersions(collectionId, history.next_cursor);
    setHistoryLoading(false);
    if (!result.ok) {
      setHistoryError('Unable to load more schema history.');
      return;
    }
    setHistory({
      versions: [...history.versions, ...result.data.versions],
      next_cursor: result.data.next_cursor,
      has_more: result.data.has_more,
    });
  }, [api, collectionId, history]);

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
