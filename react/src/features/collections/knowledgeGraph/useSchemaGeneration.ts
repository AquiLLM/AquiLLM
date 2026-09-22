import { useCallback, useEffect, useRef, useState } from 'react';
import type { MutableRefObject } from 'react';
import type { CollectionSchemaApi } from './collectionSchemaApi';
import type { CollectionSchemaEditorState } from './collectionSchemaReducer';
import { isSchemaGenerationEligibleDraft } from './schemaGenerationEligibility';
import { createGenerationPollController, type GenerationPollController } from './schemaGenerationPolling';
import type { SchemaGenerationState } from './schemaTypes';

export const initialGenerationState: SchemaGenerationState = { status: 'idle' };

interface SchemaGenerationOptions {
  api: CollectionSchemaApi;
  collectionId: string;
  editorState: CollectionSchemaEditorState;
  nextGeneration: () => number;
  requestGenerationRef: MutableRefObject<number>;
  reloadWorkspace: (generation: number) => Promise<void>;
}

export function useSchemaGeneration({
  api, collectionId, editorState, nextGeneration, requestGenerationRef, reloadWorkspace,
}: SchemaGenerationOptions) {
  const [generation, setGeneration] = useState<SchemaGenerationState>(initialGenerationState);
  const generationPollRef = useRef<GenerationPollController | null>(null);
  const automaticGenerationCollectionsRef = useRef(new Set<string>());

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
  }, [api, collectionId, nextGeneration, reloadWorkspace, requestGenerationRef]);

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

  return { generation, setGeneration, generationPollRef, onGenerateSchema };
}
