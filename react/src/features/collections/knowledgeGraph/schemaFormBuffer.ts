import type { SchemaDefinitionKind } from './schemaTypes';

export interface SchemaFormBufferState {
  open: boolean;
  definitionKey: string | null;
  definitionKind: SchemaDefinitionKind | null;
  baseRevision: number | null;
  initialValues: Record<string, unknown> | null;
  currentValues: Record<string, unknown> | null;
  dirtyFields: string[];
  pending: boolean;
  conflictFields: string[];
}

export type SchemaFormBufferAction =
  | {
      type: 'form/open';
      kind: SchemaDefinitionKind;
      key: string;
      baseRevision: number | null;
      values: Record<string, unknown>;
    }
  | { type: 'form/edit'; field: string; value: unknown }
  | { type: 'form/revert' }
  | { type: 'form/save/started' }
  | {
      type: 'form/save/succeeded';
      baseRevision: number;
      values: Record<string, unknown>;
    }
  | { type: 'form/save/rejected'; conflictFields?: string[] }
  | { type: 'form/discard' }
  | {
      type: 'form/reload';
      baseRevision: number;
      values: Record<string, unknown>;
      preserveDirty?: boolean;
    }
  | { type: 'form/close' };

export interface RebaseFieldConflict {
  field: string;
  baseValue: unknown;
  localValue: unknown;
  serverValue: unknown;
}

export interface ReviewedRebasePreview {
  autoStaged: Record<string, unknown>;
  conflicts: RebaseFieldConflict[];
}

export type ReviewedRebaseResolution = {
  field: string;
  choice: 'local' | 'server';
};

export function createInitialSchemaFormBufferState(): SchemaFormBufferState {
  return {
    open: false,
    definitionKey: null,
    definitionKind: null,
    baseRevision: null,
    initialValues: null,
    currentValues: null,
    dirtyFields: [],
    pending: false,
    conflictFields: [],
  };
}

function valuesEqual(left: unknown, right: unknown): boolean {
  return JSON.stringify(left) === JSON.stringify(right);
}

function computeDirtyFields(
  initialValues: Record<string, unknown> | null,
  currentValues: Record<string, unknown> | null,
): string[] {
  if (!initialValues || !currentValues) return [];
  const fields = new Set([...Object.keys(initialValues), ...Object.keys(currentValues)]);
  return [...fields].filter((field) => !valuesEqual(initialValues[field], currentValues[field]));
}

export function getDirtyFields(state: SchemaFormBufferState): string[] {
  return computeDirtyFields(state.initialValues, state.currentValues);
}

export function schemaFormBufferReducer(
  state: SchemaFormBufferState,
  action: SchemaFormBufferAction,
): SchemaFormBufferState {
  switch (action.type) {
    case 'form/open':
      return {
        open: true,
        definitionKey: action.key,
        definitionKind: action.kind,
        baseRevision: action.baseRevision,
        initialValues: { ...action.values },
        currentValues: { ...action.values },
        dirtyFields: [],
        pending: false,
        conflictFields: [],
      };
    case 'form/edit': {
      if (!state.currentValues) return state;
      const currentValues = { ...state.currentValues, [action.field]: action.value };
      return {
        ...state,
        currentValues,
        dirtyFields: computeDirtyFields(state.initialValues, currentValues),
        conflictFields: [],
      };
    }
    case 'form/revert':
      if (!state.initialValues) return state;
      return {
        ...state,
        currentValues: { ...state.initialValues },
        dirtyFields: [],
        conflictFields: [],
      };
    case 'form/save/started':
      return { ...state, pending: true, conflictFields: [] };
    case 'form/save/succeeded':
      return {
        ...state,
        pending: false,
        baseRevision: action.baseRevision,
        initialValues: { ...action.values },
        currentValues: { ...action.values },
        dirtyFields: [],
        conflictFields: [],
      };
    case 'form/save/rejected':
      return {
        ...state,
        pending: false,
        conflictFields: action.conflictFields ?? [],
      };
    case 'form/discard':
    case 'form/close':
      return createInitialSchemaFormBufferState();
    case 'form/reload': {
      const initialValues = { ...action.values };
      if (action.preserveDirty && state.currentValues) {
        const dirtyFields = computeDirtyFields(state.initialValues, state.currentValues);
        const currentValues = { ...initialValues, ...pickFields(state.currentValues, dirtyFields) };
        return {
          ...state,
          baseRevision: action.baseRevision,
          initialValues,
          currentValues,
          dirtyFields: computeDirtyFields(initialValues, currentValues),
          conflictFields: [],
        };
      }
      return {
        ...state,
        baseRevision: action.baseRevision,
        initialValues,
        currentValues: initialValues,
        dirtyFields: [],
        conflictFields: [],
      };
    }
    default:
      return state;
  }
}

function pickFields(values: Record<string, unknown>, fields: string[]): Record<string, unknown> {
  return Object.fromEntries(fields.map((field) => [field, values[field]]));
}

export function previewReviewedRebase(
  initialValues: Record<string, unknown>,
  currentValues: Record<string, unknown>,
  latestValues: Record<string, unknown>,
): ReviewedRebasePreview {
  const fields = new Set([
    ...Object.keys(initialValues),
    ...Object.keys(currentValues),
    ...Object.keys(latestValues),
  ]);
  const autoStaged: Record<string, unknown> = {};
  const conflicts: RebaseFieldConflict[] = [];

  for (const field of fields) {
    const baseValue = initialValues[field];
    const localValue = field in currentValues ? currentValues[field] : baseValue;
    const serverValue = field in latestValues ? latestValues[field] : baseValue;
    const localChanged = !valuesEqual(localValue, baseValue);
    const serverChanged = !valuesEqual(serverValue, baseValue);

    if (localChanged && serverChanged) {
      conflicts.push({ field, baseValue, localValue, serverValue });
    } else if (localChanged) {
      autoStaged[field] = localValue;
    }
  }

  return { autoStaged, conflicts };
}

export function applyReviewedRebase(
  preview: ReviewedRebasePreview,
  resolutions: ReviewedRebaseResolution[],
  latestRevision: number,
  latestValues: Record<string, unknown>,
): { values: Record<string, unknown>; baseRevision: number } {
  const resolutionByField = new Map(resolutions.map((resolution) => [resolution.field, resolution.choice]));
  if (preview.conflicts.some((conflict) => !resolutionByField.has(conflict.field))) {
    throw new Error('All conflicting fields require an explicit resolution');
  }

  const values = { ...latestValues, ...preview.autoStaged };
  for (const conflict of preview.conflicts) {
    const choice = resolutionByField.get(conflict.field);
    values[conflict.field] = choice === 'local' ? conflict.localValue : conflict.serverValue;
  }

  return { values, baseRevision: latestRevision };
}
