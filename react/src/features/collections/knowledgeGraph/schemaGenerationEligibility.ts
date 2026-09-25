import type { DraftSnapshot } from './schemaTypes';

export function isSchemaGenerationEligibleDraft(draft: DraftSnapshot | null): boolean {
  return draft === null || (draft.entities.length === 0 && draft.relations.length === 0);
}
