import React from 'react';
import type {
  DraftSnapshot,
  PublishedSchemaSnapshot,
  SchemaPermissionsSnapshot,
  PublishStatus,
  SchemaGenerationState,
  ValidationResult,
} from './schemaTypes';
import { buttonDangerClass, buttonPrimaryClass, buttonSecondaryClass, panelClass } from './schemaUiShared';
import { isSchemaGenerationEligibleDraft } from './schemaGenerationEligibility';

export interface SchemaDraftToolbarProps {
  collectionName: string;
  permissions: SchemaPermissionsSnapshot;
  published: PublishedSchemaSnapshot;
  draft: DraftSnapshot | null;
  dirty: boolean;
  pendingOperation: string | null;
  validationStatus: 'idle' | 'pending' | 'valid' | 'invalid';
  validationResult: ValidationResult | null;
  publishStatus: PublishStatus;
  projectionStatusLabel?: string | null;
  canValidate: boolean;
  canPublish: boolean;
  generation?: SchemaGenerationState;
  onCreateDraft?: () => void;
  onValidate?: () => void;
  onPublish?: () => void;
  onDiscard?: () => void;
  onShowDiff?: () => void;
  onShowHistory?: () => void;
  onGenerate?: () => void;
}

const SchemaDraftToolbar: React.FC<SchemaDraftToolbarProps> = ({
  collectionName,
  permissions,
  published,
  draft,
  dirty,
  pendingOperation,
  validationStatus,
  validationResult,
  publishStatus,
  projectionStatusLabel,
  canValidate,
  canPublish,
  generation = { status: 'idle' },
  onCreateDraft,
  onValidate,
  onPublish,
  onDiscard,
  onShowDiff,
  onShowHistory,
  onGenerate,
}) => {
  const level = permissions.level;
  const readOnlyMessage =
    level === 'VIEW'
      ? 'View-only access: published schema inspection only.'
      : !draft
        ? 'No active draft. Create a draft to begin editing.'
        : null;
  const generationBusy = generation.status === 'starting' || generation.status === 'queued' || generation.status === 'running';
  const generationEligible = isSchemaGenerationEligibleDraft(draft);
  const generationMessage =
    generation.status === 'starting' || generation.status === 'running'
      ? 'Generating schema from collection.'
      : generation.status === 'queued'
        ? 'Schema generation queued.'
        : generation.status === 'succeeded'
          ? 'Schema generation completed.'
          : generation.status === 'failed'
            ? `Schema generation failed${generation.errorCode ? `: ${generation.errorCode}` : ''}.`
            : null;

  return (
    <section
      className={`${panelClass} p-[14px] mb-3`}
      aria-label="Schema draft toolbar"
      data-permission-level={level}
    >
      <div className="flex flex-wrap gap-3 justify-between items-start">
        <div className="space-y-1 text-sm">
          <p>
            <span className="text-text-lower_contrast">Collection:</span> {collectionName}
          </p>
          <p>
            <span className="text-text-lower_contrast">Published:</span> v{published.version} ·{' '}
            {published.checksum}
          </p>
          {draft ? (
            <>
              <p>
                <span className="text-text-lower_contrast">Draft:</span> {draft.draft_id} · revision{' '}
                {draft.revision}
              </p>
              <p>
                <span className="text-text-lower_contrast">Last editor:</span> {draft.last_editor} ·{' '}
                {draft.updated_at}
              </p>
            </>
          ) : null}
          {dirty ? <p className="text-amber-200">Unsaved form changes</p> : null}
          {pendingOperation ? <p className="text-text-slightly_less_contrast">Request pending: {pendingOperation}</p> : null}
          <p>
            <span className="text-text-lower_contrast">Validation:</span> {validationStatus}
            {validationResult ? ` · result ${validationResult.identity.result_id}` : ''}
          </p>
          {publishStatus !== 'idle' ? (
            <p>
              <span className="text-text-lower_contrast">Publish:</span> {publishStatus}
            </p>
          ) : null}
          {projectionStatusLabel ? (
            <p>
              <span className="text-text-lower_contrast">Projection:</span> {projectionStatusLabel}
            </p>
          ) : null}
          {generationMessage ? <p aria-live="polite">{generationMessage}</p> : null}
          {readOnlyMessage ? <p className="text-text-lower_contrast">{readOnlyMessage}</p> : null}
        </div>

        <div className="flex flex-wrap gap-2">
          {permissions.can_edit_definitions && generationEligible && onGenerate ? (
            <button type="button" className={buttonSecondaryClass} disabled={generationBusy} onClick={onGenerate}>
              {generation.status === 'failed' ? 'Retry generation' : 'Generate from collection'}
            </button>
          ) : null}
          {permissions.can_create_draft && !draft ? (
            <button type="button" className={buttonPrimaryClass} onClick={onCreateDraft}>
              Create draft
            </button>
          ) : null}
          {permissions.can_validate && draft ? (
            <button type="button" className={buttonSecondaryClass} disabled={!canValidate} onClick={onValidate}>
              Validate
            </button>
          ) : null}
          {permissions.can_publish && draft ? (
            <button type="button" className={buttonPrimaryClass} disabled={!canPublish} onClick={onPublish}>
              Publish
            </button>
          ) : null}
          {permissions.can_discard_draft && draft ? (
            <button type="button" className={buttonDangerClass} onClick={onDiscard}>
              Discard draft
            </button>
          ) : null}
          {onShowDiff ? (
            <button type="button" className={buttonSecondaryClass} onClick={onShowDiff}>
              Review diff
            </button>
          ) : null}
          {permissions.can_view_history && onShowHistory ? (
            <button type="button" className={buttonSecondaryClass} onClick={onShowHistory}>
              History
            </button>
          ) : null}
        </div>
      </div>
    </section>
  );
};

export default SchemaDraftToolbar;
