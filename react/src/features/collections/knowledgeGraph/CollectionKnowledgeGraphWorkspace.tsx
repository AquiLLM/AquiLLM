import React, { useEffect, useState } from 'react';
import type { CollectionSchemaEditorState } from './collectionSchemaReducer';
import { canPublish, canValidate, selectedDefinition } from './collectionSchemaReducer';
import CollectionKnowledgeGraphWorkspaceDialogs from './CollectionKnowledgeGraphWorkspaceDialogs';
import CollectionSchemaNavigation from './CollectionSchemaNavigation';
import { definitionSource, workspacePhaseMessage } from './collectionSchemaWorkspaceHelpers';
import EntityTypeEditor from './EntityTypeEditor';
import RelationTypeEditor from './RelationTypeEditor';
import SchemaDraftToolbar from './SchemaDraftToolbar';
import SchemaHistoryPanel from './SchemaHistoryPanel';
import SchemaValidationPanel from './SchemaValidationPanel';
import type { ReviewedRebasePreview, ReviewedRebaseResolution, SchemaFormBufferState } from './schemaFormBuffer';
import type {
  SchemaDefinitionKind,
  SchemaHistoryPage,
  SchemaHistoryVersion,
  ValidationIssue,
} from './schemaTypes';
import { panelClass } from './schemaUiShared';

export interface CollectionKnowledgeGraphWorkspaceProps {
  collectionId: string;
  collectionName: string;
  initialCanEdit: boolean;
  initialCanManage: boolean;
  editorState: CollectionSchemaEditorState;
  formBuffer: SchemaFormBufferState;
  history: SchemaHistoryPage | null;
  historyLoading: boolean;
  historyError: string | null;
  conflictPreview: ReviewedRebasePreview | null;
  restoreChallengeToken?: string | null;
  statusMessage?: string | null;
  projectionStatusLabel?: string | null;
  impactSummary?: string | null;
  onSelectDefinition: (kind: SchemaDefinitionKind, key: string) => void;
  onCreateDraft?: () => void;
  onValidate?: () => void;
  onPublish?: () => void;
  onDiscardDraft?: () => void;
  onFieldChange: (field: string, value: unknown) => void;
  onSaveDefinition?: () => void;
  onRevertDefinition?: () => void;
  onRemoveDefinition?: () => void;
  onCancelDefinition?: () => void;
  onAddEntity?: () => void;
  onAddRelation?: () => void;
  onIssueSelect?: (issue: ValidationIssue) => void;
  onLoadHistory?: () => void;
  onLoadMoreHistory?: () => void;
  onRestoreVersion?: (version: SchemaHistoryVersion) => void;
  onConfirmRestore?: () => void;
  onConflictDiscard?: () => void;
  onConflictReapply?: (resolutions: ReviewedRebaseResolution[]) => void;
}

const CollectionKnowledgeGraphWorkspace: React.FC<CollectionKnowledgeGraphWorkspaceProps> = (props) => {
  const {
    collectionId,
    collectionName,
    initialCanEdit,
    initialCanManage,
    editorState,
    formBuffer,
    history,
    historyLoading,
    historyError,
    conflictPreview,
    restoreChallengeToken,
    statusMessage,
    projectionStatusLabel,
    impactSummary,
    onSelectDefinition,
    onCreateDraft,
    onValidate,
    onPublish,
    onDiscardDraft,
    onFieldChange,
    onSaveDefinition,
    onRevertDefinition,
    onRemoveDefinition,
    onCancelDefinition,
    onAddEntity,
    onAddRelation,
    onIssueSelect,
    onLoadHistory,
    onLoadMoreHistory,
    onRestoreVersion,
    onConfirmRestore,
    onConflictDiscard,
    onConflictReapply,
  } = props;

  const [diffOpen, setDiffOpen] = useState(false);
  const [publishOpen, setPublishOpen] = useState(false);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [restoreOpen, setRestoreOpen] = useState(false);
  const [selectedHistory, setSelectedHistory] = useState<SchemaHistoryVersion | null>(null);
  const [activeIssueKey, setActiveIssueKey] = useState<string | null>(null);

  const { phase, envelope, selection, validation, publish, pendingOperation } = editorState;
  const selected = selectedDefinition(editorState);
  const source = envelope ? definitionSource(envelope) : null;

  useEffect(() => {
    if (historyOpen) onLoadHistory?.();
  }, [historyOpen, onLoadHistory]);

  const phaseMessage = workspacePhaseMessage(phase);
  if (phaseMessage) {
    return (
      <div className={`${panelClass} p-[16px]`} data-testid="schema-workspace-status">
        <p>{phaseMessage}</p>
        {phase === 'loading' ? (
          <p className="text-sm text-text-lower_contrast">
            Initial edit hint: {initialCanEdit ? 'edit expected' : 'view expected'}
            {initialCanManage ? ', manage expected' : ''}
          </p>
        ) : null}
      </div>
    );
  }

  if (!envelope || !source) {
    return (
      <div className={`${panelClass} p-[16px]`} data-testid="schema-workspace-empty">
        No schema envelope loaded.
      </div>
    );
  }

  const permissions = envelope.permissions;
  const showMutationControls = permissions.can_edit_definitions;
  const formValues = formBuffer.currentValues ?? {};
  const formDirty = formBuffer.dirtyFields.length > 0;
  const readOnlyEditor = !showMutationControls || !formBuffer.open;

  return (
    <div data-testid="collection-knowledge-graph-workspace" data-collection-id={collectionId}>
      <div aria-live="polite" className="sr-only">
        {statusMessage ?? ''}
      </div>

      <SchemaDraftToolbar
        collectionName={collectionName}
        permissions={permissions}
        published={envelope.published}
        draft={envelope.draft}
        dirty={formDirty}
        pendingOperation={pendingOperation}
        validationStatus={validation.status}
        validationResult={validation.result}
        publishStatus={publish.status}
        projectionStatusLabel={projectionStatusLabel}
        canValidate={canValidate(editorState)}
        canPublish={canPublish(editorState)}
        onCreateDraft={onCreateDraft}
        onValidate={onValidate}
        onPublish={() => setPublishOpen(true)}
        onDiscard={onDiscardDraft}
        onShowDiff={() => setDiffOpen(true)}
        onShowHistory={() => setHistoryOpen(true)}
      />

      <div className="grid grid-cols-1 lg:grid-cols-[320px_minmax(0,1fr)] gap-3">
        <CollectionSchemaNavigation
          entities={source.entities}
          relations={source.relations}
          selectedKind={selection?.kind ?? null}
          selectedKey={selection?.key ?? null}
          onSelect={onSelectDefinition}
          canAddEntity={showMutationControls}
          canAddRelation={showMutationControls}
          onAddEntity={onAddEntity}
          onAddRelation={onAddRelation}
        />

        <div className="space-y-3">
          {selected?.kind === 'entity' && formBuffer.open ? (
            <EntityTypeEditor
              collectionName={collectionName}
              draftRevision={envelope.draft?.revision ?? null}
              definition={selected.definition}
              constraints={envelope.constraints}
              values={formValues}
              dirty={formDirty}
              pending={formBuffer.pending}
              readOnly={readOnlyEditor}
              conflictFields={formBuffer.conflictFields}
              onFieldChange={onFieldChange}
              onSave={() => onSaveDefinition?.()}
              onRevert={() => onRevertDefinition?.()}
              onRemove={onRemoveDefinition}
              onCancel={onCancelDefinition}
            />
          ) : null}
          {selected?.kind === 'relation' && formBuffer.open ? (
            <RelationTypeEditor
              collectionName={collectionName}
              draftRevision={envelope.draft?.revision ?? null}
              definition={selected.definition}
              entityTypes={source.entities}
              constraints={envelope.constraints}
              values={formValues}
              dirty={formDirty}
              pending={formBuffer.pending}
              readOnly={readOnlyEditor}
              conflictFields={formBuffer.conflictFields}
              onFieldChange={onFieldChange}
              onSave={() => onSaveDefinition?.()}
              onRevert={() => onRevertDefinition?.()}
              onRemove={onRemoveDefinition}
              onCancel={onCancelDefinition}
            />
          ) : null}
          {!selected || !formBuffer.open ? (
            <div className={`${panelClass} p-[16px] text-sm text-text-lower_contrast`}>
              Select a definition to inspect or edit schema details.
            </div>
          ) : null}

          <SchemaValidationPanel
            status={validation.status}
            result={validation.result}
            activeIssueKey={activeIssueKey}
            onIssueSelect={(issue) => {
              setActiveIssueKey(`${issue.severity}:${issue.code}:${issue.location}`);
              onIssueSelect?.(issue);
            }}
          />

          {historyOpen ? (
            <SchemaHistoryPanel
              permissions={permissions}
              history={history}
              loading={historyLoading}
              error={historyError}
              selectedVersion={selectedHistory}
              onSelectVersion={setSelectedHistory}
              onLoadMore={onLoadMoreHistory}
              onRestore={(version) => {
                setSelectedHistory(version);
                setRestoreOpen(true);
                onRestoreVersion?.(version);
              }}
            />
          ) : null}
        </div>
      </div>

      <CollectionKnowledgeGraphWorkspaceDialogs
        collectionName={collectionName}
        draft={envelope.draft}
        editorState={editorState}
        diffOpen={diffOpen}
        publishOpen={publishOpen}
        restoreOpen={restoreOpen}
        selectedHistory={selectedHistory}
        conflictPreview={conflictPreview}
        restoreChallengeToken={restoreChallengeToken}
        impactSummary={impactSummary}
        onCloseDiff={() => setDiffOpen(false)}
        onClosePublish={() => setPublishOpen(false)}
        onCloseRestore={() => setRestoreOpen(false)}
        onPublish={onPublish}
        onConfirmRestore={onConfirmRestore}
        onConflictDiscard={onConflictDiscard}
        onConflictReapply={onConflictReapply}
      />
    </div>
  );
};

export default CollectionKnowledgeGraphWorkspace;
